"""
Async LLM client for the TRL vllm-serve API.

TRL's server (trl vllm-serve) differs from the standard OpenAI-compatible
vllm serve in three important ways:

  1. Health:      GET  /health/    — returns {"status": "ok"}
  2. Completions: POST /chat/      — takes ALL N conversations in ONE request
                                    (messages: list[list[dict]])
  3. Response:    returns completion_ids (token IDs), not decoded text.
                  The client must decode them with the model's tokenizer.

This design is intentional — TRL batches everything server-side for
maximum GPU utilisation, and the training loop already has a tokenizer
loaded, so decoding is cheap.
"""

import json
import re
import time
from typing import Optional

import httpx
from transformers import AutoTokenizer

from sim.schemas import SYSTEM_PROMPT, DroneAction, _ZERO_ACTION


# ── Action parser (unchanged from original) ──────────────────────────────────

def _parse_action(content: str) -> dict:
    """
    Multi-stage parser. Tries in order:
      1. Direct parse — {"force_x": ..., "force_y": ..., "torque_z": ...}
      2. Unwrap one level of nesting — {"action": {"force_x": ...}}
      3. Extract any JSON object from the string (regex)
      4. Zero action fallback
    """
    # Stage 1: direct parse + Pydantic validation
    try:
        obj = json.loads(content)
        if isinstance(obj, dict):
            if all(k in obj for k in ("force_x", "force_y", "torque_z")):
                return DroneAction.model_validate(obj).model_dump()
            for v in obj.values():
                if isinstance(v, dict) and all(k in v for k in ("force_x", "force_y", "torque_z")):
                    return DroneAction.model_validate(v).model_dump()
    except (json.JSONDecodeError, Exception):
        pass

    # Stage 2: regex — find any {...} block in the string and try each
    candidates = re.findall(r'\{[^{}]+\}', content)
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict) and all(k in obj for k in ("force_x", "force_y", "torque_z")):
                return DroneAction.model_validate(obj).model_dump()
        except Exception:
            pass

    # Stage 3: zero action — parse truly failed
    return dict(_ZERO_ACTION)


# ── LLM Client ────────────────────────────────────────────────────────────────

class LLMClient:
    """
    Manages async communication with a running trl vllm-serve server.

    Usage:
        client = LLMClient(model="Qwen/Qwen2.5-1.5B-Instruct")
        actions = await client.query_batch(obs_list)
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8000",
        max_tokens: int = 64,
        temperature: float = 0.7,
        timeout: float = 60.0,
    ):
        self.model       = model
        self.base_url    = base_url.rstrip("/")
        self.max_tokens  = max_tokens
        self.temperature = temperature
        self.endpoint    = f"{self.base_url}/chat/"

        # Persistent connection pool
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(
                max_connections=10,
                max_keepalive_connections=5,
            ),
        )

        # Tokenizer — loaded lazily on first query_batch call.
        # Needed to decode token IDs returned by /chat/ back to text.
        self._tokenizer: Optional[AutoTokenizer] = None

        self._last_latency_ms: float = 0.0
        self._total_calls:     int   = 0

    def _get_tokenizer(self) -> AutoTokenizer:
        if self._tokenizer is None:
            print(f"[LLMClient] Loading tokenizer for {self.model}...")
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model, trust_remote_code=True
            )
            print("[LLMClient] Tokenizer ready.")
        return self._tokenizer

    async def query_batch(
        self,
        observations: list[dict],
        system_prompt: Optional[str] = None,
    ) -> list[dict]:
        """
        Send all N observations in ONE batched request to trl vllm-serve
        and return parsed action dicts.

        The /chat/ endpoint accepts messages: list[list[dict]] — a list of
        N separate conversations. It returns completion_ids: list[list[int]],
        which we decode with the tokenizer and then parse as JSON actions.

        Args:
            observations: list of obs dicts from ParallelEnv (one per env).
            system_prompt: override the default system prompt if needed.

        Returns:
            list of action dicts, same length as observations.
            Malformed responses fall back to zero-action gracefully.
        """
        sys_prompt = system_prompt or SYSTEM_PROMPT

        # Build one conversation per env
        messages_batch = [
            [
                {"role": "system", "content": sys_prompt},
                {"role": "user",   "content": json.dumps(obs)},
            ]
            for obs in observations
        ]

        payload = {
            "messages":    messages_batch,
            "max_tokens":  self.max_tokens,
            "temperature": self.temperature,
            "logprobs":    None,   # don't need logprobs for sim
        }

        t0 = time.monotonic()
        try:
            response = await self._http.post(self.endpoint, json=payload)
            response.raise_for_status()
        except Exception as e:
            print(f"[LLMClient] Batch request failed: {type(e).__name__}: {e}")
            return [dict(_ZERO_ACTION)] * len(observations)

        self._last_latency_ms = (time.monotonic() - t0) * 1000
        self._total_calls += 1

        data        = response.json()
        tokenizer   = self._get_tokenizer()
        actions     = []

        for i, completion_ids in enumerate(data.get("completion_ids", [])):
            try:
                text = tokenizer.decode(completion_ids, skip_special_tokens=True)
                actions.append(_parse_action(text))
            except Exception as e:
                print(f"[LLMClient] Env {i} decode/parse error: {e}")
                actions.append(dict(_ZERO_ACTION))

        # Safety: pad to expected length if server returned fewer completions
        while len(actions) < len(observations):
            actions.append(dict(_ZERO_ACTION))

        return actions

    @property
    def last_latency_ms(self) -> float:
        """Latency of the most recent batch call in milliseconds."""
        return self._last_latency_ms

    async def health_check(self) -> bool:
        """
        Verify the trl vllm-serve server is reachable.
        Uses GET /health/ (the TRL server does not expose /v1/models).
        Returns True if healthy, False otherwise.
        """
        try:
            resp = await self._http.get(
                f"{self.base_url}/health/", timeout=5.0
            )
            resp.raise_for_status()
            print(f"[LLMClient] Health check OK — server is running at {self.base_url}")
            return True
        except Exception as e:
            print(f"[LLMClient] Health check FAILED: {e}")
            return False

    async def close(self):
        """Close the HTTP connection pool."""
        await self._http.aclose()

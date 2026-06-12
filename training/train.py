"""
GRPO training script for the 2D drone navigator.

Architecture (colocate mode, bfloat16 LoRA):
  - vLLM runs inside the same process as GRPOTrainer (no separate server)
  - Model weights are bfloat16 — no 4-bit quantization
  - Why not QLoRA? vLLM generates with merged-bfloat16 weights, but QLoRA
    evaluates log-probs with 4-bit weights.  The quantisation gap makes
    importance_sampling_ratio → 0, killing all GRPO gradients.
  - Our sim_reward() function is the only custom piece

Memory budget on 12 GB GPU (bfloat16 LoRA, sleep mode ON):
  - bfloat16 model:     ~3.0 GB  (shared between trainer and vLLM)
  - LoRA + optimiser:   ~0.5 GB
  - vLLM KV cache:      ~0.5 GB  (released during backward via sleep mode)
  - Activations (b=8):  ~0.3 GB
  - Total peak:         ~4.3 GB  (leaves 7.3 GB headroom)

OOM protection:
  - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  (set below, before torch import)
    Prevents CUDA memory fragmentation that causes spurious OOMs after many steps.
  - vllm_enable_sleep_mode=True releases KV cache during backward pass.
  - per_device_train_batch_size=8 (not 16) halves peak activation memory.

Policy stability (after observing ISR collapse at step ~2000):
  - learning_rate lowered from 5e-6 → 2e-6 to slow policy drift.
  - beta=0.04 KL penalty added to keep π_new close to sampling π_old.
"""

import os

# ── MUST be set before importing torch ───────────────────────────────────────
# Switches PyTorch's CUDA allocator to use expandable virtual memory segments.
# Eliminates memory fragmentation that builds up over hundreds of training steps
# and causes OOM even when total free memory exceeds the allocation size.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import Dataset
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from trl import GRPOTrainer, GRPOConfig

from sim.schemas import SYSTEM_PROMPT, DRONE_ACTION_REGEX
from sim.parallel_env import ParallelEnv
from sim.tasks.drone_2d import Drone2DEnv
from training.reward_fn import sim_reward, init_reward_env, close_reward_env
from training.metrics_callback import JSONLMetricsCallback


# Config 

MODEL_NAME        = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR        = "./checkpoints/drone_grpo"

# ── Evaluation overrides (set via environment variables in run.sh) ────────────
# TRAIN_EPOCHS=1        → run 1 epoch instead of 4  (fast demo ≈ 6 min)
# SAMPLES_PER_ENV=5     → 5×16=80 prompts  (vs. 25×16=400 for full run)
# These only affect the evaluation pipeline; the model architecture is unchanged.
_TRAIN_EPOCHS      = int(os.environ.get("TRAIN_EPOCHS", "4"))
_SAMPLES_PER_ENV   = int(os.environ.get("SAMPLES_PER_ENV", "25"))

NUM_ENVS          = 16   # physics envs for reward_fn — must equal NUM_GENERATIONS
                         # so reward_fn can evaluate every completion in its own env
NUM_GENERATIONS   = 16   # GRPO group size — doubled from 32 to halve advantage-mean
                         # standard error (1/√32 → 1/√64).  With reward_std≈1.0 the
                         # SEM dominated the gradient signal at 32 samples (80% noise),
                         # making policy updates chaotic without improving reward.
                         # must equal per_device_train_batch_size × gradient_accumulation_steps


# Step 1: Build the prompt dataset 
# GRPOTrainer expects a HuggingFace Dataset with a "prompt" column.
# Each prompt is the "question" the model answers with an action.
# We generate prompts by resetting real environments and serialising their obs.

def build_prompt_dataset(
    num_envs: int = NUM_ENVS,
    samples_per_env: int = 50,
    max_goal_dist: float = 3.0,
) -> Dataset:
    """
    Generate a dataset of initial observations from fresh environment resets.
    Each sample = one system+user prompt pair that the model must respond to.

    Run-10 curriculum:
      max_goal_dist=3.0 restricts goal placement to 2–3 m from the drone
      spawn.  With 80 rollout steps and a 4× progress multiplier, even a
      ~35%-efficient early policy can reach a 2–3 m goal, making the +10
      goal bonus fire from the first training steps.  This gives GRPO a
      clear success signal immediately rather than waiting hundreds of steps
      for the drone to stumble near a far-away goal.

      50 per env × 64 envs = 3200 prompts (all with close goals).
    """
    import json

    print(f"[dataset] Generating prompts  max_goal_dist={max_goal_dist}m ...")
    penv = ParallelEnv(Drone2DEnv, num_envs, env_kwargs={"max_goal_dist": max_goal_dist})

    prompts = []
    for _ in range(samples_per_env):
        obs_list = penv.reset_all()
        for obs in obs_list:
            # Format as a chat prompt — same format the model sees during rollout
            prompt = (
                f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{json.dumps(obs)}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            prompts.append({"prompt": prompt})

    penv.close()
    dataset = Dataset.from_list(prompts)
    print(f"[dataset] Dataset ready: {len(dataset)} samples  (goals within {max_goal_dist} m)")
    return dataset


# Step 2: bfloat16 LoRA model setup 

def load_lora_model(model_name: str):
    """
    Load the base model in bfloat16 with LoRA adapters — NO 4-bit quantization.

    Why not QLoRA here?
    GRPO's colocate mode computes importance_sampling_ratio = P_train / P_vllm.
    vLLM generates with the MERGED bfloat16 weights; the training model evaluates
    log-probabilities with the 4-bit quantized model.  The quantization error
    makes P_train ≈ 365× smaller than P_vllm (log-diff ~5.9 at step 1),
    causing importance_sampling_ratio → 0, loss → 0, grad_norm → 0 — no learning.

    bfloat16 LoRA memory budget on 12 GB GPU:
      Model weights (shared with vLLM):  ~3.0 GB
      vLLM KV cache:                     ~0.5 GB
      LoRA adapters:                     ~0.1 GB
      Adam optimizer states for LoRA:    ~0.4 GB
      Activations / misc:                ~1.0 GB
      Total:                             ~5.0 GB  (leaves 7 GB headroom)
    """
    from transformers import AutoModelForCausalLM

    # attn_implementation="flash_attention_2" requires:
    #   uv pip install flash-attn --no-build-isolation  (one-time ~15 min compile)
    # If flash-attn is not installed, fall back to the default SDPA attention.
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
        print(f"[model] Loading {model_name} in bfloat16 + Flash Attention 2...")
    except ImportError:
        attn_impl = "sdpa"
        print(f"[model] Loading {model_name} in bfloat16 (Flash Attention not installed; using SDPA)...")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map          = "auto",
        dtype               = torch.bfloat16,
        trust_remote_code   = True,
        attn_implementation = attn_impl,
    )

    lora_config = LoraConfig(
        task_type        = TaskType.CAUSAL_LM,
        r                = 16,           # rank 16 for better capacity without QLoRA
        lora_alpha       = 32,
        lora_dropout     = 0.05,
        target_modules   = ["q_proj", "v_proj", "k_proj", "o_proj"],
        bias             = "none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# Step 3: GRPO training config 

def make_grpo_config() -> GRPOConfig:
    """
    GRPOConfig for single-GPU colocate-mode training.

    vllm_mode="colocate": vLLM runs inside the trainer process, sharing the
    same GPU. Weight sync is a direct in-memory tensor copy — no NCCL
    inter-process communication, no separate vllm-serve process needed.
    This is the only vLLM mode that works on a single GPU.

    Speed vs stability balance:
      num_generations=64          — doubled from 32. With K=30 rollouts the reward
                                    signal width grew to ~1.0 but with only 32
                                    samples the SEM of the group mean was 0.18,
                                    giving advantages ~80% noise.  64 samples
                                    cuts SEM to 0.13 and makes the tail completions
                                    statistically informative instead of random.
      per_device_train_batch_size=8, gradient_accumulation_steps=8
        - 8*8=64=num_generations
        - 8 backward passes of 8 completions; same per-pass activation memory
          as the previous setup, just twice as many passes per training step
      learning_rate=7e-7          — kept; ISR drift is being addressed via beta
                                    rather than further LR cuts which would
                                    starve learning velocity
      beta=0.15                   — raised from 0.10; the previous run hit
                                    ISR_min=0 in 43% of step-401-500 already,
                                    repeating the run-4 collapse pattern.
                                    Tighter anchor bounds chaotic updates from
                                    noisy gradients.
      vllm_enable_sleep_mode=True — releases KV cache (~0.5 GB) during backward
      vllm_gpu_memory_utilization=0.45 — gives PyTorch allocator 0.6 GB more room
      max_completion_length=48    — completions are ~29 tokens
      vllm_structured_outputs_regex=DRONE_ACTION_REGEX  ← THE run-7 fix
                                  — forces every completion to be valid JSON
                                    {"force_x":N,"force_y":N,"torque_z":N}.
                                    Previously the model emitted free-form text
                                    and the reward_fn parser pulled 3 numbers
                                    out of whatever it found, which broke
                                    credit assignment between tokens and reward.
      temperature=1.0              — raised back from 0.5 (run 8 fix).
                                    At 0.5, ISR_min hit 0.0 in 70 % of the
                                    last 200 steps of run 7 (139/200 windows).
                                    With the regex guaranteeing valid JSON,
                                    higher temperature only widens the
                                    distribution over force values, which
                                    keeps old completions within the support
                                    of the new policy and preserves ISR.
    """
    return GRPOConfig(
        # Output
        output_dir              = OUTPUT_DIR,

        # Generation — 64 per prompt for high-quality advantage estimates.
        num_generations         = NUM_GENERATIONS,
        max_completion_length   = 48,              # actual completions are ~29 tokens
        # Make the temperature 1.2 so that the distribution over force values is broad
        temperature             = 1.2,

        # Training
        # 8 * 8 = 64 = num_generations → generation_batch_size constraint met
        # batch=8 keeps peak activation memory unchanged; 2× more passes per step
        per_device_train_batch_size   = 4,
        gradient_accumulation_steps   = 4,
        num_train_epochs              = _TRAIN_EPOCHS,
        # Held at 7e-7 — drift is now being addressed by the higher beta and
        # better SNR rather than by lowering the update magnitude further.
        learning_rate                 = 7e-7,
        warmup_steps                  = 10,
        lr_scheduler_type             = "cosine",
        optim                         = "adamw_torch_fused",
        # KL penalty: adds -beta * KL(π_new || π_old) to the loss.
        # Run 12 - change the beta value to 0.5
        beta                          = 0.5,

        # vLLM colocate mode — single GPU, no separate server process
        use_vllm                      = True,
        vllm_mode                     = "colocate",
        vllm_gpu_memory_utilization   = 0.45,      # ~0.6 GB back from vLLM → more room
        vllm_max_model_length         = 2048,      # for PyTorch allocator during backward
        vllm_enable_sleep_mode        = True,      # release KV cache during backward pass

        # ── Structured outputs — THE critical fix in run 7 ─────────────────
        # The DRONE_ACTION_REGEX in sim/schemas.py was defined but never wired
        # into the trainer in any prior run.  As a result every completion was
        # free-form text, and the reward_fn parser fell back to "extract any 3
        # numbers from the text" — sometimes those numbers came from prompt
        # values the model echoed back, not from any actual policy decision.
        # Credit assignment was structurally broken: identical action choices
        # could arrive from arbitrarily different token sequences, so the
        # gradient could not consistently push the model toward the right
        # numbers.  Forcing every completion through the regex makes the
        # action a direct function of the model's tokens, which is what
        # GRPO is designed to optimise.
        vllm_structured_outputs_regex = DRONE_ACTION_REGEX,

        # Logging & saving
        logging_steps           = 1,
        save_steps              = 50,
        save_total_limit        = 3,
        report_to               = "none",

        # Memory
        bf16                    = True,
        dataloader_num_workers  = 0,               # no multiprocessing (PyBullet is forky)
        remove_unused_columns   = False,
    )


# Main 

def main(_resume_ckpt=None):
    print(f"\n{'-'*60}")
    print(f"  GRPO Training — Drone2D Navigator")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Envs:  {NUM_ENVS}  |  Output: {OUTPUT_DIR}")
    print(f"{'-'*60}\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Pre-warm reward environments 
    # Must happen before GRPOTrainer is instantiated
    init_reward_env(num_envs=NUM_ENVS)

    # Load tokenizer 
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build dataset — curriculum: goals within 3 m for early navigation learning
    dataset = build_prompt_dataset(num_envs=NUM_ENVS, samples_per_env=_SAMPLES_PER_ENV, max_goal_dist=3.0)

    # Load bfloat16 LoRA model 
    model = load_lora_model(MODEL_NAME)

    # Training config 
    config = make_grpo_config()

    # GRPOTrainer 
    # reward_funcs receives our sim_reward — TRL calls it after each
    # generation batch with (prompts, completions) and expects list[float]
    metrics_file = os.path.join(OUTPUT_DIR, "metrics.jsonl")
    trainer = GRPOTrainer(
        model            = model,
        args             = config,
        processing_class = tokenizer,
        train_dataset    = dataset,
        reward_funcs     = [sim_reward],    # our physics simulator IS the reward
        callbacks        = [JSONLMetricsCallback(metrics_file)],
    )

    try:
        trainer.train(resume_from_checkpoint=_resume_ckpt)
    finally:
        # Always clean up, even on crash/interrupt
        close_reward_env()

    # Save final adapter weights 
    final_dir = os.path.join(OUTPUT_DIR, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\n[train] Saved final model to {final_dir}")
    print(f"\n{'-'*60}")
    print(f"  Training complete")
    print(f"{'-'*60}\n")


if __name__ == "__main__":
    import argparse, glob as _glob

    parser = argparse.ArgumentParser(description="GRPO training — Drone2D Navigator")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint in OUTPUT_DIR.  "
             "Without this flag training always starts from step 0.",
    )
    args = parser.parse_args()

    _resume_ckpt = None
    if args.resume:
        ckpt_dirs = sorted(_glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*")))
        _resume_ckpt = ckpt_dirs[-1] if ckpt_dirs else None

    if _resume_ckpt:
        print(f"\n[train] Resuming from checkpoint: {_resume_ckpt}\n")
    else:
        if args.resume:
            print("\n[train] --resume passed but no checkpoint found; starting fresh.\n")
        else:
            print("\n[train] Starting GRPO training from scratch (step 0).\n")

    main(_resume_ckpt)

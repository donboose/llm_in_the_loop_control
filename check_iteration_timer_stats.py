import json 
import statistics


times = [float(r['step_time']) 
         for line in open('checkpoints/drone_grpo/metrics.jsonl') 
            if (r := json.loads(line)).get('step_time')
         ]


print(f'n={len(times)}  \
      mean={statistics.mean(times):.2f}s  \
      median={statistics.median(times):.2f}s  \
      min={min(times):.2f}s  \
      max={max(times):.2f}s  \
      ETA={statistics.mean(times)*4800/3600:.1f}h' \
    )


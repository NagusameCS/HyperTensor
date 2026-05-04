#!/usr/bin/env python3
"""Model L completion monitor --- polls every 2 min, triggers post-L pipeline on completion."""
import os, sys, time, subprocess
from datetime import datetime

FINAL = "outputs/pure_models/smollm2-135m-language-pure/final/adapter_model.safetensors"
LOG = "benchmarks/model_l_monitor.log"

with open(LOG, 'w') as log:
    log.write(f"Monitor started: {datetime.now()}\n")
    
    while True:
        ts = datetime.now().strftime("%H:%M:%S")
        
        if os.path.exists(FINAL):
            log.write(f"[{ts}] DETECTED --- Model L complete!\n")
            print(f"[{ts}] MODEL L COMPLETE! Running post-L pipeline...")
            result = subprocess.run([sys.executable, "scripts/post_l_runner.py"], 
                                    capture_output=True, text=True)
            log.write(f"Post-L exit: {result.returncode}\n")
            log.write(result.stdout[-500:] if result.stdout else "no stdout")
            log.write(result.stderr[-500:] if result.stderr else "no stderr")
            log.flush()
            print("Post-L pipeline finished. Check benchmarks/model_l_monitor.log")
            break
        
        # Check checkpoint progress
        ckpt_dir = "outputs/pure_models/smollm2-135m-language-pure"
        ckpts = sorted([d for d in os.listdir(ckpt_dir) if d.startswith("checkpoint-")]) if os.path.exists(ckpt_dir) else []
        latest = ckpts[-1] if ckpts else "none"
        log.write(f"[{ts}] Latest checkpoint: {latest}\n")
        log.flush()
        
        time.sleep(120)  # Check every 2 minutes

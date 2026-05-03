#!/usr/bin/env python3
"""EC2 Pipeline Watcher: waits for math to finish, runs language, then CECI."""
import os, subprocess, time, sys

MATH_LOG = os.path.expanduser("~/train_qwen_math.log")
STATUS = os.path.expanduser("~/pipe_status.txt")
VENV_PY = "/home/ubuntu/venv/bin/python3"

def log(msg):
    with open(STATUS, "a") as f:
        f.write(msg + "\n")
    print(msg)

log("WATCHER STARTED")

# Wait for math training to complete
while True:
    if os.path.exists(MATH_LOG):
        with open(MATH_LOG) as f:
            content = f.read()
            if "DONE" in content:
                break
    time.sleep(60)

log("MATH_DONE")

# Run language training
ret = subprocess.run([VENV_PY, "-u", os.path.expanduser("~/train_qwen_pure.py"), 
                      "--skill", "language", "--steps", "4000"],
                     stdout=open(os.path.expanduser("~/train_qwen_lang.log"), "w"),
                     stderr=subprocess.STDOUT)
log(f"LANG_DONE exit={ret.returncode}")

# Run CECI
ret = subprocess.run([VENV_PY, "-u", os.path.expanduser("~/ceci_qwen.py")],
                     stdout=open(os.path.expanduser("~/ceci_qwen.log"), "w"),
                     stderr=subprocess.STDOUT)
log(f"CECI_DONE exit={ret.returncode}")
log("ALL_DONE")

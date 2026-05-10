#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::.................:::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::.............................::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::......................................:::::::::::::::::::::::::::
#  ::::::::::::::::::::::::......................*%:....................::::::::::::::::::::::::
#  ::::::::::::::::::::::.......................+@@@-......................::::::::::::::::::::::
#  ::::::::::::::::::::........................+@@@@@:.......................:::::::::::::::::::
#  ::::::::::::::::::.........................=@@@@@@@:........................:::::::::::::::::
#  ::::::::::::::::..........................:@@@@@@@@@-........................:::::::::::::::
#  :::::::::::::::..........................-@@@@@@@@@@@=.........................:::::::::::::
#  :::::::::::::...........................=@@@@@@@@@@@@@-.........................::::::::::::::
#  ::::::::::::...........................-@@@@@@@@@@@@@@@..........................:::::::::::
#  :::::::::::............................:%@@@@@@@@@@@@@+...........................:::::::::
#  ::::::::::..............................=@@@@@@@@@@@@%:............................:::::::::
#  ::::::::::...............................*@@@@@@@@@@@=..............................::::::::
#  :::::::::................................:@@@@@@@@@@%:...............................::::::
#  ::::::::..................................*@@@@@@@@@-................................::::::::
#  ::::::::..................:@@+:...........:@@@@@@@@@.............:+-..................:::::::
#  :::::::...................*@@@@@@*-:.......%@@@@@@@+........:-*@@@@@..................:::::::
#  :::::::..................:@@@@@@@@@@@%:....*@@@@@@@:....:=%@@@@@@@@@=.................:::::::
#  :::::::..................*@@@@@@@@@@@@#....=@@@@@@@....:*@@@@@@@@@@@#..................::::::
#  :::::::.................:@@@@@@@@@@@@@@-...=@@@@@@@....*@@@@@@@@@@@@@:.................::::::
#  :::::::.................*@@@@@@@@@@@@@@@:..=@@@@@@#...+@@@@@@@@@@@@@@=.................::::::
#  :::::::................:@@@@@@@@@@@@@@@@*..=@@@@@@#..+@@@@@@@@@@@@@@@+.................::::::
#  :::::::................=@@@@@@@@@@@@@@@@@-.#@@@@@@@.-@@@@@@@@@@@@@@@@*................:::::::
#  :::::::...............:#@@@@@@@@@@@@@@@@@*.@@@@@@@@:@@@@@@@@@@@@@@@@@%:...............:::::::
#  ::::::::..............:*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%:...............:::::::
#  ::::::::................:*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@-...............::::::::
#  :::::::::.................:=#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%-.................::::::::
#  ::::::::::....................:#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@=...................::::::::::
#  ::::::::::.......................:*@@@@@@@@@@@@@@@@@@@@@@@@@#-.....................:::::::::
#  :::::::::::.........................:=@@@@@@@@@@@@@@@@@@*:........................:::::::::::
#  ::::::::::::......................:=%@@@@@@@@@@@@@@@@@@@@#:......................::::::::::::
#  :::::::::::::.............+#%@@@@@@@@@@@@@@%-::*-.:%@@@@@@@@%=:.................::::::::::::::
#  :::::::::::::::...........:#@@@@@@@@@@@#--+%@@@@@@@#=:=%@@@@@@@@@@-............::::::::::::::::
#  ::::::::::::::::............-@@@@@@+-=#@@@@@@@@@@@@@@@@#=-=#@@@@*:............::::::::::::::::
#  ::::::::::::::::::...........:==:...-@@@@@@@@@@@@@@@@@@@@:...:=-............:::::::::::::::::
#  :::::::::::::::::::...................@@@@@@@@@@@@@@@@@-..................::::::::::::::::::::
#  ::::::::::::::::::::::................:#@@@@@@@@@@@@@*:.................::::::::::::::::::::::
#  ::::::::::::::::::::::::...............:*@@%+-.:=#@%-................::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::.............:........................:::::::::::::::::::::::::::
#  :::::::::::::::::::::::::::::::...............................:::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::.....................:::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

"""isagi_launcher.py — Auto-calibrating ISAGI entry point.

Checks for per-model calibration. If missing, runs ROC sweep automatically.
Then launches ISAGI chat with calibrated parameters.

USAGE:
  python scripts/isagi_launcher.py                          # auto-calibrate + chat
  python scripts/isagi_launcher.py --model Qwen/Qwen2.5-7B-Instruct
  python scripts/isagi_launcher.py --recalibrate             # force recalibration
  python scripts/isagi_launcher.py --validate --n 500        # validation mode
  python scripts/isagi_launcher.py --benchmark --n 200       # benchmark mode
"""
import torch, json, time, os, sys, argparse, subprocess
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CAL_DIR = Path("benchmarks/isagi_calibration")
CAL_DIR.mkdir(parents=True, exist_ok=True)

def get_calibration_path(model_name):
    slug = model_name.replace("/", "_")
    return CAL_DIR / f"calibration_{slug}.json"

def needs_calibration(model_name):
    """Check if calibration exists and is from today."""
    path = get_calibration_path(model_name)
    if not path.exists():
        return True
    try:
        data = json.loads(path.read_text())
        if data.get("n_adversarial", 0) < 10:
            return True  # quick mode, redo full
        return False
    except:
        return True

def run_calibration(model_name):
    """Run full calibration sweep."""
    print("=" * 60)
    print("  AUTO-CALIBRATING ISAGI")
    print(f"  Model: {model_name}")
    print("  This runs a ROC sweep to find optimal snipe coords + TEH threshold.")
    print("  Takes ~5-10 minutes on 1.5B, ~15-20 on 7B.")
    print("=" * 60)
    
    script = Path(__file__).parent / "isagi_calibrate.py"
    result = subprocess.run([
        sys.executable, str(script),
        "--model", model_name,
    ], capture_output=False, text=False)
    
    if result.returncode != 0:
        print("  Calibration failed. Using defaults.")
        return None
    
    path = get_calibration_path(model_name)
    if path.exists():
        cal = json.loads(path.read_text())
        print(f"\n  Calibration complete:")
        print(f"    Snipe coords: {cal['snipe']['snipe_coords']}")
        print(f"    Specificity:  {cal['snipe']['specificity']:.1f}x")
        print(f"    TEH threshold: {cal['teh']['best_threshold']:.3f}")
        print(f"    Separation:   {cal['separation_ratio']:.0f}x")
        return cal
    return None

def load_calibration(model_name):
    """Load existing calibration, or run one."""
    if needs_calibration(model_name):
        return run_calibration(model_name)
    
    path = get_calibration_path(model_name)
    cal = json.loads(path.read_text())
    print(f"  Loaded calibration from {path}")
    print(f"  Snipe coords: {cal['snipe']['snipe_coords']}")
    print(f"  TEH threshold: {cal['teh']['best_threshold']:.3f}")
    return cal

def launch_chat(model_name):
    """Launch interactive ISAGI chat."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("=" * 60)
    print("  ISAGI — Interactive Living Model")
    print(f"  Model: {model_name}")
    print("=" * 60)
    
    # Load calibration
    cal = load_calibration(model_name)
    
    # Load model
    print(f"\n  Loading {model_name}...")
    try:
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb, device_map="auto")
    except:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"  VRAM: {vram:.1f}GB")
    
    # Launch chat - delegate to isagi_chat.py
    print(f"\n  Starting chat... (type /quit to exit)")
    print(f"  Commands: /status, /save, /load, /gtc, /quit")
    print(f"  Calibrated: snipe={cal['snipe']['snipe_coords']}, "
          f"teh_thresh={cal['teh']['best_threshold']:.3f}")
    print()
    
    # Run isagi_chat.py as subprocess with calibration env
    os.environ["ISAGI_CALIBRATION"] = str(get_calibration_path(model_name))
    script = Path(__file__).parent / "isagi_chat.py"
    subprocess.run([sys.executable, str(script), "--model", model_name])

def main():
    parser = argparse.ArgumentParser(description="ISAGI Launcher")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--recalibrate", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--n", type=int, default=200)
    args = parser.parse_args()
    
    if args.recalibrate:
        run_calibration(args.model)
        return
    
    if args.validate:
        script = Path(__file__).parent / "isagi_validate.py"
        subprocess.run([sys.executable, str(script), "--model", args.model, "--n", str(args.n)])
        return
    
    if args.benchmark:
        script = Path(__file__).parent / "isagi_benchmark.py"
        subprocess.run([sys.executable, str(script), "--model", args.model, "--n", str(args.n), "--compare"])
        return
    
    launch_chat(args.model)

if __name__ == "__main__":
    main()

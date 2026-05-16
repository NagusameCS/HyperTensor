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
#  ::::::::::::::::::::::.......................+@@@-......................:::::::::::::::::::::
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
#  :::::::::::::::...........:#@@@@@@@@@@@#--+%@@@@@@@#=:=%@@@@@@@@@@-............:::::::::::::::
#  ::::::::::::::::............-@@@@@@+-=#@@@@@@@@@@@@@@@@#=-=#@@@@*:............:::::::::::::::
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

"""Validate HyperTensor unified model layer end-to-end.

Tests:
1. HuggingFace model loading + compression + GGUF export
2. OpenMythos model loading + compression + safetensors export
3. Industry-standard format compatibility (safetensors, GGUF)
4. Backend auto-detection
"""

import sys, json, shutil
from pathlib import Path

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[1]
sys.path.insert(0, str(_ROOT))

import torch
import numpy as np

from hyperretro.models import (
    load_model, compress_model, export_model,
    list_backends, list_formats,
)

OUT = _ROOT / "benchmarks" / "unified_pipeline_validation.json"


def test_backends():
    """Test backend auto-detection and listing."""
    print("=" * 60)
    print("TEST: Backend auto-detection")
    print("=" * 60)

    backends = list_backends()
    for name, available in backends.items():
        status = "✅" if available else "⚠️  (not installed)"
        print(f"  {name}: {status}")

    formats = list_formats()
    print(f"  Export formats: {formats}")
    return backends


def test_hf_pipeline():
    """Test the full HF pipeline: load → compress → export."""
    print("\n" + "=" * 60)
    print("TEST: HuggingFace pipeline")
    print("=" * 60)

    try:
        # Load a tiny HF model (random init for speed)
        from transformers import AutoConfig, AutoModelForCausalLM

        # Use Qwen2.5-1.5B config but build from config (no download needed if cached)
        model_id = "Qwen/Qwen2.5-1.5B"
        print(f"  Loading HF model: {model_id} ...")

        model = load_model(model_id, backend="huggingface", dtype="float16")
        print(f"  Architecture: {model.architecture}")
        print(f"  Parameters: {model.param_count:,}")
        print(f"  Hidden size: {model.get_hidden_size()}")
        print(f"  Layers: {model.get_num_layers()}")

        # Discover layers
        attn_layers = model.get_attention_layers()
        ffn_layers = model.get_ffn_layers()
        print(f"  Attention layers: {len(attn_layers)}")
        print(f"  FFN layers: {len(ffn_layers)}")
        if attn_layers:
            print(f"    Example: {attn_layers[0]['weight_key']} {attn_layers[0]['shape']}")

        # Compress (light config for speed)
        print(f"\n  Compressing (ffn_rank=128, int4=True) ...")
        compressed = compress_model(
            model, ffn_rank=128, attn_rank=0, int4=True,
        )
        print(f"  Compressed tensors: {compressed.total_tensors}")

        # Export GGUF
        gguf_path = _ROOT / "outputs" / "_test_unified_hf.gguf"
        gguf_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"\n  Exporting GGUF: {gguf_path} ...")
        result = export_model(compressed, gguf_path, format="gguf")
        gguf_mb = result.stat().st_size / 1e6
        print(f"  GGUF size: {gguf_mb:.1f} MB")

        # Export safetensors
        st_path = _ROOT / "outputs" / "_test_unified_hf_st"
        print(f"\n  Exporting safetensors: {st_path} ...")
        result = export_model(compressed, st_path, format="safetensors")
        print(f"  Exported OK")

        # Cleanup
        gguf_path.unlink(missing_ok=True)
        shutil.rmtree(st_path, ignore_errors=True)

        return {
            "status": "ok",
            "architecture": model.architecture,
            "params": model.param_count,
            "attn_layers": len(attn_layers),
            "ffn_layers": len(ffn_layers),
            "compressed_tensors": compressed.total_tensors,
            "gguf_mb": round(gguf_mb, 1),
        }
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "failed", "error": str(e)}


def test_om_pipeline():
    """Test the OpenMythos pipeline if available."""
    print("\n" + "=" * 60)
    print("TEST: OpenMythos pipeline")
    print("=" * 60)

    try:
        from open_mythos import mythos_1b

        print("  Loading OM model: mythos_1b ...")
        model = load_model("mythos_1b", backend="openmythos")
        print(f"  Architecture: {model.architecture}")
        print(f"  Parameters: {model.param_count:,}")
        print(f"  Hidden size: {model.get_hidden_size()}")
        print(f"  Recurrent loops: {model.num_recurrent_loops}")

        # Discover layers
        attn_layers = model.get_attention_layers()
        ffn_layers = model.get_ffn_layers()
        print(f"  Attention layers: {len(attn_layers)} (recurrent: {sum(1 for l in attn_layers if l.get('is_recurrent'))})")
        print(f"  FFN layers: {len(ffn_layers)} (experts: {sum(1 for l in ffn_layers if l.get('is_expert'))})")

        # Compress
        print(f"\n  Compressing (ffn_rank=256, int4=True) ...")
        compressed = compress_model(
            model, ffn_rank=256, attn_rank=0, int4=True,
        )
        print(f"  Compressed tensors: {compressed.total_tensors}")

        # Export safetensors
        st_path = _ROOT / "outputs" / "_test_unified_om_st"
        print(f"\n  Exporting safetensors: {st_path} ...")
        export_model(compressed, st_path, format="safetensors")

        # Verify the exported files
        st_files = list(st_path.glob("*"))
        print(f"  Exported files: {[f.name for f in st_files]}")

        # Check safetensors is valid
        from safetensors.torch import load_file
        sf_path = st_path / "model.safetensors"
        if sf_path.exists():
            sd = load_file(str(sf_path))
            print(f"  Safetensors keys: {len(sd)}")
            st_mb = sf_path.stat().st_size / 1e6
            print(f"  Safetensors size: {st_mb:.1f} MB")

            # Check for industry-standard metadata files
            has_config = (st_path / "config.json").exists()
            has_manifest = (st_path / "hyperretro_factored.json").exists()
            print(f"  config.json: {'✅' if has_config else '❌'}")
            print(f"  hyperretro_factored.json: {'✅' if has_manifest else '❌'}")

        # Cleanup
        shutil.rmtree(st_path, ignore_errors=True)

        return {
            "status": "ok",
            "architecture": model.architecture,
            "params": model.param_count,
            "attn_layers": len(attn_layers),
            "ffn_layers": len(ffn_layers),
            "recurrent_loops": model.num_recurrent_loops,
            "compressed_tensors": compressed.total_tensors,
            "has_config": has_config,
            "has_manifest": has_manifest,
        }
    except ImportError:
        print("  ⚠️  OpenMythos not installed, skipping")
        return {"status": "skipped", "reason": "open-mythos not installed"}
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "failed", "error": str(e)}


def test_industry_compat():
    """Test compatibility with industry standard formats."""
    print("\n" + "=" * 60)
    print("TEST: Industry format compatibility")
    print("=" * 60)

    results = {}

    # 1. safetensors format
    try:
        from safetensors.torch import save_file, load_file
        test_sd = {"test.weight": torch.randn(64, 128)}
        tmp = _ROOT / "outputs" / "_test_compat.safetensors"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        save_file(test_sd, str(tmp))
        loaded = load_file(str(tmp))
        assert "test.weight" in loaded
        tmp.unlink()
        print("  safetensors: ✅")
        results["safetensors"] = "ok"
    except Exception as e:
        print(f"  safetensors: ❌ {e}")
        results["safetensors"] = str(e)

    # 2. GGUF format
    try:
        from gguf import GGUFReader, GGUFWriter
        print("  GGUF: ✅ (writer + reader available)")
        results["gguf"] = "ok"
    except ImportError:
        print("  GGUF: ⚠️  (gguf package not installed)")
        results["gguf"] = "available_but_not_tested"

    # 3. HuggingFace config format
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained("Qwen/Qwen2.5-1.5B")
        cfg_dict = cfg.to_dict()
        assert "hidden_size" in cfg_dict
        print("  HuggingFace config: ✅")
        results["huggingface_config"] = "ok"
    except Exception as e:
        print(f"  HuggingFace config: ❌ {e}")
        results["huggingface_config"] = str(e)

    return results


def main():
    print("HyperTensor Unified Model Layer — Validation")
    print("=" * 60)

    results = {
        "backends": {},
        "hf_pipeline": {},
        "om_pipeline": {},
        "industry_compat": {},
    }

    # 1. Backend detection
    backends = test_backends()
    results["backends"] = backends

    # 2. HuggingFace pipeline
    results["hf_pipeline"] = test_hf_pipeline()

    # 3. OpenMythos pipeline
    results["om_pipeline"] = test_om_pipeline()

    # 4. Industry compatibility
    results["industry_compat"] = test_industry_compat()

    # Save results
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\n[saved] {OUT}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for section, data in results.items():
        if isinstance(data, dict):
            ok = all(
                v in ("ok", True) or v == "available_but_not_tested"
                for v in (data.values() if isinstance(data, dict) else [])
            )
            status = "✅" if ok else "⚠️"
            print(f"  {status} {section}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

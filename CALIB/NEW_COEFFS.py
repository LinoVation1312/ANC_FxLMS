from __future__ import annotations
import json, sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent
root_dir = script_dir.parent

mic_sens_path = root_dir / "mic_sensitivity.json"
raw_coeffs_path = script_dir / "filter_coeffs_RAW.json"
output_coeffs_path = root_dir / "filter_coeffs.json"

def load_mic2_sens() -> float:
    if not mic_sens_path.exists():
        print(f"[WARN] Fichier {mic_sens_path.name} introuvable.")
        resp = input("Lancer quand même avec mic2_sens=1.0 ? (o/N) ").strip().lower()
        if resp not in ("o","oui","y","yes"):
            print("Abandon. Exécutez d'abord: python CALIB/CALIB_MIC.py")
            sys.exit(1)
        return 1.0
    with mic_sens_path.open("r") as f:
        data = json.load(f)
    if "mic2_sens" not in data:
        raise KeyError("Clé 'mic2_sens' absente dans mic_sensitivity.json")
    return float(data["mic2_sens"])

def load_raw_filter():
    if not raw_coeffs_path.exists():
        raise FileNotFoundError(f"Fichier {raw_coeffs_path} introuvable")
    with raw_coeffs_path.open("r") as f:
        fr = json.load(f)
    for k in ("A","B"):
        if k not in fr:
            raise KeyError(f"Clé '{k}' absente dans {raw_coeffs_path.name}")
    return fr

def main():
    mic2_sens = load_mic2_sens()
    filter_raw = load_raw_filter()
    B_raw = filter_raw["B"]
    # Ensure numeric list
    try:
        B_scaled = [float(b) * mic2_sens for b in B_raw]
    except TypeError as e:
        raise TypeError("Les coefficients B doivent être itérables de nombres") from e
    filter_scaled = {
        "B": B_scaled,
        "A": filter_raw["A"],
        "tau_ms": filter_raw.get("tau_ms", 0.0)
    }
    with output_coeffs_path.open("w") as f:
        json.dump(filter_scaled, f, indent=2)
    print(f"Saved adjusted filter coefficients to {output_coeffs_path.relative_to(root_dir)} using mic2_sens={mic2_sens:.6g}")

if __name__ == "__main__":
    main()

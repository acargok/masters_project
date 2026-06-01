from pathlib import Path

QE_PSI_C = 1.5

# Paths
ROOT = Path(__file__).resolve().parent.parent
IV_DIR = ROOT / "iv_surface"
DUPIRE_DIR = ROOT / "dupire_vol"
LSV_DIR = ROOT / "lsv_heston"
BERGOMI_DIR = ROOT / "lsv_bergomi"

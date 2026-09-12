import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.orchestrate import run_all

if __name__ == "__main__":
    genai = "--genai" in sys.argv or "--no-genai" not in sys.argv
    out = run_all(root=str(ROOT), genai=genai)
    print("PDF:", out["pdf"])
    print("PWECL:", out["results"]["Probability_Weighted_ECL"])

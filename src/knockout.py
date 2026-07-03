"""
knockout.py

Runs the trained model on the ACTUAL knockout matchups (which only become known
after each round) and writes the model's pick + win% for each game to
docs/data/knockout.json. Knockout games can't draw, so the draw probability is
removed and win/win is renormalized.

Update KNOCKOUT below as each round's matchups are confirmed, then re-run.
"""
import json
from pathlib import Path

import pandas as pd

from snapshot import prepare, _knockout_probs

DATA = Path("docs/data")


def wc_results_as_matches():
    """Turn the logged 2026 WC results (group + knockout) into match rows so ELO
    becomes form-aware. compute_elo only needs home_team/away_team/result/date."""
    rows = []
    for r in json.load(open(DATA / "results.json")):
        res = "Home Win" if r["hs"] > r["as"] else ("Away Win" if r["hs"] < r["as"] else "Draw")
        rows.append({"date": pd.Timestamp(r["date"]), "home_team": r["home"],
                     "away_team": r["away"], "result": res, "augmented": False})
    ko_path = DATA / "knockout_results.json"
    if ko_path.exists():
        for k in json.load(open(ko_path)):
            if "pens" in k.get("score", ""):
                res = "Draw"                      # shootout = a draw over 120 min
            elif k["winner"] == k["home"]:
                res = "Home Win"
            else:
                res = "Away Win"
            rows.append({"date": pd.Timestamp("2026-06-30"), "home_team": k["home"],
                         "away_team": k["away"], "result": res, "augmented": False})
    return pd.DataFrame(rows)

# Actual matchups per round — extend as each round is set.
KNOCKOUT = [
    # Round of 32
    ("Round of 32", "Canada", "South Africa"),
    ("Round of 32", "Brazil", "Japan"),
    ("Round of 32", "Germany", "Paraguay"),
    ("Round of 32", "Netherlands", "Morocco"),
    ("Round of 32", "Norway", "Ivory Coast"),
    ("Round of 32", "France", "Sweden"),
    ("Round of 32", "Mexico", "Ecuador"),
    ("Round of 32", "United States", "Bosnia and Herzegovina"),
    ("Round of 32", "Spain", "Austria"),
    ("Round of 32", "Belgium", "Senegal"),
    ("Round of 32", "Portugal", "Croatia"),
    ("Round of 32", "Switzerland", "Algeria"),
    ("Round of 32", "Australia", "Egypt"),
    ("Round of 32", "England", "DR Congo"),
    ("Round of 32", "Argentina", "Cape Verde"),
    ("Round of 32", "Colombia", "Ghana"),
]


def main():
    wc = wc_results_as_matches()
    print(f"Folding {len(wc)} played 2026 WC matches into ELO (form-aware)...")
    _, elo_dict, strengths, prob_cache = prepare(extra_matches=wc)

    out = []
    print("\n=== MODEL KNOCKOUT PREDICTIONS ===")
    current_round = None
    for rnd, a, b in KNOCKOUT:
        if rnd != current_round:
            print(f"\n-- {rnd} --")
            current_round = rnd
        hw, aw = _knockout_probs(prob_cache, a, b)
        hw, aw = float(hw), float(aw)
        if hw >= aw:
            pick, pct = a, round(hw * 100, 1)
        else:
            pick, pct = b, round(aw * 100, 1)
        out.append({
            "round": rnd, "home": a, "away": b,
            "pick": pick, "pick_pct": pct,
            "home_pct": round(hw * 100, 1), "away_pct": round(aw * 100, 1),
        })
        print(f"  {a} vs {b:<24} -> {pick} ({pct}%)")

    Path("docs/data").mkdir(parents=True, exist_ok=True)
    with open("docs/data/knockout.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {len(out)} predictions to docs/data/knockout.json")


if __name__ == "__main__":
    main()

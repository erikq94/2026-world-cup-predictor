"""
knockout.py

Runs the trained model on the ACTUAL knockout matchups (which only become known
after each round) and writes the model's pick + win% for each game to
docs/data/knockout.json. Knockout games can't draw, so the draw probability is
removed and win/win is renormalized.

Update KNOCKOUT below as each round's matchups are confirmed, then re-run.
"""
import json
from datetime import date
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


# The actual bracket in tree order: consecutive pairs of R32 matches feed each
# R16 slot, and so on up to the final (from the official 2026 bracket).
R32_TREE_ORDER = [
    ("Canada", "South Africa"), ("Netherlands", "Morocco"),
    ("Germany", "Paraguay"), ("France", "Sweden"),
    ("Brazil", "Japan"), ("Ivory Coast", "Norway"),
    ("Mexico", "Ecuador"), ("England", "DR Congo"),
    ("Portugal", "Croatia"), ("Spain", "Austria"),
    ("United States", "Bosnia and Herzegovina"), ("Belgium", "Senegal"),
    ("Argentina", "Cape Verde"), ("Australia", "Egypt"),
    ("Switzerland", "Algeria"), ("Colombia", "Ghana"),
]


def load_actual_winners():
    """Every played knockout matchup (both orientations) -> actual winner."""
    actual = {}
    ko_path = DATA / "knockout_results.json"
    if ko_path.exists():
        for k in json.load(open(ko_path)):
            actual[(k["home"], k["away"])] = k["winner"]
            actual[(k["away"], k["home"])] = k["winner"]
    return actual


def build_predicted_bracket(prob_cache, actual):
    """Walk the real bracket: use the actual winner where a game is played, else
    the form-aware model's favorite. Returns rounds of matches + the champion."""
    round_names = ["Round of 32", "Round of 16", "Quarterfinal", "Semifinal", "Final"]
    pairs = list(R32_TREE_ORDER)
    rounds = []
    for rname in round_names:
        matches, winners = [], []
        for a, b in pairs:
            hw, aw = _knockout_probs(prob_cache, a, b)
            hw, aw = float(hw), float(aw)
            act = actual.get((a, b))
            if act:
                winner, win_pct, played = act, round((hw if act == a else aw) * 100, 1), True
            else:
                winner = a if hw >= aw else b
                win_pct, played = round(max(hw, aw) * 100, 1), False
            matches.append({"home": a, "away": b, "winner": winner,
                            "win_pct": win_pct, "played": played})
            winners.append(winner)
        rounds.append({"round": rname, "matches": matches})
        if len(winners) > 1:
            pairs = [(winners[i], winners[i + 1]) for i in range(0, len(winners), 2)]
    return {"rounds": rounds, "champion": rounds[-1]["matches"][0]["winner"]}

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

    DATA.mkdir(parents=True, exist_ok=True)
    with open(DATA / "knockout.json", "w") as f:
        json.dump(out, f, indent=2)

    # Full predicted bracket (real results + form-aware picks) -> updated champion
    bracket = build_predicted_bracket(prob_cache, load_actual_winners())
    bracket["updated"] = date.today().isoformat()
    with open(DATA / "knockout_bracket.json", "w") as f:
        json.dump(bracket, f, indent=2)

    print("\n=== UPDATED PREDICTED BRACKET (form-aware, real results folded in) ===")
    for rnd in bracket["rounds"]:
        print(f"  {rnd['round']:<14}: {', '.join(m['winner'] for m in rnd['matches'])}")
    print(f"  >>> UPDATED CHAMPION: {bracket['champion']} <<<")


if __name__ == "__main__":
    main()

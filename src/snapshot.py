"""
snapshot.py

Generates a "living bracket" snapshot from the trained v2 model and writes it as
web-friendly JSON (results/snapshots/). A snapshot contains, for every team:
  - ELO + squad stats (for display)
  - per-round REACH probabilities  (Spain: R16 87%, QF 64%, ... champion 21%)
and for the tournament:
  - a single PREDICTED bracket (favorite advances each match)
  - every group-stage match prediction (win/draw/loss %)

Pre-tournament we simulate the group stage. As the real tournament unfolds, pass
the actual surviving teams via `fixed_qualifiers` to re-cast the forecast from
that round forward and save the next snapshot.

No model training happens here — this is pure orchestration on top of the saved
model, reusing the simulation helpers from simulate.py.
"""
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from data_loader import load_all
from features import get_team_strength, compute_elo, VENUE_INFO, STARTING_ELO
from simulate import (
    GROUPS, CITY_MAP, get_qualifiers, simulate_group_stage_fast,
    simulate_match, TEMPERATURE,
)

MODELS_DIR = Path('models')
SNAP_DIR   = Path('results/snapshots')

# Knockout rounds, in order. round_of_32 = "made the knockouts".
ROUND_KEYS  = ['round_of_32', 'round_of_16', 'quarterfinal', 'semifinal', 'final', 'champion']
# Rounds a winner advances INTO after each knockout game-round.
ADVANCES_TO = ['round_of_16', 'quarterfinal', 'semifinal', 'final', 'champion']

FEATURE_STAT_KEYS = ['attack_strength', 'defensive_strength', 'gk_rating', 'squad_depth',
                     'team_pace', 'team_stamina', 'passing_quality']


def _to_native(o):
    """Convert numpy scalar types to native Python so json can serialize them."""
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    raise TypeError(f'Object of type {o.__class__.__name__} is not JSON serializable')


# --------------------------------------------------------------------------- #
# Setup: load model + build the probability cache (self-contained, no changes
# to simulate.py so the working v2 simulation stays untouched).
# --------------------------------------------------------------------------- #
def _match_features(home, away, strengths, elo_dict, venue):
    hs, as_ = strengths[home], strengths[away]
    return pd.DataFrame([{
        'elo_diff':     elo_dict.get(home, STARTING_ELO) - elo_dict.get(away, STARTING_ELO),
        'form_diff':    0,
        'attack_diff':  hs['attack_strength']    - as_['attack_strength'],
        'defense_diff': hs['defensive_strength'] - as_['defensive_strength'],
        'gk_diff':      hs['gk_rating']          - as_['gk_rating'],
        'depth_diff':   hs['squad_depth']         - as_['squad_depth'],
        'pace_diff':    hs['team_pace']           - as_['team_pace'],
        'stamina_diff': hs['team_stamina']        - as_['team_stamina'],
        'passing_diff': hs['passing_quality']     - as_['passing_quality'],
        'altitude':     venue[0], 'heat': venue[1], 'humidity': venue[2],
        'is_wc':        1, 'neutral': 1,
    }])


def prepare():
    """Load everything once: model, ELO, squad strengths, and the match cache."""
    print("Loading model + data...")
    model = joblib.load(MODELS_DIR / 'xgboost_model.pkl')
    results, wc_fixtures, _, players, _ = load_all()

    print("Computing ELO...")
    elo_hist = compute_elo(results)
    elo_dict = {r['team']: r['elo'] for _, r in elo_hist.sort_values('date').iterrows()}

    all_teams = [t for teams in GROUPS.values() for t in teams]
    strengths = {t: get_team_strength(players, t) for t in all_teams}

    print("Building match-probability cache...")
    prob_cache = {}
    # neutral-venue probabilities for every possible knockout pairing
    for i, home in enumerate(all_teams):
        for away in all_teams[i + 1:]:
            p = model.predict_proba(_match_features(home, away, strengths, elo_dict, (0, 25, 60)))[0]
            prob_cache[(home, away)] = {'away_win': p[0], 'draw': p[1], 'home_win': p[2]}
    # group fixtures with their real venues (override neutral where applicable)
    for _, fix in wc_fixtures.iterrows():
        home, away = fix['home_team'], fix['away_team']
        venue = VENUE_INFO.get(CITY_MAP.get(fix['city'], fix['city']), (0, 25, 60))
        p = model.predict_proba(_match_features(home, away, strengths, elo_dict, venue))[0]
        prob_cache[(home, away)] = {'away_win': p[0], 'draw': p[1], 'home_win': p[2]}

    return wc_fixtures, elo_dict, strengths, prob_cache


# --------------------------------------------------------------------------- #
# Knockout helpers
# --------------------------------------------------------------------------- #
def _knockout_probs(prob_cache, home, away):
    """Normalized (home_win, away_win) with draw removed; handles reverse lookup."""
    if (home, away) in prob_cache:
        p = prob_cache[(home, away)]
        hw, aw = p['home_win'], p['away_win']
    elif (away, home) in prob_cache:
        p = prob_cache[(away, home)]
        hw, aw = p['away_win'], p['home_win']   # swap perspective
    else:
        hw, aw = 0.4, 0.4
    total = hw + aw
    return hw / total, aw / total


def _seed_bracket(qualifiers):
    """Same seeding as simulate.py: 1st-place teams vs 2nd-place from offset groups."""
    firsts  = qualifiers[0:24:2]
    seconds = qualifiers[1:24:2]
    thirds  = qualifiers[24:]
    bracket = []
    for j in range(12):
        bracket.append(firsts[j])
        bracket.append(seconds[(j + 6) % 12])
    bracket.extend(thirds)
    return bracket


# --------------------------------------------------------------------------- #
# Monte Carlo: per-round reach probabilities
# --------------------------------------------------------------------------- #
def reach_probabilities(prob_cache, wc_fixtures, n_sims, fixed_qualifiers=None):
    teams_all = [t for teams in GROUPS.values() for t in teams]
    reach = {t: {r: 0 for r in ROUND_KEYS} for t in teams_all}

    for _ in range(n_sims):
        if fixed_qualifiers is None:
            standings  = simulate_group_stage_fast(wc_fixtures, prob_cache)
            qualifiers = get_qualifiers(standings)
        else:
            qualifiers = list(fixed_qualifiers)

        teams = _seed_bracket(qualifiers)
        for t in teams:
            reach[t]['round_of_32'] += 1

        for nxt in ADVANCES_TO:
            winners = []
            for i in range(0, len(teams), 2):
                home, away = teams[i], teams[i + 1]
                hw, aw = _knockout_probs(prob_cache, home, away)
                w, _ = simulate_match({'home_win': hw, 'away_win': aw, 'draw': 0.0}, home, away)
                winners.append(w)
            teams = winners
            for t in teams:
                reach[t][nxt] += 1

    return {t: {r: reach[t][r] / n_sims for r in ROUND_KEYS} for t in teams_all}


# --------------------------------------------------------------------------- #
# Deterministic "chalk" bracket: favorite advances every match
# --------------------------------------------------------------------------- #
def chalk_bracket(prob_cache, elo_dict, fixed_qualifiers=None):
    if fixed_qualifiers is None:
        # Predicted qualifiers: top 2 per group by ELO + 8 best thirds by ELO
        qualifiers, thirds = [], []
        for group_teams in GROUPS.values():
            ranked = sorted(group_teams, key=lambda t: elo_dict.get(t, STARTING_ELO), reverse=True)
            qualifiers += [ranked[0], ranked[1]]
            thirds.append(ranked[2])
        best_thirds = sorted(thirds, key=lambda t: elo_dict.get(t, STARTING_ELO), reverse=True)[:8]
        qualifiers += best_thirds
    else:
        qualifiers = list(fixed_qualifiers)

    teams = _seed_bracket(qualifiers)
    # Output each round as a list of MATCHES so the bracket clearly shows who plays
    # whom: {home, away, winner, win_pct}. Consecutive matches feed the next round.
    round_names = ['round_of_32', 'round_of_16', 'quarterfinal', 'semifinal', 'final']
    rounds = []
    for rname in round_names:
        matches, winners = [], []
        for i in range(0, len(teams), 2):
            home, away = teams[i], teams[i + 1]
            hw, aw = _knockout_probs(prob_cache, home, away)
            winner = home if hw >= aw else away
            matches.append({
                'home': home, 'away': away, 'winner': winner,
                'win_pct': round(float(max(hw, aw)) * 100, 1),
            })
            winners.append(winner)
        rounds.append({'round': rname, 'matches': matches})
        teams = winners
    return {'rounds': rounds, 'champion': teams[0]}


# --------------------------------------------------------------------------- #
# Group-stage match predictions (win/draw/loss %)
# --------------------------------------------------------------------------- #
def group_predictions(prob_cache, wc_fixtures):
    out = []
    for group_name, group_teams in GROUPS.items():
        for _, fix in wc_fixtures.iterrows():
            home, away = fix['home_team'], fix['away_team']
            if home in group_teams and away in group_teams:
                p = prob_cache.get((home, away))
                if p is None:
                    rev = prob_cache.get((away, home), {'home_win': .4, 'draw': .2, 'away_win': .4})
                    p = {'home_win': rev['away_win'], 'draw': rev['draw'], 'away_win': rev['home_win']}
                out.append({
                    'group': group_name, 'date': fix['date'],
                    'home': home, 'away': away,
                    'home_win': round(float(p['home_win']) * 100, 1),
                    'draw':     round(float(p['draw']) * 100, 1),
                    'away_win': round(float(p['away_win']) * 100, 1),
                })
    return out


# --------------------------------------------------------------------------- #
# Assemble + save the snapshot JSON
# --------------------------------------------------------------------------- #
def build_snapshot(label, round_name, n_sims=50000, fixed_qualifiers=None):
    wc_fixtures, elo_dict, strengths, prob_cache = prepare()

    print(f"Running {n_sims} simulations for reach probabilities...")
    reach = reach_probabilities(prob_cache, wc_fixtures, n_sims, fixed_qualifiers)
    bracket = chalk_bracket(prob_cache, elo_dict, fixed_qualifiers)

    teams_json = []
    for t in reach:
        s = strengths[t]
        teams_json.append({
            'team':  t,
            'elo':   round(elo_dict.get(t, STARTING_ELO)),
            'stats': {
                'attack':  round(s['attack_strength'], 1),
                'defense': round(s['defensive_strength'], 1),
                'gk':      round(s['gk_rating'], 1),
                'depth':   round(s['squad_depth'], 1),
            },
            'reach': {r: round(reach[t][r] * 100, 1) for r in ROUND_KEYS},
        })
    teams_json.sort(key=lambda x: -x['reach']['champion'])

    snapshot = {
        'snapshot': label,
        'round': round_name,
        'date': date.today().isoformat(),
        'n_simulations': n_sims,
        'temperature': TEMPERATURE,
        'model_version': 'v2',
        'teams': teams_json,
        'predicted_bracket': bracket,
        'group_predictions': group_predictions(prob_cache, wc_fixtures),
    }

    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SNAP_DIR / f'{label}.json'
    with open(out_path, 'w') as f:
        json.dump(snapshot, f, indent=2, default=_to_native)
    print(f"\nSnapshot saved to {out_path}")

    # Readable summary
    print("\n=== Predicted champion path ===")
    for rnd in bracket['rounds']:
        winners = [m['winner'] for m in rnd['matches']]
        print(f"  {rnd['round']:<14}: {', '.join(winners)}")
    print(f"  champion      : {bracket['champion']} 🏆")
    print("\n=== Top 8 by title probability ===")
    for tj in teams_json[:8]:
        print(f"  {tj['team']:<14} champion {tj['reach']['champion']:>5.1f}%  "
              f"final {tj['reach']['final']:>5.1f}%  SF {tj['reach']['semifinal']:>5.1f}%")
    return snapshot


if __name__ == '__main__':
    build_snapshot(label='snapshot_00_pretournament', round_name='group_stage', n_sims=50000)

"""
simulate.py

Uses the trained model to simulate the full 2026 World Cup bracket:
  - Group stage (48 teams, 12 groups of 4)
  - Round of 32 → Round of 16 → Quarterfinals → Semifinals → Final
  - Updates ELO and player availability after each simulated match
  - Supports live re-simulation after real match results come in
"""
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from features import get_team_strength, VENUE_INFO, STARTING_ELO, compute_elo
from data_loader import load_all

DATA_PROCESSED = Path('data/processed')
MODELS_DIR = Path('models')

# Fixture city names → our VENUE_INFO keys
CITY_MAP = {
    'East Rutherford': 'New York',
    'Inglewood':       'Los Angeles',
    'Arlington':       'Dallas',
    'Santa Clara':     'San Francisco',
    'Miami Gardens':   'Miami',
    'Foxborough':      'Boston',
    'Zapopan':         'Guadalajara',
    'Guadalupe':       'Monterrey',
}

GROUPS = {
    'A': ['Mexico', 'South Africa', 'South Korea', 'Czech Republic'],
    'B': ['Canada', 'Qatar', 'Switzerland', 'Bosnia and Herzegovina'],
    'C': ['Brazil', 'Morocco', 'Scotland', 'Haiti'],
    'D': ['United States', 'Paraguay', 'Australia', 'Turkey'],
    'E': ['Germany', 'Ivory Coast', 'Ecuador', 'Curaçao'],
    'F': ['Netherlands', 'Sweden', 'Japan', 'Tunisia'],
    'G': ['Belgium', 'Iran', 'Egypt', 'New Zealand'],
    'H': ['Spain', 'Saudi Arabia', 'Uruguay', 'Cape Verde'],
    'I': ['France', 'Senegal', 'Iraq', 'Norway'],
    'J': ['Argentina', 'Algeria', 'Austria', 'Jordan'],
    'K': ['Portugal', 'Uzbekistan', 'Colombia', 'DR Congo'],
    'L': ['England', 'Ghana', 'Panama', 'Croatia'],
}

# Temperature: chaos dial for the TOURNAMENT simulation only (group/match
# predictions are printed from the raw calibrated cache, unaffected by this).
#   1.0 = model's raw calibrated probabilities
#   1.3 = injects tournament-level uncertainty so winner % matches bookmaker
#         spread (single-elimination chaos the per-match model can't see)
#   >1.3 = even more upsets;  <1.0 = favorites crush everyone
TEMPERATURE = 1.3


def apply_temperature(weights, T):
    if T == 1.0:
        return weights
    w = np.power(weights, 1.0 / T)
    return w / w.sum()


def load_model():
    model = joblib.load(MODELS_DIR / 'xgboost_model.pkl')
    le = joblib.load(MODELS_DIR / 'label_encoder.pkl')
    print(f"Model loaded. Classes: {le.classes_}")
    return model, le

def predict_match_proba(model, le, home_team, away_team, elo_dict, players_df, city):
    city_key = CITY_MAP.get(city, city)
    venue    = VENUE_INFO.get(city_key, (0, 25, 60))

    hs  = get_team_strength(players_df, home_team)
    as_ = get_team_strength(players_df, away_team)

    elo_h = elo_dict.get(home_team, STARTING_ELO)
    elo_a = elo_dict.get(away_team, STARTING_ELO)

    features = pd.DataFrame([{
        'elo_diff':     elo_h - elo_a,
        'form_diff':    0,
        'attack_diff':  hs['attack_strength']    - as_['attack_strength'],
        'defense_diff': hs['defensive_strength'] - as_['defensive_strength'],
        'gk_diff':      hs['gk_rating']          - as_['gk_rating'],
        'depth_diff':   hs['squad_depth']         - as_['squad_depth'],
        'pace_diff':    hs['team_pace']           - as_['team_pace'],
        'stamina_diff': hs['team_stamina']        - as_['team_stamina'],
        'passing_diff': hs['passing_quality']     - as_['passing_quality'],
        'altitude':     venue[0],
        'heat':         venue[1],
        'humidity':     venue[2],
        'is_wc':        1,
        'neutral':      1,
    }])

    probs = model.predict_proba(features)[0]
    # probs order: [Away Win, Draw, Home Win] matching le.classes_
    return {
        'away_win': probs[0],
        'draw':     probs[1],
        'home_win': probs[2],
    }

def simulate_match(probs, home_team, away_team):
    outcomes   = ['away_win', 'draw', 'home_win']
    weights    = np.array([probs['away_win'], probs['draw'], probs['home_win']])
    weights    = weights / weights.sum()   # normalize floating point rounding
    weights    = apply_temperature(weights, TEMPERATURE)
    result     = np.random.choice(outcomes, p=weights)

    if result == 'home_win':
        return home_team, away_team    # winner, loser
    elif result == 'away_win':
        return away_team, home_team
    else:
        return None, None              # draw — both teams get 1 point

def simulate_group_stage(model, le, fixtures_df, elo_dict, players_df):
    standings = {team: {'points': 0, 'wins': 0}
                 for teams in GROUPS.values() for team in teams}

    for _, fix in fixtures_df.iterrows():
        home, away, city = fix['home_team'], fix['away_team'], fix['city']
        probs  = predict_match_proba(model, le, home, away, elo_dict, players_df, city)
        winner, loser = simulate_match(probs, home, away)

        if winner is None:
            standings[home]['points'] += 1
            standings[away]['points'] += 1
        else:
            standings[winner]['points'] += 3
            standings[winner]['wins']   += 1

    return standings


def get_qualifiers(standings):
    qualifiers    = []
    third_place   = []

    for group_teams in GROUPS.values():
        ranked = sorted(group_teams,
                        key=lambda t: (standings[t]['points'], standings[t]['wins']),
                        reverse=True)
        qualifiers.append(ranked[0])   # 1st place
        qualifiers.append(ranked[1])   # 2nd place
        third_place.append(ranked[2])  # 3rd place — might advance

    # Best 8 third-place teams advance
    best_thirds = sorted(third_place,
                         key=lambda t: standings[t]['points'],
                         reverse=True)[:8]
    qualifiers.extend(best_thirds)
    return qualifiers   # 32 teams total

def simulate_group_stage_fast(fixtures_df, prob_cache):
    standings = {team: {'points': 0, 'wins': 0}
                 for teams in GROUPS.values() for team in teams}
    for _, fix in fixtures_df.iterrows():
        home, away = fix['home_team'], fix['away_team']
        probs  = prob_cache.get((home, away), {'home_win': 0.4, 'away_win': 0.4, 'draw': 0.2})
        winner, _ = simulate_match(probs, home, away)
        if winner is None:
            standings[home]['points'] += 1
            standings[away]['points'] += 1
        else:
            standings[winner]['points'] += 3
            standings[winner]['wins']   += 1
    return standings


def simulate_knockout(teams, prob_cache):
    while len(teams) > 1:
        next_round = []
        for i in range(0, len(teams), 2):
            home, away = teams[i], teams[i+1]
            # Cache stores each pair once. If only the reverse exists, we must
            # SWAP home/away win probs — otherwise the favorite gets flipped.
            if (home, away) in prob_cache:
                probs = prob_cache[(home, away)]
                hw, aw = probs['home_win'], probs['away_win']
            elif (away, home) in prob_cache:
                rev = prob_cache[(away, home)]
                hw, aw = rev['away_win'], rev['home_win']   # swapped perspective
            else:
                hw, aw = 0.4, 0.4
            total = hw + aw
            adj   = {'home_win': hw/total, 'away_win': aw/total, 'draw': 0.0}
            winner, _ = simulate_match(adj, home, away)
            next_round.append(winner)
        teams = next_round
    return teams[0]


def run_monte_carlo(n_simulations=10000):
    print("Loading model and data...")
    model, le = load_model()
    results, wc_fixtures, rankings, players, season = load_all()

    print("Computing current ELO ratings...")
    elo_history = compute_elo(results)
    elo_dict = {row['team']: row['elo']
                for _, row in elo_history.sort_values('date').iterrows()}

    # Precompute team strengths once — all 48 WC teams
    print("Precomputing team strengths...")
    all_wc_teams = {t for teams in GROUPS.values() for t in teams}
    strength_cache = {team: get_team_strength(players, team) for team in all_wc_teams}

    # Precompute all pairwise match probabilities once — same for every simulation
    print("Precomputing match probabilities for all team pairs...")
    all_teams_list = list(all_wc_teams)
    prob_cache = {}
    for i, home in enumerate(all_teams_list):
        for away in all_teams_list[i+1:]:
            hs  = strength_cache[home]
            as_ = strength_cache[away]
            elo_h = elo_dict.get(home, STARTING_ELO)
            elo_a = elo_dict.get(away, STARTING_ELO)
            city_key = 'New York'   # neutral default for knockout
            venue = VENUE_INFO.get(city_key, (0, 25, 60))
            features = pd.DataFrame([{
                'elo_diff': elo_h - elo_a, 'form_diff': 0,
                'attack_diff':  hs['attack_strength']    - as_['attack_strength'],
                'defense_diff': hs['defensive_strength'] - as_['defensive_strength'],
                'gk_diff':      hs['gk_rating']          - as_['gk_rating'],
                'depth_diff':   hs['squad_depth']         - as_['squad_depth'],
                'pace_diff':    hs['team_pace']           - as_['team_pace'],
                'stamina_diff': hs['team_stamina']        - as_['team_stamina'],
                'passing_diff': hs['passing_quality']     - as_['passing_quality'],
                'altitude': venue[0], 'heat': venue[1], 'humidity': venue[2],
                'is_wc': 1, 'neutral': 1,
            }])
            p = model.predict_proba(features)[0]
            prob_cache[(home, away)] = {'away_win': p[0], 'draw': p[1], 'home_win': p[2]}

    # Also precompute group stage fixture probabilities with correct venues
    for _, fix in wc_fixtures.iterrows():
        home, away, city = fix['home_team'], fix['away_team'], fix['city']
        city_key = CITY_MAP.get(city, city)
        venue = VENUE_INFO.get(city_key, (0, 25, 60))
        hs  = strength_cache.get(home, get_team_strength(players, home))
        as_ = strength_cache.get(away, get_team_strength(players, away))
        elo_h = elo_dict.get(home, STARTING_ELO)
        elo_a = elo_dict.get(away, STARTING_ELO)
        features = pd.DataFrame([{
            'elo_diff': elo_h - elo_a, 'form_diff': 0,
            'attack_diff':  hs['attack_strength']    - as_['attack_strength'],
            'defense_diff': hs['defensive_strength'] - as_['defensive_strength'],
            'gk_diff':      hs['gk_rating']          - as_['gk_rating'],
            'depth_diff':   hs['squad_depth']         - as_['squad_depth'],
            'pace_diff':    hs['team_pace']           - as_['team_pace'],
            'stamina_diff': hs['team_stamina']        - as_['team_stamina'],
            'passing_diff': hs['passing_quality']     - as_['passing_quality'],
            'altitude': venue[0], 'heat': venue[1], 'humidity': venue[2],
            'is_wc': 1, 'neutral': 1,
        }])
        p = model.predict_proba(features)[0]
        prob_cache[(home, away)] = {'away_win': p[0], 'draw': p[1], 'home_win': p[2]}

    print(f"Probability cache built for {len(prob_cache)} matchups.")
    champions = simulate_tournaments(prob_cache, wc_fixtures, n_simulations)
    return champions, n_simulations, prob_cache, wc_fixtures


def simulate_tournaments(prob_cache, wc_fixtures, n_simulations):
    """Run the tournament n times using a prebuilt probability cache.
    Uses the module-level TEMPERATURE so we can sweep values cheaply."""
    print(f"Running {n_simulations} simulations (temperature={TEMPERATURE})...")
    champions = {}
    for _ in range(n_simulations):
        standings  = simulate_group_stage_fast(wc_fixtures, prob_cache)
        qualifiers = get_qualifiers(standings)
        # Seed bracket: 1st place teams vs 2nd place from offset groups
        firsts  = qualifiers[0:24:2]
        seconds = qualifiers[1:24:2]
        thirds  = qualifiers[24:]
        offset  = 6
        bracket = []
        for j in range(12):
            bracket.append(firsts[j])
            bracket.append(seconds[(j + offset) % 12])
        bracket.extend(thirds)
        champion = simulate_knockout(bracket, prob_cache)
        champions[champion] = champions.get(champion, 0) + 1
    return champions


def print_results(champions, n_simulations):
    print(f"\n{'='*50}")
    print(f"2026 WORLD CUP PREDICTIONS ({n_simulations} simulations)")
    print(f"{'='*50}")
    sorted_champs = sorted(champions.items(), key=lambda x: x[1], reverse=True)
    print(f"\n{'Team':<25} {'Win %':>8}")
    print('-' * 35)
    for team, count in sorted_champs[:20]:
        pct = count / n_simulations * 100
        bar = '█' * int(pct / 2)
        print(f"{team:<25} {pct:>7.1f}%  {bar}")


def print_group_stage_predictions(prob_cache, fixtures_df):
    print(f"\n{'='*65}")
    print("GROUP STAGE MATCH PREDICTIONS")
    print(f"{'='*65}")
    for group_name, teams in GROUPS.items():
        print(f"\n--- Group {group_name}: {', '.join(teams)} ---")
        group_fixtures = fixtures_df[
            fixtures_df['home_team'].isin(teams) &
            fixtures_df['away_team'].isin(teams)
        ]
        for _, fix in group_fixtures.iterrows():
            home, away = fix['home_team'], fix['away_team']
            p = prob_cache.get((home, away),
                prob_cache.get((away, home), {'home_win':0.4,'draw':0.2,'away_win':0.4}))
            if (away, home) in prob_cache and (home, away) not in prob_cache:
                home_pct = p['away_win'] * 100
                away_pct = p['home_win'] * 100
            else:
                home_pct = p['home_win'] * 100
                away_pct = p['away_win'] * 100
            draw_pct = p['draw'] * 100
            print(f"  {fix['date'][5:]}  {home:<22} {home_pct:>5.1f}%"
                  f"  Draw {draw_pct:>4.1f}%  {away_pct:>5.1f}%  {away}")


if __name__ == '__main__':
    champions, n, prob_cache, wc_fixtures = run_monte_carlo(n_simulations=50000)
    print_results(champions, n)
    print_group_stage_predictions(prob_cache, wc_fixtures)

    Path('results').mkdir(exist_ok=True)
    pd.DataFrame(list(champions.items()),
                 columns=['team', 'wins']).sort_values('wins', ascending=False)\
      .to_csv('results/wc2026_predictions.csv', index=False)
    print("\nResults saved to results/wc2026_predictions.csv")


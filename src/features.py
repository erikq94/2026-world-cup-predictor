"""
features.py

Builds the feature matrix for model training:
  - ELO ratings (computed dynamically from full match history)
  - FIFA ranking differential
  - Team attack / defensive / GK strength (from FC26 player data)
  - Squad depth
  - Recent form (goal differential over last 6 matches)
  - Player availability (suspensions / injuries)
  - Tournament stage / match importance
"""
import pandas as pd
import numpy as np
from pathlib import Path
from data_loader import load_all

DATA_PROCESSED = Path('data/processed')

K_FACTOR = 30
STARTING_ELO = 1500

VENUE_INFO = {
    'New York':      (5,    26, 65),
    'Los Angeles':   (80,   24, 70),
    'Dallas':        (180,  35, 55),
    'San Francisco': (20,   17, 80),
    'Miami':         (3,    32, 80),
    'Atlanta':       (315,  30, 70),
    'Seattle':       (10,   18, 65),
    'Kansas City':   (275,  30, 65),
    'Houston':       (15,   34, 75),
    'Philadelphia':  (10,   27, 65),
    'Boston':        (10,   23, 65),
    'Mexico City':   (2240, 18, 50),
    'Guadalajara':   (1560, 28, 60),
    'Monterrey':     (538,  38, 55),
    'Toronto':       (76,   24, 65),
    'Vancouver':     (15,   18, 70),
}

def compute_elo(results_df):
    elo = {}      # current rating for every team
    history = []  # one record per match: team, date, elo after this match

    # Sort by date - must be processed in order
    matches = results_df[results_df['augmented'] == False].sort_values('date').reset_index(drop=True)

    for _, match in matches.iterrows():
        home = match['home_team']
        away = match['away_team']

        # Get current ratings - new teams start at STARTING_ELO
        elo_home = elo.get(home, STARTING_ELO)
        elo_away = elo.get(away, STARTING_ELO)

        # Expected probability home team wins (ELO formula)
        expected_home = 1 / (1 + 10 ** ((elo_away - elo_home) / 400))

        # Actual outcome: 1 = win, 0.5 = draw, 0 = loss
        if match['result'] == 'Home Win':
            actual_home = 1.0
        elif match['result'] == 'Draw':
            actual_home = 0.5
        else:
            actual_home = 0.0

        # Update both ratings
        elo[home] = elo_home + K_FACTOR * (actual_home - expected_home)
        elo[away] = elo_away + K_FACTOR * ((1 - actual_home) - (1 - expected_home))

        # Save a snapshot of both teams' ratings at this point in time.
        history.append({'team': home, 'date': match['date'], 'elo': elo[home]})
        history.append({'team': away, 'date': match['date'], 'elo': elo[away]})

    return pd.DataFrame(history)


def get_elo_at_date(elo_history, team, date):
    team_history = elo_history[
        (elo_history['team'] == team) &
        (elo_history['date'] < date)
    ]

    if team_history.empty:
        return STARTING_ELO
    return team_history.iloc[-1]['elo']

def compute_recent_form(results_df, team, date, n=6):
    # Get all matches this team played this date (real matches only)
    team_matches = results_df[
        (results_df['augmented'] == False) &
        (results_df['date'] < date ) & 
        ((results_df['home_team'] == team) | (results_df['away_team'] == team))
    ].sort_values('date').tail(n)

    if team_matches.empty:
        return 0.0
    
    goal_diff = 0
    for _, match in team_matches.iterrows():
        if match['home_team'] == team:
            goal_diff += match['home_score'] - match['away_score']
        else:
            goal_diff += match['away_score'] - match['home_score']

    return goal_diff

def get_team_strength(players_df, nationality):
    team = players_df[players_df['nationality'] == nationality]

    if team.empty:
        return {
            'attack_strength':    65.0,
            'defensive_strength': 65.0,
            'gk_rating':          65.0,
            'gk_height':          183.0,
            'squad_depth':        65.0,
            'passing_quality':    65.0,
            'team_pace':          65.0,
            'team_stamina':       65.0,
        }

    # Goalkeepers are mislabeled as positionType=='Defense' in FC26 — the only
    # reliable keeper flag is the 'position' column. Separate them first so they
    # don't pollute the defender pool and so gk_rating is actually a keeper.
    gks      = team[team['position'] == 'GK'].nlargest(3, 'overallRating')   # 3-keeper squad
    outfield = team[team['position'] != 'GK']

    # 26-man squad = best 3 GK + top 23 outfield; of the 23, top 11 start, 12 bench.
    squad_outfield = outfield.nlargest(23, 'overallRating')
    starting_xi    = squad_outfield.head(11)
    bench          = squad_outfield.tail(12)

    attackers   = squad_outfield[squad_outfield['positionType'] == 'Attack'].nlargest(5, 'overallRating')
    defenders   = squad_outfield[squad_outfield['positionType'] == 'Defense'].nlargest(4, 'overallRating')
    midfielders = squad_outfield[squad_outfield['positionType'] == 'Midfielder'].nlargest(4, 'overallRating')

    def safe_mean(df, col):
        return df[col].mean() if not df.empty else 65.0

    return {
        'attack_strength':    safe_mean(attackers,        'sho'),
        'defensive_strength': safe_mean(defenders,        'def'),
        'gk_rating':          safe_mean(gks.head(1),      'overallRating'),  # starting keeper
        'gk_height':          safe_mean(gks.head(1),      'height'),
        'squad_depth':        safe_mean(bench,            'overallRating'),
        'passing_quality':    safe_mean(midfielders,      'pas'),
        'team_pace':          safe_mean(starting_xi,      'pac'),
        'team_stamina':       safe_mean(starting_xi,      'stamina'),
    }


def build_feature_matrix(results_df, elo_history, rankings_df, players_df):
    # Precompute team strengths once
    all_teams = set(results_df['home_team'].tolist() + results_df['away_team'].tolist())
    team_strengths = {team: get_team_strength(players_df, team) for team in all_teams}

    # Precompute ELO as a dict {team: [(date, elo), ...]} — fast lookup vs scanning DataFrame
    elo_dict = {}
    for _, row in elo_history.sort_values('date').iterrows():
        elo_dict.setdefault(row['team'], []).append((row['date'], row['elo']))

    def fast_elo(team, date):
        entries = elo_dict.get(team, [])
        result = STARTING_ELO
        for d, e in entries:
            if d < date:
                result = e
            else:
                break
        return result

    # Precompute each team's real match history sorted by date — fast form lookup
    real_matches = results_df[results_df['augmented'] == False].sort_values('date')
    team_history = {}
    for _, match in real_matches.iterrows():
        for team in [match['home_team'], match['away_team']]:
            team_history.setdefault(team, []).append(match)

    def fast_form(team, date, n=6):
        history = [m for m in team_history.get(team, []) if m['date'] < date][-n:]
        if not history:
            return 0.0
        total = 0
        for match in history:
            if match['home_team'] == team:
                total += match['home_score'] - match['away_score']
            else:
                total += match['away_score'] - match['home_score']
        return total

    features = []
    for _, match in results_df.iterrows():
        home = match['home_team']
        away = match['away_team']
        date = match['date']

        hs  = team_strengths.get(home, get_team_strength(players_df, home))
        as_ = team_strengths.get(away, get_team_strength(players_df, away))
        city  = match.get('city', '')
        venue = VENUE_INFO.get(city, (0, 25, 60))

        features.append({
            'elo_diff':     fast_elo(home, date) - fast_elo(away, date),
            'form_diff':    fast_form(home, date) - fast_form(away, date),
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
            'is_wc':        1 if match['tournament'] == 'FIFA World Cup' else 0,
            'neutral':      1 if match['neutral'] else 0,
            'augmented':    match['augmented'],
            'result':       match['result'],
        })
    return pd.DataFrame(features)



if __name__ == '__main__':
    from tqdm import tqdm
    print("Loading data...")
    results, wc_fixtures, rankings, players, season = load_all()

    print("Computing ELO ratings across 30k matches (this takes ~2 mins)...")
    elo_history = compute_elo(results)
    print(f"  ELO computed for {elo_history['team'].nunique()} teams")

    print("Building feature matrix...")
    feature_matrix = build_feature_matrix(results, elo_history, rankings, players)
    feature_matrix.to_csv(DATA_PROCESSED / 'feature_matrix.csv', index=False)
    print(f"  Done. {len(feature_matrix)} rows, {len(feature_matrix.columns)} features")
    print(feature_matrix.head())

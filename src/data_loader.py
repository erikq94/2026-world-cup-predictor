"""
data_loader.py

Loads and cleans raw Kaggle datasets:
  - International match results (1872–2026)
  - FIFA World Rankings (1992–2024)
  - FC26 player ratings

Outputs: cleaned DataFrames ready for feature engineering in features.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
DATA_RAW = Path('data/raw')
DATA_PROCESSED = Path('data/processed')

# Standardize country names: FC26 use different names than results.csv
NATIONALITY_MAP = {
    'Holland':            'Netherlands',
    'Korea Republic':     'South Korea',
    "Côte d'Ivoire":      'Ivory Coast',
    'Congo DR':           'DR Congo',
    'Cape Verde Islands': 'Cape Verde',
    'IR Iran':            'Iran',
    'United States':      'USA',
}

def load_results():
    df = pd.read_csv(DATA_RAW / 'results.csv')

    # Seperate WC fixtures (no scores yet) from training data
    wc_fixtures = df[df['home_score'].isnull()].copy()
    df = df[df['home_score'].notna()].copy()

    # Modern era only - FIFA rankings after 1993
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df = df[df['year'] >= 1993].copy()

    # Create target variable - what the model predicts
    df['result'] = df.apply(
        lambda row: 'Home Win' if row['home_score'] > row['away_score']
        else ('Away Win' if row['home_score'] < row['away_score'] else 'Draw'),
        axis=1
    )

    return df, wc_fixtures

def load_rankings():
    df = pd.read_csv(DATA_RAW / 'fifa_ranking-2024-06-20.csv')
    df['rank_date'] = pd.to_datetime(df['rank_date'])
    df['country_full'] = df['country_full'].replace(NATIONALITY_MAP)
    return df

def load_players():
    df = pd.read_csv(DATA_RAW / 'ea_fc26_players.csv')
    df['nationality'] = df['nationality'].replace(NATIONALITY_MAP)
    df['height'] = pd.to_numeric(df['height'], errors='coerce')
    df['weight'] = pd.to_numeric(df['weight'], errors='coerce')
    return df

def load_season_stats():
    df = pd.read_csv(DATA_RAW / 'players_data-2025_2026.csv')
    # Nation column is "xx XXX" format — extract just the 3-letter code
    df['nation_code'] = df['Nation'].str.split().str[1]
    # Keep only what we need
    cols = ['Player', 'nation_code', 'Pos', 'Squad', 'Comp',
            'Min', 'Gls', 'Ast', 'CrdY', 'CrdR']
    return df[cols].copy()

def augment_results(df):
    flipped = df.copy()
    flipped['home_team']  = df['away_team']
    flipped['away_team']  = df['home_team']
    flipped['home_score'] = df['away_score']
    flipped['away_score'] = df['home_score']
    flipped['result'] = df['result'].map({
        'Home Win': 'Away Win',
        'Away Win': 'Home Win',
        'Draw':     'Draw'
    })
    flipped['augmented'] = True
    df = df.copy()
    df['augmented'] = False
    return pd.concat([df, flipped], ignore_index=True)

def load_all():
    DATA_PROCESSED.mkdir(exist_ok=True)

    print("Loading results...")
    results, wc_fixtures = load_results()
    results = augment_results(results)
    print(f"  After augmentation: {len(results)} training matches")
    results.to_csv(DATA_PROCESSED / 'results_clean.csv', index=False)
    wc_fixtures.to_csv(DATA_PROCESSED / 'wc_fixtures.csv', index=False)
    print(f"  {len(results)} training matches, {len(wc_fixtures)} WC fixtures")

    print("Loading rankings...")
    rankings = load_rankings()
    rankings.to_csv(DATA_PROCESSED / 'rankings_clean.csv', index=False)
    print(f"  {len(rankings)} ranking entries")

    print("Loading players...")
    players = load_players()
    players.to_csv(DATA_PROCESSED / 'players_clean.csv', index=False)
    print(f" {len(players)} players across {players['nationality'].nunique()} nationalities")

    print("Loading season stats...")
    season = load_season_stats()
    season.to_csv(DATA_PROCESSED / 'season_stats_clean.csv', index=False)
    print(f" {len(season)} player season records")

    print("Done. Processed files saved to data/processed/")
    return results, wc_fixtures, rankings, players, season

if __name__ == '__main__':
    load_all()
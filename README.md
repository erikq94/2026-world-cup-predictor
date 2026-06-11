# 2026 World Cup Predictor

XGBoost-based machine learning model that predicts match outcomes (Win/Draw/Loss) and simulates the full FIFA 2026 World Cup bracket across 48 teams and 104 matches.

## How It Works

1. Trains on 49,000+ historical international matches (1993–2026)
2. Builds team-level features: ELO ratings, FIFA rankings, player strength, form, altitude/climate
3. Simulates the full tournament bracket 10,000 times to produce win probabilities
4. Updates live after each match day — suspensions, injuries, and results feed back into re-predictions

## Data Sources

| Dataset | Source | What it provides |
|---|---|---|
| International match results (1872–2026) | Kaggle / martj42 | Training backbone — 49k matches |
| FIFA World Rankings (1992–2024) | Kaggle / cashncarry | Ranking differential feature |
| EA Sports FC 26 player ratings | Kaggle / justdhia | Individual player quality ratings |
| Football player stats 2025–26 | Kaggle / hubertsidorowicz | Current season form (goals, assists, minutes) |

## Known Data Limitations

**FC26 does not include LigaMX (Mexican domestic league).**
Mexico is the most affected team — only 27 players available in FC26, all playing in Europe or MLS.
Mitigation: ELO rating (computed from full match history) is the primary team strength signal and is unaffected by this gap. Player data is a supplementary feature. After the official 26-man squad announcement (June 2, 2026), missing LigaMX players will be manually supplemented with ratings from FUTBIN.

**Nationality naming inconsistencies across datasets** require a mapping table (e.g. FC26 uses "Holland" for Netherlands, "Korea Republic" for South Korea, "Côte d'Ivoire" for Ivory Coast). This is handled in `data_loader.py` via a `NATIONALITY_MAP` dictionary applied at load time.

**Teams with thin FC26 coverage** use a three-tier fallback:
- 20+ players → full feature set
- 5–19 players → top players only, squad depth flagged unreliable  
- 0 players (Qatar) → ELO + FIFA ranking only, player features skipped

**Qatar has zero FC26 coverage** — their domestic Stars League is not licensed by EA Sports.

**FIFA rankings only available through June 2024.**
Mitigation: ELO ratings computed from match results through March 2026 serve as the primary recency signal.

**FC26 ratings reflect EA Sports assessments, not official FIFA data.**
They are widely regarded as accurate proxies for player quality but are not authoritative.

## Project Structure

```
data/
  raw/          <- Kaggle downloads (gitignored)
  processed/    <- cleaned, merged features
  squads/       <- official 26-man WC rosters (added June 2, 2026)
src/
  data_loader.py   <- loads and cleans raw datasets
  features.py      <- engineers ELO, player strength, form, venue features
  model.py         <- trains and evaluates XGBoost classifier
  simulate.py      <- simulates full tournament bracket
notebooks/
  01_data_exploration.ipynb  <- exploratory data analysis
results/          <- prediction outputs
```

## Feature Engineering

- **ELO differential** — computed from full match history, updates after every match
- **FIFA ranking differential** — secondary team strength signal  
- **Attack strength** — avg shooting rating of top 5 forwards (FC26)
- **Defensive strength** — avg defending rating of top 4 defenders (FC26)
- **GK rating + height** — goalkeeper quality and aerial coverage
- **Squad depth** — avg rating of bench players (squad positions 12–26)
- **Current form** — goals scored/conceded over last 6 matches
- **Venue altitude** — meters above sea level (significant for Mexico City at 2,240m)
- **Venue heat index** — June climate score (Monterrey reaches 38–40°C)
- **Climate advantage** — how adapted each team is to the venue's climate
- **Player availability** — rating points lost to suspensions/injuries

## Planned Improvements (Post-MVP)

These are intentionally deferred — build something working first, then enhance:

1. **Player trajectory feature** — download FC24 + FC25, compute rating change per player over 3 editions (rising vs declining form). Weight FC26 highest, FC24 lowest.
2. **Weather integration** — Open-Meteo API has free historical weather back to 1940. Retro-fetch temperature + humidity for training matches, add forecast 2 days before each WC match.
3. **StatsBomb xG features** — install `statsbombpy`, pull team-level expected goals from past World Cups and recent internationals. Better than raw goals for attack strength.
4. **Venue closed/open** — hardcode retractable roof status for AT&T Stadium (Dallas), NRG Stadium (Houston), BC Place (Vancouver), SoFi Stadium (LA). Closed = no heat penalty.
5. **LigaMX player supplementation** — after June 2 squad announcement, manually look up missing Mexican players on FUTBIN and add to a `data/squads/mexico_supplement.json`.
6. **API-Football live integration** — use free tier (100 req/day) for confirmed lineups and injury updates during the tournament. Wire into simulate.py re-run pipeline.
7. **soccerdata FBref scraper** — `pip install soccerdata`, pull per-player progressive passes, pressures, xA for richer midfield features.
8. **Coach tenure feature** — how long has the current coach been in charge? New coaches = higher variance in results.

## WC 2026 Timeline

- May 28, 2026 — model training in progress
- June 2, 2026 — official 26-man squads announced, player features locked in
- June 9–10, 2026 — final predictions published before tournament
- June 11, 2026 — tournament begins, live re-simulation after each match day

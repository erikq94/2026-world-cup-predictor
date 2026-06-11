# Findings & Analysis — 2026 World Cup Predictor

A write-up of what we built, what worked, what didn't, and — most importantly —
*why*. The headline takeaway is a deliberately honest one: not every feature we
added improved the model, and understanding **why** taught us more than a lucky
result would have.

---

## 1. Goal & Approach

Predict 2026 World Cup match outcomes (Win / Draw / Loss) and simulate the full
48-team tournament to produce winner probabilities.

**Pipeline:**
```
Kaggle data → clean & standardize → feature engineering → XGBoost → Monte Carlo simulation
```

- **Training data:** ~30,000 international matches (1993–2026), doubled to ~61,000
  via home/away augmentation.
- **Model:** XGBoost multi-class classifier.
- **Core feature:** ELO rating differential, computed from scratch by replaying
  every match in chronological order (K=30, start=1500, scale=400).
- **Tournament:** 50,000 Monte Carlo simulations sampling from the model's
  probabilities.

---

## 2. Results

**Model performance (5-fold cross-validation):**

| Metric | Value | Benchmark |
|--------|-------|-----------|
| Accuracy | 56.5% | Random = 33%, always-majority = 38% |
| Log-loss | 0.930 | (calibration metric) |
| Draw calibration | predicted 23.3% vs actual 23.3% | — |

**Tournament winner probabilities (top of the table) vs. the betting market:**

| Team | Our model | Bookmaker implied | 
|------|-----------|-------------------|
| Spain | ~21% | ~18% |
| Argentina | ~16% | ~9% |
| France | ~13% | ~17% |
| Brazil | ~6% | ~10% |
| England | ~5% | ~12% |

Our **top favorites match the global betting market** — Spain #1, France and
Argentina in the leading tier. Divergences (we rate Argentina higher, England
lower) trace directly to our reliance on **ELO** (results-based) vs. bookmakers'
weighting of **squad reputation** — an explainable property, not a bug.

The ELO ratings alone reproduce the real FIFA hierarchy almost exactly:
Spain, Argentina, France, Brazil, Portugal, England, Germany at the top.

---

## 3. Bugs Found & Fixed

Three real correctness issues surfaced during development. Finding them was
itself a large part of the learning.

### 3.1 Draw probabilities were inflated (calibration vs. accuracy)
We initially used `class_weight='balanced'` to improve draw *recall*. This
inflated predicted draws to **38%** (reality: 23%). But the simulation **samples
from `predict_proba`** — it never takes the single most-likely label — so what
matters is **calibration**, not recall. Removing the class weighting restored
calibration (predicted draw rate = actual, 23.3%) with no loss of accuracy.
> **Lesson:** optimize the metric your downstream system actually uses. Ours
> samples probabilities, so calibration (log-loss) matters, not classification
> accuracy/recall.

### 3.2 Knockout home/away probability flip
The match-probability cache stored each pairing once (e.g. `(Spain, Brazil)`).
When the knockout drew the reverse order `(Brazil, Spain)`, a fallback lookup
returned the stored dict **without swapping** home/away win probabilities — so in
~half of knockout games the **weaker team was favored**. This scrambled the
bracket and pushed mid-tier teams (Morocco, Canada) above Spain/France/Argentina.
Fixing the swap snapped the winner distribution back to a sensible ordering.
> **Lesson:** a silent logic bug with no error message is the most dangerous
> kind. It only showed up as "the results look wrong," not a crash.

### 3.3 Goalkeepers were mislabeled
FC26's `positionType` column has only three values — Defense, Midfielder,
Attack — and labels **all 1,816 goalkeepers as "Defense."** Our code filtered
keepers by `positionType == 'Goalkeeper'`, which matched nothing, so:
- `gk_rating` fell back to a constant 65 for **every** team (the feature was dead)
- keepers polluted the **defender** pool, corrupting `defensive_strength`

The fix: identify keepers by the `position == 'GK'` column instead. This made
`gk_diff` meaningful for all 48 teams (e.g., Brazil's Alisson now correctly
rates 89 vs. the old constant 65).
> **Lesson:** validate your assumptions about the data's encoding before trusting
> a feature. We only caught this by inspecting individual players.

---

## 4. The Headline Finding: more features ≠ better

After fixing the goalkeeper bug and restructuring the player aggregation into a
proper 26-man squad model (3 GK + 11 starters + 12 bench), we retrained and
re-evaluated:

| | Before player fixes | After player fixes |
|--|--|--|
| CV Accuracy | 0.566 | 0.565 |
| Log-loss | 0.930 | 0.930 |

**Accuracy did not improve.** This is the most valuable result in the project.

**Why:** the **ELO rating already encodes team strength**, because it is computed
from *actual match results* — results that already reflect the goalkeeper, the
defense, and every player. The player-rating features are therefore **collinear**
with ELO: Brazil has elite keepers *and* a high ELO precisely because those
keepers helped produce the results that built the ELO. Adding the player features
contributes almost no *independent* signal.

> **Lesson:** a feature only helps if it adds information the model doesn't
> already have. We measured this instead of assuming "more features = better,"
> got a null result, and understood the cause. The player fixes remain in the
> codebase as correctness improvements, not accuracy improvements.

Where player data *would* add unique value is **live availability** — ELO cannot
know that a player is suspended for the next match. That (not accuracy) is the
real use case for a squad/roster layer, and is listed as future work.

---

## 5. Calibration at Two Levels (Temperature)

The per-match model is well-calibrated, but compounding 7 independent matches
into a tournament **over-concentrated** probability on the favorites (Spain ~24%
vs. a bookmaker spread of ~18%). Real tournaments contain chaos the per-match
model can't see (injuries, red cards, single-elimination variance). We added a
**temperature** parameter that softens the probabilities used in the simulation
(T=1.3), bringing the winner distribution in line with the market — while leaving
the printed per-match predictions at their raw, calibrated values.
> **Lesson:** per-match calibration and tournament-level calibration are
> different things. Separating them is a deliberate modeling choice.

---

## 6. Honest Limitations

- **FC26 league coverage:** no Liga MX or Qatari league, so some squads (Mexico,
  Qatar) are thin or absent. ELO is the fallback for those teams.
- **Retroactive player ratings:** historical matches use 2026 FC26 ratings, so
  the player features are near-constant per team across history — part of why
  they're collinear with ELO.
- **FIFA rankings only through 2024:** ELO (computed through 2026) is the primary
  recency signal instead.
- **Bracket seeding is simplified** and doesn't replicate FIFA's exact seeding
  rules, which slightly affects favorites' paths.

---

## 7. Future Work

1. Live availability layer (suspensions/injuries) — the real use case for
   player-level data; requires roster→FC26 name matching.
2. Player trajectory features (FC24/FC25) — rising vs. declining form, which
   *would* add signal independent of ELO.
3. Weather integration (Open-Meteo) with historical backfill.
4. `evaluate.py` — score predictions against actual results as the tournament
   unfolds (live backtesting).
5. Streamlit dashboard for interactive, auto-updating predictions.

---

*This project's value is not a single prediction but a reproducible, debugged,
and honestly-evaluated pipeline — and a clear understanding of why each piece
behaves the way it does.*

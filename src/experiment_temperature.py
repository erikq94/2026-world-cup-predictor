"""
experiment_temperature.py

Sweeps the TEMPERATURE knob to see how it changes the World Cup winner
distribution. Builds the expensive probability cache ONCE, then re-simulates
cheaply at each temperature.

  T = 1.0  -> model's calibrated probabilities (baseline)
  T > 1.0  -> more upsets, favorites less dominant
  T < 1.0  -> favorites crush everyone
"""
import simulate as sim

N = 50000
TEMPS = [0.8, 1.0, 1.3, 1.6]

# Build the cache once by running a throwaway 1-sim pass (it returns the cache)
print("Building probability cache once...")
_, _, prob_cache, wc_fixtures = sim.run_monte_carlo(n_simulations=1)

# Run the full sim at each temperature, store champion counts
results_by_temp = {}
for T in TEMPS:
    sim.TEMPERATURE = T
    champions = sim.simulate_tournaments(prob_cache, wc_fixtures, N)
    results_by_temp[T] = champions

# Compact side-by-side comparison of the top teams
all_top = set()
for champs in results_by_temp.values():
    top = sorted(champs.items(), key=lambda x: -x[1])[:12]
    all_top.update(t for t, _ in top)

# Order rows by the baseline (T=1.0) ranking
baseline = results_by_temp[1.0]
ordered = sorted(all_top, key=lambda t: -baseline.get(t, 0))

header = f"{'Team':<16}" + "".join(f"T={T:<6}" for T in TEMPS)
print("\n" + "=" * len(header))
print("WIN % BY TEMPERATURE")
print("=" * len(header))
print(header)
print("-" * len(header))
for team in ordered:
    row = f"{team:<16}"
    for T in TEMPS:
        pct = results_by_temp[T].get(team, 0) / N * 100
        row += f"{pct:>5.1f}%  "
    print(row)

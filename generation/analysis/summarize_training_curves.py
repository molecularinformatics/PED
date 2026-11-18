"""Convergence summary of the reward curves dumped by dump_training_curves.py.

Average-score values are NOT comparable across arms (each arm's reward contains a
different similarity term), so the comparable quantity is how fast each arm reaches
its own plateau. plateau = mean over the last 10% of steps; t90 = first step whose
5-step moving average reaches 90% of that plateau.
"""
import pandas as pd

df = pd.read_csv(str(Path(__file__).resolve().parent / "training_curves_avg_score.csv"))
rows = []
for (model, method), g in df.groupby(["Model", "Method"], sort=False):
    g = g.sort_values("step")
    n = len(g)
    plateau = g["mean"].iloc[int(n * 0.9):].mean()
    sm = g["mean"].rolling(5, min_periods=1).mean()
    reached = g["step"][sm >= 0.9 * plateau]
    rows.append({
        "Model": model, "Method": method, "steps": n,
        "start": round(g["mean"].iloc[0], 3),
        "plateau": round(plateau, 3),
        "t90": int(reached.iloc[0]) if len(reached) else None,
        "t90_frac": round(reached.iloc[0] / g["step"].max(), 3) if len(reached) else None,
    })
out = pd.DataFrame(rows)
print(out.to_string(index=False))

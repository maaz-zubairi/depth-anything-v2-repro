"""Answer the experiment question from results/da2k_pairs.csv.

Question: does V2's accuracy gain over V1 concentrate on transparent_reflective scenes?
For every size: per-scene gain (V2 - V1, paired by pair), bootstrap 95% CI, and the
difference-in-differences (gain on transparent_reflective minus gain on all other scenes).
Run:  python scripts/05_analyze_da2k.py
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

TARGET = "transparent_reflective"
rng = np.random.default_rng(0)
B = 5000
df = pd.read_csv("results/da2k_pairs.csv")
wide = df.pivot_table(index=["scene", "image", "pair"], columns="model", values="correct").reset_index()


def boot_ci(x, n=B):
    x = np.asarray(x, float)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


rows, did_rows = [], []
for size in ["small", "base", "large"]:
    a, b = f"v1-{size}", f"v2-{size}"
    if a not in wide or b not in wide:
        continue
    w = wide.dropna(subset=[a, b]).copy()
    w["gain"] = w[b] - w[a]
    for scene, g in list(w.groupby("scene")) + [("ALL", w)]:
        lo, hi = boot_ci(g["gain"])
        rows.append({"size": size, "scene": scene, "n_pairs": len(g), "acc_v1": g[a].mean(), "acc_v2": g[b].mean(),
                     "gain": g["gain"].mean(), "ci_lo": lo, "ci_hi": hi})
    t, o = w[w.scene == TARGET]["gain"].to_numpy(), w[w.scene != TARGET]["gain"].to_numpy()
    if len(t) and len(o):
        did = [rng.choice(t, len(t)).mean() - rng.choice(o, len(o)).mean() for _ in range(B)]
        did_rows.append({"size": size, "gain_target": t.mean(), "gain_other": o.mean(),
                         "did": t.mean() - o.mean(), "ci_lo": np.percentile(did, 2.5), "ci_hi": np.percentile(did, 97.5)})

res, did = pd.DataFrame(rows), pd.DataFrame(did_rows)
res.to_csv("results/da2k_gain_by_scene.csv", index=False)
did.to_csv("results/da2k_did.csv", index=False)
print(res.round(3).to_string(index=False)); print("\nDifference-in-differences:\n", did.round(3).to_string(index=False))

# long-format overall/per-scene accuracy so it can join with paper numbers
acc = df.groupby(["model", "scene"])["correct"].mean().reset_index()
overall = df.groupby("model")["correct"].mean().reset_index().assign(scene="overall")
pd.concat([acc, overall]).rename(columns={"correct": "ours"}).assign(
    dataset="da2k", metric=lambda d: "acc_" + d["scene"]).drop(columns="scene").to_csv(
    "results/da2k_summary.csv", index=False)

# figure: gain per scene, one panel per size
sizes = res["size"].unique()
fig, axs = plt.subplots(1, len(sizes), figsize=(5 * len(sizes), 4), sharey=True, squeeze=False)
for ax, size in zip(axs[0], sizes):
    r = res[res["size"] == size].sort_values("scene")
    colors = ["#d9480f" if s == TARGET else "#495057" for s in r["scene"]]
    ax.bar(r["scene"], r["gain"] * 100, color=colors,
           yerr=[(r["gain"] - r["ci_lo"]) * 100, (r["ci_hi"] - r["gain"]) * 100], capsize=3)
    ax.axhline(0, color="k", lw=0.8); ax.set_title(f"{size}: V2 - V1 pair accuracy (pp)")
    ax.tick_params(axis="x", rotation=60)
axs[0][0].set_ylabel("accuracy gain (percentage points)")
plt.tight_layout(); plt.savefig("figures/da2k_gain_by_scene.png", dpi=200)
print("saved figures/da2k_gain_by_scene.png")

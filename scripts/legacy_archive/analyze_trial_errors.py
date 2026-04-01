import json
import numpy as np

with open("results/downstream_comprehensive_eval.json") as f:
    data = json.load(f)

EXCLUDE = {"Oracle_GT", "Atlas"}
pairs = []
for task in data:
    for rep, rec in data[task].items():
        if rep in EXCLUDE:
            continue
        for t in rec.get("trials", []):
            pairs.append({
                "task": task,
                "rep": rep,
                "geo_error": t["geo_error"],
                "success": int(t["success"]),
                "region": t.get("region_type", "unknown")
            })

pairs.sort(key=lambda x: x["geo_error"])
errors  = np.array([p["geo_error"]  for p in pairs])
successes = np.array([p["success"]  for p in pairs])

print(f"Total trials: {len(pairs)}")
print(f"Error range: {errors.min():.2f} - {errors.max():.2f}")
print()

# Bin by error magnitude and show success rate per task
bins = [0, 2, 5, 10, 15, 25, 50, 100, 200]
print(f"{'Error Bin':<22} {'N':>4}  {'Overall%':>10}  {'PickPlace%':>11}  {'Stack%':>8}")
print("-" * 60)
for lo, hi in zip(bins, bins[1:]):
    mask = (errors >= lo) & (errors < hi)
    n = mask.sum()
    if n == 0:
        continue
    sr_all = successes[mask].mean() * 100
    pp_mask = mask & np.array([p["task"] == "PickPlace" for p in pairs])
    st_mask = mask & np.array([p["task"] == "Stack" for p in pairs])
    sr_pp = successes[pp_mask].mean() * 100 if pp_mask.sum() > 0 else float("nan")
    sr_st = successes[st_mask].mean() * 100 if st_mask.sum() > 0 else float("nan")
    print(f"{lo:4.0f} - {hi:4.0f} deg          {n:4d}    {sr_all:6.1f}%     {sr_pp:6.1f}%   {sr_st:6.1f}%")

print()
print("Per-rep error vs success summary (excluding Atlas/Oracle):")
print(f"{'Rep':<15} {'Task':<12} {'N':>3}  {'mean_err':>9}  {'max_err':>9}  {'success%':>9}")
print("-" * 60)
for rep in ["SVD", "6D", "Lie_FullFix", "Quat", "Euler"]:
    for task in sorted(data.keys()):
        if rep not in data[task]: continue
        trials = data[task][rep].get("trials", [])
        if not trials: continue
        errs = [t["geo_error"] for t in trials]
        suc  = [t["success"]   for t in trials]
        print(f"{rep:<15} {task:<12} {len(errs):3d}  {np.mean(errs):9.2f}  {np.max(errs):9.2f}  {np.mean(suc)*100:9.1f}%")

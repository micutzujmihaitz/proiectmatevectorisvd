"""
Evaluare in 3 parti:
  1. Cat de bine aproximeaza SVD-ul de rang k matricea de ascultari.
  2. ANN vs cautarea exacta: recall@30, cat din catalog se scaneaza, timp per interogare.
  3. Calitatea recomandarilor pe date ascunse (holdout) vs baseline-uri.

    python evaluate.py           # 5.000 piese, 2.000 utilizatori
    python evaluate.py --big     # 20.000 piese, 6.000 utilizatori
"""
import argparse
import os
import time

# Pentru produse matrice-vector mici, pornirea thread-urilor BLAS costa mai mult decat calculul
# si face timpii instabili. Un singur thread => o comparatie corecta intre metode.
for _var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402  (trebuie importat DUPA setarea variabilelor de mai sus)

from discover_weekly import (DiscoverWeekly, HyperplaneLSHIndex, IVFIndex,  # noqa: E402
                             generate, split_holdout)

K = 30


def recall(found, truth) -> float:
    return len(np.intersect1d(found, truth)) / len(truth) if len(truth) else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--big", action="store_true", help="catalog de 20.000 piese / 6.000 utilizatori")
    ap.add_argument("--users", type=int, default=300, help="cati utilizatori se evalueaza")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = dict(n_users=6000, artists_per_genre=40, tracks_per_artist=50) if args.big else {}
    ds = generate(seed=args.seed, **cfg)
    cat = ds.catalog
    n_users, n_tracks = ds.plays.shape
    rng = np.random.default_rng(args.seed + 1)
    users = rng.choice(n_users, min(args.users, n_users), replace=False)
    print(f"Date: {n_users} utilizatori x {n_tracks} piese, {ds.plays.nnz} perechi (densitate "
          f"{ds.plays.nnz / (n_users * n_tracks):.2%}); evaluam pe {len(users)} utilizatori\n")

    # ------------------------------------------------------------------ 1. SVD
    model = DiscoverWeekly(index="exact", seed=args.seed).fit(ds.plays, cat.track_artist)
    S = model.singular_values
    energy = (S ** 2).sum() / (model.weights.data ** 2).sum()
    print(f"[1] SVD de rang k={len(S)}: pastreaza {energy:.1%} din ||W||_F^2")
    print("    primele valori singulare: " + " ".join(f"{s:.1f}" for s in S[:12]) + " ...\n")

    # ------------------------------------------------------------------ 2. ANN
    V = model.track_vectors
    queries = [(model.taste_vector(u), model.heard(u)) for u in users]
    truth = [model.exact_index.search(q, K, exclude=h).ids for q, h in queries]
    configs = [("exact (brute force)", lambda: model.exact_index)]
    for L, b, p in [(16, 10, 3), (32, 6, 2)]:
        configs.append((f"LSH  L={L} b={b} probes={p}",
                        lambda L=L, b=b, p=p: HyperplaneLSHIndex(V, n_tables=L, n_bits=b, n_probes=p)))
    for npr in [16, 32, 48]:
        configs.append((f"IVF  C={int(round(2 * np.sqrt(n_tracks)))} n_probe={npr}",
                        lambda npr=npr: IVFIndex(V, n_probe=npr)))

    print(f"[2] Cautare: top-{K} piese neascultate, comparat cu rezultatul exact")
    print(f"    {'metoda':28s} {'recall@30':>9s} {'scanat':>8s} {'us/query':>9s} {'constructie':>12s}")
    for name, build in configs:
        t0 = time.perf_counter()
        index = build()
        build_s = time.perf_counter() - t0
        rec, scanned, times = [], [], []
        for (q, h), t in zip(queries, truth):
            t0 = time.perf_counter()
            res = index.search(q, K, exclude=h)
            times.append(time.perf_counter() - t0)
            rec.append(recall(res.ids, t))
            scanned.append(res.n_scored)
        print(f"    {name:28s} {np.mean(rec):9.3f} {np.mean(scanned) / n_tracks:8.1%} "
              f"{np.median(times) * 1e6:9.0f} {build_s:11.2f}s")
    print()

    # ------------------------------------------------------------------ 3. holdout
    train, test = split_holdout(ds.plays, fraction=0.2, seed=args.seed)
    popularity = np.asarray(train.sum(axis=0)).ravel()
    liked = ds.user_genre_pref >= 0.10            # genurile "adevarate" ale fiecaruia (ascunse)
    systems = {
        "SVD + IVF (playlistul final)": DiscoverWeekly(index="ivf", seed=args.seed).fit(train, cat.track_artist),
        "SVD + cautare exacta": DiscoverWeekly(index="exact", seed=args.seed).fit(train, cat.track_artist),
    }
    rows = {name: ([], []) for name in [*systems, "cele mai populare piese", "aleator"]}
    for u in users:
        heard = train[u].indices
        for name, m in systems.items():
            ids = m.recommend(u, n=K).track_ids
            rows[name][0].append(recall(ids, test[u]))
            rows[name][1].append(liked[u, cat.track_genre[ids]].mean())
        pop = popularity.astype(float).copy()
        pop[heard] = -np.inf
        ids = np.argsort(-pop, kind="stable")[:K]
        rows["cele mai populare piese"][0].append(recall(ids, test[u]))
        rows["cele mai populare piese"][1].append(liked[u, cat.track_genre[ids]].mean())
        ids = rng.choice(np.setdiff1d(np.arange(n_tracks), heard), K, replace=False)
        rows["aleator"][0].append(recall(ids, test[u]))
        rows["aleator"][1].append(liked[u, cat.track_genre[ids]].mean())

    print("[3] Recomandari pe date ascunse (20% din piesele fiecaruia scoase inainte de antrenare)")
    print(f"    {'sistem':30s} {'recall@30':>9s} {'din genurile lui':>17s}")
    for name, (r, g) in rows.items():
        print(f"    {name:30s} {np.nanmean(r):9.3f} {np.mean(g):17.1%}")


if __name__ == "__main__":
    main()

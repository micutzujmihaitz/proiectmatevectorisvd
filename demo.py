"""
Demo: genereaza date, antreneaza modelul si afiseaza playlistul Discover Weekly al unui utilizator.

    python demo.py                 # utilizatorul 7, index IVF
    python demo.py --user 42 --index lsh
"""
import argparse

import numpy as np

from discover_weekly import DiscoverWeekly, generate


def main():
    ap = argparse.ArgumentParser(description="Discover Weekly MVP")
    ap.add_argument("--user", type=int, default=7)
    ap.add_argument("--n", type=int, default=30, help="lungimea playlistului")
    ap.add_argument("--index", choices=["ivf", "lsh", "exact"], default="ivf")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ds = generate(seed=args.seed)
    cat = ds.catalog
    n_users, n_tracks = ds.plays.shape
    if not 0 <= args.user < n_users:
        ap.error(f"--user trebuie sa fie intre 0 si {n_users - 1}")
    model = DiscoverWeekly(index=args.index, seed=args.seed).fit(ds.plays, cat.track_artist)
    u = args.user

    row = ds.plays[u]
    genre_share = np.bincount(cat.track_genre[row.indices], weights=row.data,
                              minlength=len(cat.genre_names)) / row.data.sum()
    artist_plays = np.bincount(cat.track_artist[row.indices], weights=row.data,
                               minlength=len(cat.artist_names))

    print(f"Catalog: {n_tracks} piese, {len(cat.artist_names)} artisti, {n_users} utilizatori")
    print(f"\n=== Istoricul utilizatorului #{u}: {int(row.data.sum())} ascultari, "
          f"{row.nnz} piese distincte ===")
    print("Genuri: " + ", ".join(f"{cat.genre_names[g]} {genre_share[g]:.0%}"
                                 for g in np.argsort(-genre_share) if genre_share[g] >= 0.03))
    print("Artisti preferati: " + ", ".join(f"{cat.artist_names[a]} ({int(artist_plays[a])})"
                                            for a in np.argsort(-artist_plays)[:5]))

    pl = model.recommend(u, n=args.n)
    known_artists = set(np.flatnonzero(artist_plays).tolist())
    print(f"\n=== Discover Weekly pentru #{u} (index: {args.index}) ===")
    for i, (t, s) in enumerate(zip(pl.track_ids, pl.scores), 1):
        new = "  * artist nou" if cat.track_artist[t] not in known_artists else ""
        print(f"{i:2d}. {cat.describe(t):52s} scor {s:.3f}{new}")

    rec_share = np.bincount(cat.track_genre[pl.track_ids], minlength=len(cat.genre_names)) / len(pl.track_ids)
    print("\nGenurile din playlist: " + ", ".join(f"{cat.genre_names[g]} {rec_share[g]:.0%}"
                                                 for g in np.argsort(-rec_share) if rec_share[g] > 0))
    print(f"Cautarea a evaluat {pl.n_scored} din {n_tracks} piese ({pl.n_scored / n_tracks:.1%} din catalog)"
          + ("  [fallback la cautarea exacta]" if pl.used_fallback else ""))


if __name__ == "__main__":
    main()

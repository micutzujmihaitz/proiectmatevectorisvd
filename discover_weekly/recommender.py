"""
Discover Weekly (MVP): SVD pentru embedding-uri + ANN pentru cautare.

Vectorul de gust al unui utilizator cu randul de ascultari ponderat w (1 x n_piese):
    q = w V_k                         (proiectia istoricului pe spatiul latent de dimensiune k)
Scorul piesei j:
    q . v_j = (w V_k V_k^T)_j         = intrarea (u, j) din aproximarea de rang k W_k
Adica: cat de mult "ar trebui" sa asculte u piesa j, judecand dupa tiparele intregii
populatii. Cautam piesele cu scorul maxim pe care utilizatorul NU le-a ascultat inca.
Formula merge si pentru un utilizator nou (care nu era in date la antrenare).
"""
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .ann import BruteForceIndex, HyperplaneLSHIndex, IVFIndex
from .embeddings import log_weights, randomized_svd

INDEXES = {"exact": BruteForceIndex, "ivf": IVFIndex, "lsh": HyperplaneLSHIndex}


@dataclass
class Playlist:
    track_ids: np.ndarray
    scores: np.ndarray
    n_scored: int           # cate piese din catalog a evaluat cautarea
    used_fallback: bool     # True daca ANN n-a gasit destui candidati si s-a folosit cautarea exacta


class DiscoverWeekly:
    def __init__(self, n_factors: int = 32, index: str = "ivf", index_params=None,
                 pool_factor: int = 4, seed: int = 0):
        if index not in INDEXES:
            raise ValueError(f"index trebuie sa fie unul din {sorted(INDEXES)}")
        self.n_factors = n_factors
        self.index_name = index
        self.index_params = dict(index_params or {})
        if index != "exact":
            self.index_params.setdefault("seed", seed)
        self.pool_factor = pool_factor
        self.seed = seed

    def fit(self, plays, track_artist=None) -> "DiscoverWeekly":
        self.plays = sp.csr_matrix(plays)
        self.weights = log_weights(self.plays)
        _, self.singular_values, Vt = randomized_svd(self.weights, self.n_factors, seed=self.seed)
        self.track_vectors = np.ascontiguousarray(Vt.T)        # (n_piese, k): randurile lui V_k
        self.exact_index = BruteForceIndex(self.track_vectors)
        self.index = (self.exact_index if self.index_name == "exact"
                      else INDEXES[self.index_name](self.track_vectors, **self.index_params))
        self.track_artist = None if track_artist is None else np.asarray(track_artist)
        return self

    def taste_vector(self, user_id: int) -> np.ndarray:
        return np.asarray(self.weights[user_id] @ self.track_vectors).ravel()

    def heard(self, user_id: int) -> np.ndarray:
        row = self.plays[user_id]
        return row.indices[row.data > 0]

    def recommend(self, user_id: int, n: int = 30, max_per_artist=2) -> Playlist:
        q = self.taste_vector(user_id)
        heard = self.heard(user_id)
        k = n * self.pool_factor
        pool = self.index.search(q, k=k, exclude=heard)
        picked, satisfied = self._diversify(pool.ids, n, max_per_artist)
        used_fallback = False
        # Daca ANN n-a dat destui candidati (sau prea putini artisti diferiti), cautam exact.
        if not satisfied and self.index is not self.exact_index:
            n_scored = pool.n_scored
            pool = self.exact_index.search(q, k=k, exclude=heard)
            pool.n_scored += n_scored
            picked, _ = self._diversify(pool.ids, n, max_per_artist)
            used_fallback = True
        return Playlist(pool.ids[picked], pool.scores[picked], pool.n_scored, used_fallback)

    def _diversify(self, ids: np.ndarray, n: int, max_per_artist):
        """
        Primele n piese (dupa scor) cu cel mult `max_per_artist` de la acelasi artist.
        Intoarce (pozitii alese, regula_respectata). Daca nu sunt destui artisti diferiti,
        regula se relaxeaza (se completeaza cu piesele sarite), ca playlistul sa aiba n piese.
        """
        if self.track_artist is None or max_per_artist is None:
            picked = np.arange(min(n, len(ids)))
            return picked, len(picked) == n
        per_artist, picked, skipped = {}, [], []
        for pos, t in enumerate(ids):
            a = self.track_artist[t]
            if per_artist.get(a, 0) < max_per_artist:
                per_artist[a] = per_artist.get(a, 0) + 1
                picked.append(pos)
                if len(picked) == n:
                    break
            else:
                skipped.append(pos)
        satisfied = len(picked) == n
        if not satisfied:
            picked = sorted(picked + skipped[:n - len(picked)])
        return np.array(picked, dtype=np.int64), satisfied

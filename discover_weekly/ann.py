"""
Cautarea vectorilor: exacta (brute force) si aproximativa (ANN).

Problema (Maximum Inner Product Search, MIPS): dat vectorul de gust q, gaseste piesele j
cu produsul scalar q . v_j maxim, fara sa calculam scorul pentru TOATE piesele.

Doua metode ANN, ambele cu reordonare exacta a candidatilor la final:

* IVFIndex (inverted file, ca in FAISS) - metoda folosita de recomandator.
  Impartim piesele in C clustere cu k-means (algoritmul lui Lloyd). Pentru o piesa x din
  clusterul cu centroidul c:   q . x = q . c + q . (x - c).
  Deci clusterele cu q . c mare sunt cele in care probabil se afla scorurile mari.
  La interogare sondam doar cele n_probe clustere cu q . c maxim si scanam exact piesele lor.

* HyperplaneLSHIndex (Locality Sensitive Hashing cu hiperplane aleatoare) - pentru comparatie.
  Vezi docstring-ul clasei: are o teorie frumoasa, dar pe datele noastre merge prost
  (motivul e explicat in README si se vede in evaluate.py).
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class SearchResult:
    ids: np.ndarray      # piesele gasite, in ordinea descrescatoare a scorului
    scores: np.ndarray   # q . v_j pentru fiecare
    n_scored: int        # cate piese au fost evaluate efectiv (masura costului)


def _top_k(ids: np.ndarray, scores: np.ndarray, k: int):
    k = min(k, len(ids))
    if k <= 0:
        return ids[:0], scores[:0]
    part = np.argpartition(-scores, k - 1)[:k]
    order = part[np.argsort(-scores[part], kind="stable")]
    return ids[order], scores[order]


def _drop_excluded(ids: np.ndarray, exclude, n_items: int) -> np.ndarray:
    """Elimina piesele deja ascultate (cu o masca booleana: mult mai rapid decat np.isin)."""
    if exclude is None or len(exclude) == 0:
        return ids
    mask = np.zeros(n_items, dtype=bool)
    mask[np.asarray(exclude, dtype=np.int64)] = True
    return ids[~mask[ids]]


# --------------------------------------------------------------------------- exact
class BruteForceIndex:
    """Cautare exacta: calculeaza q . v_j pentru TOATE piesele. Cost O(n * d)."""

    def __init__(self, vectors: np.ndarray):
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float64)

    def search(self, q: np.ndarray, k: int, exclude=None) -> SearchResult:
        scores = self.vectors @ q
        ids = _drop_excluded(np.arange(len(self.vectors)), exclude, len(self.vectors))
        top_ids, top_scores = _top_k(ids, scores[ids], k)
        return SearchResult(top_ids, top_scores, n_scored=len(self.vectors))


# --------------------------------------------------------------------------- IVF
def _kmeans_plus_plus(X: np.ndarray, n_clusters: int, rng) -> np.ndarray:
    """
    Initializarea k-means++ (Arthur & Vassilvitskii, 2007): fiecare centroid nou este ales
    cu probabilitate proportionala cu D(x)^2 = distanta^2 pana la cel mai apropiat centroid
    deja ales. Punctele departate sunt favorizate, deci centroizii se "imprastie"; costul
    asteptat este cel mult O(log k) ori optimul.
    """
    n = len(X)
    centroids = np.empty((n_clusters, X.shape[1]))
    centroids[0] = X[rng.integers(n)]
    d2 = ((X - centroids[0]) ** 2).sum(axis=1)
    for c in range(1, n_clusters):
        total = d2.sum()
        i = rng.choice(n, p=d2 / total) if total > 0 else rng.integers(n)
        centroids[c] = X[i]
        d2 = np.minimum(d2, ((X - X[i]) ** 2).sum(axis=1))
    return centroids


def kmeans(X: np.ndarray, n_clusters: int, n_iter: int = 25, seed: int = 0):
    """
    Algoritmul lui Lloyd: minimizeaza suma ||x_i - c_{a(i)}||^2 alternand
      (1) atribuirea fiecarui punct la cel mai apropiat centroid si
      (2) mutarea fiecarui centroid in media punctelor lui.
    Distanta se calculeaza vectorizat: ||x - c||^2 = ||x||^2 - 2 x.c + ||c||^2
    (||x||^2 nu depinde de c, deci nu conteaza la argmin).
    """
    X = np.asarray(X, dtype=np.float64)
    n = len(X)
    if not 1 <= n_clusters <= n:
        raise ValueError(f"n_clusters={n_clusters} invalid pentru {n} puncte")
    rng = np.random.default_rng(seed)

    def nearest(centroids):
        return np.argmin((centroids ** 2).sum(axis=1) - 2.0 * (X @ centroids.T), axis=1)

    centroids = _kmeans_plus_plus(X, n_clusters, rng)
    assign = nearest(centroids)
    for _ in range(n_iter):
        counts = np.bincount(assign, minlength=n_clusters)
        sums = np.zeros_like(centroids)
        np.add.at(sums, assign, X)
        centroids = sums / np.maximum(counts, 1)[:, None]
        empty = counts == 0                    # cluster gol: il reinitializam pe un punct aleator
        if empty.any():
            centroids[empty] = X[rng.choice(n, empty.sum(), replace=False)]
        new_assign = nearest(centroids)
        if np.array_equal(new_assign, assign):
            break                              # convergenta: atribuirile nu se mai schimba
        assign = new_assign
    return centroids, assign


class IVFIndex:
    """ANN cu inverted file: k-means pe vectori + scanarea celor n_probe clustere cu q . c maxim."""

    def __init__(self, vectors: np.ndarray, n_clusters=None, n_probe: int = 32,
                 n_iter: int = 25, seed: int = 0):
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float64)
        n = len(self.vectors)
        # Implicit C = 2 sqrt(n): cel mai bun compromis recall / cost masurat in evaluate.py.
        self.n_clusters = n_clusters or max(1, min(n, int(round(2 * np.sqrt(n)))))
        self.n_probe = min(n_probe, self.n_clusters)
        self.centroids, assign = kmeans(self.vectors, self.n_clusters, n_iter, seed)
        order = np.argsort(assign, kind="stable")
        bounds = np.searchsorted(assign[order], np.arange(self.n_clusters + 1))
        self.lists = [order[bounds[c]:bounds[c + 1]] for c in range(self.n_clusters)]

    def candidates(self, q: np.ndarray, exclude=None, min_count: int = 0) -> np.ndarray:
        """
        Piesele din cele n_probe clustere cu q . c maxim. Daca dupa eliminarea pieselor
        excluse raman mai putin de `min_count`, sondam in continuare clusterele urmatoare.
        (Necesar la Discover: q "arata" exact spre clusterele cu piese deja ascultate.)
        """
        mask = None
        if exclude is not None and len(exclude) > 0:
            mask = np.zeros(len(self.vectors), dtype=bool)
            mask[np.asarray(exclude, dtype=np.int64)] = True
        keep = (lambda ids: ids) if mask is None else (lambda ids: ids[~mask[ids]])

        order = np.argsort(-(self.centroids @ q), kind="stable")
        chunks = [keep(np.concatenate([self.lists[c] for c in order[:self.n_probe]]))]
        count, i = len(chunks[0]), self.n_probe
        while count < min_count and i < self.n_clusters:     # sondam in continuare, cate un cluster
            chunks.append(keep(self.lists[order[i]]))
            count += len(chunks[-1])
            i += 1
        return np.concatenate(chunks)

    def search(self, q: np.ndarray, k: int, exclude=None) -> SearchResult:
        cand = self.candidates(q, exclude, min_count=k)
        top_ids, top_scores = _top_k(cand, self.vectors[cand] @ q, k)
        return SearchResult(top_ids, top_scores, n_scored=len(cand))


# --------------------------------------------------------------------------- LSH
def mips_augment(vectors: np.ndarray) -> np.ndarray:
    """
    Reducerea MIPS -> cosinus (Bachrach et al., 2014). Cu M = max_j ||v_j||:
        x_j = [v_j, sqrt(M^2 - ||v_j||^2)]   (toti x_j au norma exact M)
        y   = [q, 0]
    => x_j . y = v_j . q  si  cos(x_j, y) = (v_j . q) / (M ||q||),
    deci ordinea dupa unghi este exact ordinea dupa produs scalar.
    """
    V = np.asarray(vectors, dtype=np.float64)
    norms_sq = np.einsum("ij,ij->i", V, V)
    extra = np.sqrt(np.maximum(norms_sq.max() - norms_sq, 0.0))
    return np.hstack([V, extra[:, None]])


class HyperplaneLSHIndex:
    """
    LSH cu hiperplane aleatoare (Charikar, 2002), aplicat dupa reducerea MIPS -> cosinus.

    Alegem r ~ N(0, I) si h(x) = semn(r . x). Pentru doi vectori cu unghiul theta:
        P[h(x) = h(y)] = 1 - theta / pi
    (hiperplanul aleator ii separa exact cand cade "intre" ei: probabilitate theta / pi).
    b biti concatenati = o "galeata" per tabel:  P[aceeasi galeata] = (1 - theta/pi)^b.
    Cu L tabele independente:  P[se intalnesc macar o data] = 1 - (1 - (1 - theta/pi)^b)^L.
    Multi-probe: verificam si galetile obtinute inversand bitii cei mai "nesiguri"
    (hiperplanele de care q este cel mai aproape, adica |r . q| mic).
    """

    def __init__(self, vectors: np.ndarray, n_tables: int = 16, n_bits: int = 10,
                 n_probes: int = 3, seed: int = 0):
        if not 1 <= n_bits <= 62:
            raise ValueError("n_bits trebuie sa fie intre 1 si 62")
        if not 0 <= n_probes <= n_bits:
            raise ValueError("n_probes trebuie sa fie intre 0 si n_bits")
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float64)
        self.n_tables, self.n_bits, self.n_probes = n_tables, n_bits, n_probes

        augmented = mips_augment(self.vectors)
        rng = np.random.default_rng(seed)
        self.planes = rng.standard_normal((n_tables, n_bits, augmented.shape[1]))
        self._powers = np.left_shift(np.int64(1), np.arange(n_bits, dtype=np.int64))

        self.tables = []
        for t in range(n_tables):
            codes = ((augmented @ self.planes[t].T) > 0).astype(np.int64) @ self._powers
            order = np.argsort(codes, kind="stable")
            uniq, starts = np.unique(codes[order], return_index=True)
            ends = np.append(starts[1:], len(order))
            self.tables.append({int(c): order[s:e] for c, s, e in zip(uniq, starts, ends)})

    def candidates(self, q: np.ndarray) -> np.ndarray:
        proj = self.planes @ np.append(q, 0.0)               # interogarea transformata [q, 0]
        base = (proj > 0).astype(np.int64) @ self._powers    # (n_tables,)
        unsure = np.argsort(np.abs(proj), axis=1)[:, :self.n_probes]
        codes = np.concatenate([base[:, None], base[:, None] ^ self._powers[unsure]], axis=1)
        found = [self.tables[t].get(int(c)) for t in range(self.n_tables) for c in codes[t]]
        found = [f for f in found if f is not None]
        if not found:
            return np.empty(0, dtype=np.int64)
        return np.unique(np.concatenate(found))

    def search(self, q: np.ndarray, k: int, exclude=None) -> SearchResult:
        cand = _drop_excluded(self.candidates(q), exclude, len(self.vectors))
        # Reordonare exacta: x_j . y = v_j . q, deci folosim direct vectorii originali.
        top_ids, top_scores = _top_k(cand, self.vectors[cand] @ q, k)
        return SearchResult(top_ids, top_scores, n_scored=len(cand))

"""Teste: matematica (SVD, MIPS, LSH, k-means), cautarea si regulile playlistului.

    python -m unittest -v
"""
import unittest

import numpy as np

from discover_weekly import (BruteForceIndex, DiscoverWeekly, HyperplaneLSHIndex, IVFIndex,
                             generate, kmeans, log_weights, mips_augment, randomized_svd,
                             split_holdout)

SMALL = dict(n_users=600, n_genres=5, artists_per_genre=8, tracks_per_artist=15,
             mean_listens=80, seed=1)


class TestSVD(unittest.TestCase):
    def test_exact_on_low_rank_matrix(self):
        rng = np.random.default_rng(0)
        A = rng.standard_normal((120, 8)) @ rng.standard_normal((8, 90))   # rang exact 8
        U, S, Vt = randomized_svd(A, 8)
        np.testing.assert_allclose(S, np.linalg.svd(A, compute_uv=False)[:8], rtol=1e-9)
        np.testing.assert_allclose((U * S) @ Vt, A, atol=1e-8)
        np.testing.assert_allclose(U.T @ U, np.eye(8), atol=1e-10)
        np.testing.assert_allclose(Vt @ Vt.T, np.eye(8), atol=1e-10)

    def test_sparse_matches_numpy(self):
        W = log_weights(generate(**SMALL).plays)
        _, S, _ = randomized_svd(W, 10)
        np.testing.assert_allclose(S, np.linalg.svd(W.toarray(), compute_uv=False)[:10], rtol=1e-3)

    def test_scores_are_rank_k_reconstruction(self):
        # Identitatea matematica: W V_k V_k^T = W_k (trunchierea Eckart-Young), adica
        # scorul q . v_j este intrarea (u, j) din aproximarea de rang k.
        ds = generate(**SMALL)
        k = 5                                       # 5 genuri => salt mare dupa a 5-a valoare singulara
        W = log_weights(ds.plays).toarray()
        U, S, Vt = np.linalg.svd(W, full_matrices=False)
        Wk = (U[:, :k] * S[:k]) @ Vt[:k]
        np.testing.assert_allclose(W @ Vt[:k].T @ Vt[:k], Wk, atol=1e-10)

        # Modelul (cu SVD randomizat) trebuie sa dea aceleasi scoruri, pana la eroarea algoritmului.
        model = DiscoverWeekly(n_factors=k, index="exact").fit(ds.plays)
        for u in [0, 17, 333]:
            scores = model.track_vectors @ model.taste_vector(u)
            np.testing.assert_allclose(scores, Wk[u], atol=5e-3 * np.abs(Wk).max())

    def test_power_iterations_improve_accuracy(self):
        W = log_weights(generate(**SMALL).plays)
        Wd = W.toarray()
        U, S, Vt = np.linalg.svd(Wd, full_matrices=False)
        Wk = (U[:, :5] * S[:5]) @ Vt[:5]
        errors = []
        for it in [0, 2, 8]:
            _, _, Vr = randomized_svd(W, 5, n_power_iter=it)
            errors.append(np.abs(Wd @ Vr.T @ Vr - Wk).max())
        self.assertTrue(errors[0] > errors[1] > errors[2])
        self.assertLess(errors[2], 1e-4 * np.abs(Wk).max())


class TestVectorMath(unittest.TestCase):
    def test_mips_augment_preserves_inner_product_order(self):
        rng = np.random.default_rng(0)
        V = rng.standard_normal((500, 16)) * rng.exponential(size=(500, 1))   # norme foarte diferite
        X = mips_augment(V)
        norms = np.linalg.norm(X, axis=1)
        np.testing.assert_allclose(norms, norms.max())                          # toate au norma M
        for _ in range(20):
            q = rng.standard_normal(16)
            y = np.append(q, 0.0)
            np.testing.assert_allclose(X @ y, V @ q, atol=1e-12)
            cos = (X @ y) / (norms * np.linalg.norm(y))
            np.testing.assert_array_equal(np.argsort(-cos)[:10], np.argsort(-(V @ q))[:10])

    def test_hyperplane_collision_probability(self):
        # P[semn(r.x) = semn(r.y)] = 1 - theta/pi, cu r ~ N(0, I)
        rng = np.random.default_rng(0)
        R = rng.standard_normal((200_000, 10))
        x = np.eye(10)[0]
        for theta in [np.pi / 6, np.pi / 3, np.pi / 2, 2 * np.pi / 3]:
            y = np.cos(theta) * np.eye(10)[0] + np.sin(theta) * np.eye(10)[1]
            same = np.mean(np.sign(R @ x) == np.sign(R @ y))
            self.assertAlmostEqual(same, 1 - theta / np.pi, delta=0.005)

    def test_kmeans_recovers_separated_blobs(self):
        rng = np.random.default_rng(0)
        centers = 10.0 * np.eye(4)
        X = np.vstack([c + 0.1 * rng.standard_normal((50, 4)) for c in centers])
        truth = np.repeat(np.arange(4), 50)
        for seed in range(5):
            _, assign = kmeans(X, 4, seed=seed)
            for b in range(4):
                self.assertEqual(len(set(assign[truth == b])), 1)   # un blob = un singur cluster
            self.assertEqual(len(set(assign)), 4)                    # clustere diferite


class TestIndexes(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.V = rng.standard_normal((400, 8)) * rng.exponential(size=(400, 1))
        self.queries = rng.standard_normal((15, 8))
        self.exact = BruteForceIndex(self.V)

    def test_brute_force(self):
        q = self.queries[0]
        order = list(np.argsort(-(self.V @ q)))
        exclude = order[:3]
        res = self.exact.search(q, 10, exclude=exclude)
        self.assertEqual(list(res.ids), order[3:13])
        np.testing.assert_allclose(res.scores, self.V[res.ids] @ q)
        self.assertEqual(len(self.exact.search(q, 1000, exclude=exclude).ids), 397)

    def test_full_probing_equals_exact(self):
        full = [IVFIndex(self.V, n_clusters=20, n_probe=20),
                HyperplaneLSHIndex(self.V, n_tables=1, n_bits=1, n_probes=1)]  # ambele galeti
        for index in full:
            for q in self.queries:
                exclude = np.argsort(-(self.V @ q))[:5]
                a, b = index.search(q, 25, exclude), self.exact.search(q, 25, exclude)
                np.testing.assert_array_equal(a.ids, b.ids)

    def test_ivf_probes_more_clusters_when_needed(self):
        index = IVFIndex(self.V, n_clusters=20, n_probe=1)
        q = self.queries[0]
        first = index.lists[int(np.argmax(index.centroids @ q))]
        self.assertEqual(len(index.candidates(q)), len(first))          # n_probe=1: un cluster
        res = index.search(q, 30, exclude=first)                           # excludem tot clusterul
        self.assertEqual(len(res.ids), 30)                                 # => sondeaza mai departe
        self.assertFalse(set(res.ids) & set(first))

    def test_ann_results_are_valid(self):
        for index in [IVFIndex(self.V, n_probe=4), HyperplaneLSHIndex(self.V, n_bits=6)]:
            for q in self.queries:
                exclude = np.argsort(-(self.V @ q))[:5]
                res = index.search(q, 30, exclude)
                self.assertFalse(set(res.ids) & set(exclude))
                self.assertEqual(len(set(res.ids)), len(res.ids))
                np.testing.assert_allclose(res.scores, self.V[res.ids] @ q)
                self.assertTrue(np.all(np.diff(res.scores) <= 0))
                self.assertGreaterEqual(res.n_scored, len(res.ids))


class TestRecommender(unittest.TestCase):
    """Pe configuratia implicita (cea din demo.py): 2.000 utilizatori x 5.000 piese."""

    @classmethod
    def setUpClass(cls):
        cls.ds = generate(seed=1)
        cls.model = DiscoverWeekly(index="ivf").fit(cls.ds.plays, cls.ds.catalog.track_artist)
        cls.users = list(range(0, cls.ds.plays.shape[0], 13))

    def test_playlist_rules(self):
        fallbacks = 0
        for u in self.users:
            pl = self.model.recommend(u, n=30, max_per_artist=2)
            self.assertEqual(len(pl.track_ids), 30)
            self.assertEqual(len(set(pl.track_ids)), 30)
            self.assertFalse(set(pl.track_ids) & set(self.ds.plays[u].indices))   # doar piese noi
            per_artist = np.bincount(self.ds.catalog.track_artist[pl.track_ids])
            self.assertLessEqual(per_artist.max(), 2)
            self.assertTrue(np.all(np.diff(pl.scores) <= 0))
            self.assertLess(pl.n_scored, 0.2 * self.ds.plays.shape[1])            # ANN chiar economiseste
            fallbacks += pl.used_fallback
        self.assertLessEqual(fallbacks, 0.05 * len(self.users))

    def test_matches_hidden_taste(self):
        liked = self.ds.user_genre_pref >= 0.10
        genre = self.ds.catalog.track_genre
        match = np.mean([liked[u, genre[self.model.recommend(u).track_ids]].mean() for u in self.users])
        chance = liked[self.users].mean()        # fractia asteptata la recomandari aleatoare
        self.assertLess(chance, 0.4)
        self.assertGreater(match, 0.8)

    def test_holdout_beats_popularity(self):
        train, test = split_holdout(self.ds.plays, fraction=0.2, seed=0)
        model = DiscoverWeekly(index="ivf").fit(train, self.ds.catalog.track_artist)
        pop = np.asarray(train.sum(axis=0)).ravel().astype(float)
        ours, base = [], []
        for u in self.users:
            ours.append(len(np.intersect1d(model.recommend(u).track_ids, test[u])) / len(test[u]))
            p = pop.copy()
            p[train[u].indices] = -np.inf
            base.append(len(np.intersect1d(np.argsort(-p)[:30], test[u])) / len(test[u]))
        self.assertGreater(np.mean(ours), 2 * np.mean(base))


class TestData(unittest.TestCase):
    def test_generate_and_split(self):
        ds = generate(**SMALL)
        n_tracks = SMALL["n_genres"] * SMALL["artists_per_genre"] * SMALL["tracks_per_artist"]
        self.assertEqual(ds.plays.shape, (SMALL["n_users"], n_tracks))
        self.assertTrue(np.all(ds.plays.data >= 1) and np.all(ds.plays.data == np.round(ds.plays.data)))
        np.testing.assert_allclose(ds.user_genre_pref.sum(axis=1), 1.0)

        train, test = split_holdout(ds.plays, fraction=0.2, seed=0)
        for u in [0, 5, 99]:
            full, tr = set(ds.plays[u].indices), set(train[u].indices)
            self.assertEqual(tr | set(test[u]), full)
            self.assertFalse(tr & set(test[u]))
            self.assertEqual(len(test[u]), int(0.2 * len(full)))


if __name__ == "__main__":
    unittest.main()

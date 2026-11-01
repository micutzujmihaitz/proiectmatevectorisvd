"""
Generator de date sintetice: un catalog muzical si istoricul de ascultari.

Structura ascunsa (pe care algoritmul NU o vede, o folosim doar la evaluare):
  gen -> artisti -> piese
Fiecare utilizator are cateva genuri preferate (distributie Dirichlet) si
artisti preferati in cadrul lor. Din asta se genereaza ascultarile.
"""
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

GENRES = ["rock", "jazz", "hip-hop", "electronic", "classical",
          "pop", "metal", "folk", "reggae", "blues"]

_WORDS_A = ["Midnight", "Electric", "Silent", "Golden", "Broken", "Neon", "Lost",
            "Wild", "Velvet", "Crystal", "Hollow", "Burning", "Paper", "Frozen",
            "Distant", "Little", "Endless", "Secret", "Crimson", "Lonely"]
_WORDS_B = ["River", "Dreams", "Highway", "Heart", "Echoes", "City", "Rain",
            "Fire", "Shadows", "Ocean", "Sky", "Garden", "Signal", "Moon",
            "Letters", "Mirror", "Train", "Summer", "Waves", "Static"]


@dataclass
class Catalog:
    genre_names: list
    artist_names: list
    artist_genre: np.ndarray   # (n_artists,) genul fiecarui artist
    track_names: list
    track_artist: np.ndarray   # (n_tracks,) artistul fiecarei piese

    @property
    def track_genre(self) -> np.ndarray:
        return self.artist_genre[self.track_artist]

    @property
    def n_tracks(self) -> int:
        return len(self.track_names)

    def describe(self, track_id: int) -> str:
        a = self.track_artist[track_id]
        g = self.artist_genre[a]
        return f"{self.track_names[track_id]} - {self.artist_names[a]} [{self.genre_names[g]}]"


@dataclass
class Dataset:
    catalog: Catalog
    plays: sp.csr_matrix          # (n_users, n_tracks): de cate ori a ascultat u piesa t
    user_genre_pref: np.ndarray   # (n_users, n_genres) ADEVARUL ASCUNS, doar pentru evaluare


def _zipf_weights(n: int, exponent: float, rng) -> np.ndarray:
    """Ponderi de popularitate de tip Zipf (cateva elemente foarte populare), in ordine aleatoare."""
    w = 1.0 / np.arange(1, n + 1) ** exponent
    return rng.permutation(w)


def generate(n_users: int = 2000,
             n_genres: int = 10,
             artists_per_genre: int = 20,
             tracks_per_artist: int = 25,
             mean_listens: int = 200,
             genre_concentration: float = 0.3,
             seed: int = 0) -> Dataset:
    if not 1 <= n_genres <= len(GENRES):
        raise ValueError(f"n_genres trebuie sa fie intre 1 si {len(GENRES)}")
    rng = np.random.default_rng(seed)

    genre_names = GENRES[:n_genres]
    n_artists = n_genres * artists_per_genre
    n_tracks = n_artists * tracks_per_artist

    artist_genre = np.repeat(np.arange(n_genres), artists_per_genre)
    artist_names = [f"{genre_names[artist_genre[a]].title()} Artist {a % artists_per_genre:02d}"
                    for a in range(n_artists)]
    track_artist = np.repeat(np.arange(n_artists), tracks_per_artist)
    track_names = [f"{rng.choice(_WORDS_A)} {rng.choice(_WORDS_B)}" for _ in range(n_tracks)]

    # Popularitate globala: artisti in cadrul genului, piese in cadrul artistului.
    artist_pop = np.concatenate([_zipf_weights(artists_per_genre, 0.8, rng) for _ in range(n_genres)])
    track_pop_in_artist = 1.0 / np.arange(1, tracks_per_artist + 1) ** 1.0
    track_pop_in_artist /= track_pop_in_artist.sum()

    # Gusturi ascunse ale utilizatorilor.
    user_genre_pref = rng.dirichlet(np.full(n_genres, genre_concentration), size=n_users)
    # Zgomot multiplicativ: fiecare om are artistii lui preferati, nu doar pe cei populari.
    user_artist_taste = rng.gamma(shape=0.5, scale=1.0, size=(n_users, n_artists))

    rows, cols = [], []
    for u in range(n_users):
        n_listens = max(10, rng.poisson(mean_listens))
        genres, counts = np.unique(rng.choice(n_genres, size=n_listens, p=user_genre_pref[u]),
                                   return_counts=True)
        for g, c in zip(genres, counts):
            artists_g = np.arange(g * artists_per_genre, (g + 1) * artists_per_genre)
            w = artist_pop[artists_g] * user_artist_taste[u, artists_g]
            artists = rng.choice(artists_g, size=c, p=w / w.sum())
            offsets = rng.choice(tracks_per_artist, size=c, p=track_pop_in_artist)
            rows.append(np.full(c, u))
            cols.append(artists * tracks_per_artist + offsets)

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    # Conversia COO -> CSR aduna duplicatele => numarul de ascultari per (u, t).
    plays = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_users, n_tracks)).tocsr()

    catalog = Catalog(genre_names, artist_names, artist_genre, track_names, track_artist)
    return Dataset(catalog, plays, user_genre_pref)


def split_holdout(plays, fraction: float = 0.2, seed: int = 0):
    """
    Ascunde aleator `fraction` din piesele distincte ale fiecarui utilizator.
    Antrenam pe `train` si verificam daca recomandarile "ghicesc" piesele ascunse (`test`).
    """
    rng = np.random.default_rng(seed)
    plays = sp.csr_matrix(plays, copy=True)
    plays.sum_duplicates()
    keep = np.ones(plays.nnz, dtype=bool)
    test = []
    for u in range(plays.shape[0]):
        start, end = plays.indptr[u], plays.indptr[u + 1]
        n_hide = int(fraction * (end - start))
        pos = start + rng.choice(end - start, n_hide, replace=False)
        keep[pos] = False
        test.append(np.sort(plays.indices[pos]))
    plays.data[~keep] = 0
    plays.eliminate_zeros()
    return plays, test

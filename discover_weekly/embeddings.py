"""
Embedding-uri pentru piese prin SVD trunchiat (descompunere in valori singulare).

Fie W (utilizatori x piese) matricea de ascultari ponderata. SVD:
    W = U diag(S) V^T
Pastrand doar primele k valori singulare obtinem W_k, cea mai buna aproximare
de rang k a lui W (teorema Eckart-Young). Randurile lui V_k (n_piese x k) sunt
embedding-urile pieselor: piese ascultate de aceiasi oameni ajung vectori apropiati.
"""
import numpy as np
import scipy.sparse as sp


def log_weights(plays) -> sp.csr_matrix:
    """w_ut = log(1 + ascultari): 50 de ascultari nu inseamna de 50 de ori mai mult 'imi place'."""
    W = sp.csr_matrix(plays, dtype=np.float64, copy=True)
    W.data = np.log1p(W.data)
    return W


def randomized_svd(A, k: int, n_oversamples: int = 10, n_power_iter: int = 4, seed: int = 0):
    """
    Primele k componente singulare ale lui A (m x n): A ~= U diag(S) Vt.

    Algoritmul Halko, Martinsson & Tropp (2011):
      1. Y = A * Omega, cu Omega o matrice gaussiana aleatoare (n x l), l = k + oversampling.
         Coloanele lui Y acopera (aproape) spatiul generat de primii k vectori singulari stangi.
      2. Iteratii de putere (A A^T)^q Y, cu re-ortogonalizare QR, ca sa amplificam
         valorile singulare mari fata de cele mici.
      3. Q = baza ortonormala a lui Y;  B = Q^T A este mica (l x n).
      4. SVD exact pe B:  B = Ub S Vt   =>   A ~= (Q Ub) S Vt.
    Functioneaza si cu A matrice rara (scipy.sparse): folosim doar produse A*X si A^T*X.
    """
    m, n = A.shape
    l = min(k + n_oversamples, m, n)
    if not 1 <= k <= l:
        raise ValueError(f"k={k} invalid pentru o matrice {m}x{n}")
    rng = np.random.default_rng(seed)

    Q, _ = np.linalg.qr(A @ rng.standard_normal((n, l)))
    for _ in range(n_power_iter):
        Z, _ = np.linalg.qr(A.T @ Q)
        Q, _ = np.linalg.qr(A @ Z)

    B = np.asarray((A.T @ Q).T)          # = Q^T A, calculat fara a transpune matricea rara
    Ub, S, Vt = np.linalg.svd(B, full_matrices=False)
    U = Q @ Ub
    return U[:, :k], S[:k], Vt[:k]

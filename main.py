import numpy as np
from scipy.special import factorial
from scipy.special import sph_harm
from scipy import integrate
import quantum_numbers

NUM_BASIS = 5 #1s, 2s, 2px, 2py, 2pz

#a basis is a list of lists, each list is [i, j, k, alpha] where i, j, k are the quantum numbers and alpha is the exponent
#start with simply 1s, 2s, 2px, 2py, 2pz basis
basis = [[0, 0, 0,2.709498091], [0, 0, 0, 1.012151084], [1, 0, 0, 1.759666885], [0, 1, 0, 1.759666885], [0, 0, 1, 1.759666885]]
normalisation = [get_normalization_constant(i) for i in range(NUM_BASIS)]

def get_normalization_constant(n):
    i, j, k, alpha = basis[n]
    leading_term = (2 * alpha / np.pi) ** (3 / 4) * (4 * alpha) ** ((i + j + k) / 2)
    factorial_term = np.prod(np.arange(2 * i - 1, 0, -2) * np.arange(2 * j - 1, 0, -2) * np.arange(2 * k - 1, 0, -2))
    return leading_term / np.sqrt(factorial_term)

def get_gaussian_orbital(x, y, z, i, j, k, alpha):
    r2 = x * x + y * y + z * z
    return np.exp(-alpha * r2) * x ** i * y ** j * z ** k

def get_basis(x, y, z, n):
    # Returns unnormalized gaussian orbital
    i, j, k, alpha = basis[n]
    return get_gaussian_orbital(x, y, z, i, j, k, alpha)

def multiply_basis(x, y, z, n, m):
    i_n, j_n, k_n, alpha_n = basis[n]
    i_m, j_m, k_m, alpha_m = basis[m]
    return get_gaussian_orbital(x, y, z, i_n + i_m, j_n + j_m, k_n + k_m, alpha_n + alpha_m)

def overlap_integral(n0, n1):
    integrand = lambda x, y, z: multiply_basis(x, y, z, n0, n1)
    normalization = normalisation[n0] * normalisation[n1]
    return normalization * integrate.tplquad(integrand, -np.inf, np.inf, -np.inf, np.inf, -np.inf, np.inf)[0]

def nuclear_attraction_integral(n0, n1, Z):
    integrand = lambda x, y, z: multiply_basis(x, y, z, n0, n1) / np.sqrt(x ** 2 + y ** 2 + z ** 2)
    normalization = normalisation[n0] * normalisation[n1] * Z
    return normalization * integrate.tplquad(integrand, -np.inf, np.inf, -np.inf, np.inf, -np.inf, np.inf)[0]

def kinetic_energy_integral(n0, n1):
    i0, j0, k0, alpha0 = basis[n0]
    i1, j1, k1, alpha1 = basis[n1]
    integrand = lambda x, y, z: multiply_basis(x, y, z, n0, n1) * (alpha0 * alpha1 * (3 - 2 * alpha0 * (x ** 2 + y ** 2 + z ** 2)) - 2 * (i0 + j0 + k0 + 3) * (i1 + j1 + k1 + 3))
    normalization = normalisation[n0] * normalisation[n1]
    return normalization * integrate.tplquad(integrand, -np.inf, np.inf, -np.inf, np.inf, -np.inf, np.inf)[0]


def construct_matrices(basis):
    S = np.zeros((NUM_BASIS, NUM_BASIS))
    T = np.zeros((NUM_BASIS, NUM_BASIS))
    V = np.zeros((NUM_BASIS, NUM_BASIS))
    E = np.zeros((NUM_BASIS, NUM_BASIS, NUM_BASIS, NUM_BASIS))
    for i in range(NUM_BASIS):
        for j in range(NUM_BASIS):
            i0, j0, k0, alpha0 = basis[i]
            i1, j1, k1, alpha1 = basis[j]
            S[i, j] = overlap_integral(i0, j0, k0, alpha0, i1, j1, k1, alpha1)
            T[i, j] = kinetic_energy_integral(i0, j0, k0, alpha0, i1, j1, k1, alpha1)
            V[i, j] = nuclear_attraction_integral(i0, j0, k0, alpha0, i1, j1, k1, alpha1, 1)
            print(f"({i}, {j}): ", S[i, j], T[i, j], V[i, j])
            for k in range(NUM_BASIS):
                for l in range(NUM_BASIS):
                    i2, j2, k2, alpha2 = basis[k]
                    i3, j3, k3, alpha3 = basis[l]
                    E[i, j, k, l] = ERI(i0, j0, k0, alpha0, i1, j1, k1, alpha1, i2, j2, k2, alpha2, i3, j3, k3, alpha3)
                    print(E[i, j, k, l])
    return S, T, V, E


def orthogonalize(S):
    w, v = np.linalg.eig(S)
    w = np.diag(1 / np.sqrt(w))
    return v @ w @ v.T    

def constuct_fock_matrix(T, V, E, P):
    G = np.zeros((NUM_BASIS, NUM_BASIS))
    for i in range(NUM_BASIS):
        for j in range(NUM_BASIS):
            for k in range(NUM_BASIS):
                for l in range(NUM_BASIS):
                    G[i, j] += P[k, l] * (E[i, j, k, l] - 0.5 * E[i, l, k, j])
    return T + V + G


def transform_fock_matrix(F, X):
    return X.T @ F @ X

def get_density_matrix(C):
    P = np.zeros((NUM_BASIS, NUM_BASIS))
    for i in range(NUM_BASIS):
        for j in range(NUM_BASIS):
            for a in range(NUM_BASIS // 2):
                P[i, j] += 2 * C[i, a] * C[j, a]
    return P

def get_initial_density_matrix(X, T, V):
    H = T + V
    Hp = X.T @ H @ X
    w, C = np.linalg.eig(Hp)
    return get_density_matrix(X @ C)

def SCF(basis):
    S, T, V, E = construct_matrices(basis)
    X = orthogonalize(S)
    P = get_initial_density_matrix(X, T, V)
    last_P = P
    while True:
        F = constuct_fock_matrix(T, V, E, P)
        Fp = transform_fock_matrix(F, X)
        w, C = np.linalg.eig(Fp)
        P = get_density_matrix(X @ C)
        if np.linalg.norm(P - last_P) < 1e-6:
            break
        last_P = P
    return w, F, P


def main():
    w, F, P = SCF(basis)
    print(w)
main()
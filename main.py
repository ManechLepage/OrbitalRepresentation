import numpy as np
from scipy.special import factorial
from scipy.special import erf
from scipy import integrate

NUM_BASIS = 5 #1s, 2s, 2px, 2py, 2pz
Ft = [1/(2*i+1) for i in range(0, 11)]
#a basis is a list of lists, each list is [i, j, k, alpha] where i, j, k are the quantum numbers and alpha is the exponent
#start with simply 1s, 2s, 2px, 2py, 2pz basis
basis = [[0, 0, 0,2.709498091], [0, 0, 0, 1.012151084], [1, 0, 0, 1.759666885], [0, 1, 0, 1.759666885], [0, 0, 1, 1.759666885]]

def get_normalization_constant(n):
    i, j, k, alpha = basis[n]
    leading_term = (2 * alpha / np.pi) ** (3 / 4) * (4 * alpha) ** ((i + j + k) / 2)
    factorial_term = np.prod(np.arange(2 * i - 1, 0, -2) * np.arange(2 * j - 1, 0, -2) * np.arange(2 * k - 1, 0, -2))
    return leading_term / np.sqrt(factorial_term)
normalisation = [get_normalization_constant(i) for i in range(NUM_BASIS)]



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

def boys_function(T):
    if T < 1e-8:  # Avoid numerical instability for small T
        return 1 - T / 3
    return 0.5 * np.sqrt(np.pi / T) * erf(np.sqrt(T))

def gaussian_product_center(alpha1, r1, alpha2, r2):
    p = alpha1 + alpha2
    rp = (alpha1 * r1 + alpha2 * r2) / p
    return p, rp

def eri_ssss(alpha1, r1, alpha2, r2, alpha3, r3, alpha4, r4):
    """Electron repulsion integral for s-type Gaussians."""
    # Compute Gaussian product centers
    p, rp = gaussian_product_center(alpha1, r1, alpha2, r2)
    q, rq = gaussian_product_center(alpha3, r3, alpha4, r4)
    
    # Distance between the two Gaussian product centers
    rab2 = np.dot(rp - rq, rp - rq)
    
    # Boys function argument
    T = p * q / (p + q) * rab2
    
    # Pre-factor
    prefactor = 2 * (np.pi**2.5) / ((p * q * np.sqrt(p + q)))
    
    return prefactor * boys_function(T)

def obara_saika(alpha1, l1, r1, alpha2, l2, r2, alpha3, l3, r3, alpha4, l4, r4):
      # Base case: All angular momentum quantum numbers are zero
    if l1 == l2 == l3 == l4 == 0:
        return eri_ssss(alpha1, r1, alpha2, r2, alpha3, r3, alpha4, r4)
    
    # Initialize result
    result = 0.0
    
    # Recursion over l1
    if l1 > 0:
        result += (r1[0] - r2[0]) * obara_saika(alpha1, l1 - 1, r1, alpha2, l2, r2, alpha3, l3, r3, alpha4, l4, r4)
        result += (1 / (2 * (alpha1 + alpha2))) * (
            obara_saika(alpha1, l1 - 1, r1, alpha2, l2, r2, alpha3, l3, r3, alpha4, l4, r4) +
            (obara_saika(alpha1, l1 - 2, r1, alpha2, l2, r2, alpha3, l3, r3, alpha4, l4, r4) if l1 > 1 else 0.0)
        )
    
    # Recursion over l2
    if l2 > 0:
        result += (r2[0] - r1[0]) * obara_saika(alpha1, l1, r1, alpha2, l2 - 1, r2, alpha3, l3, r3, alpha4, l4, r4)
        result += (1 / (2 * (alpha1 + alpha2))) * (
            obara_saika(alpha1, l1, r1, alpha2, l2 - 1, r2, alpha3, l3, r3, alpha4, l4, r4) +
            (obara_saika(alpha1, l1, r1, alpha2, l2 - 2, r2, alpha3, l3, r3, alpha4, l4, r4) if l2 > 1 else 0.0)
        )
    
    # Recursion over l3
    if l3 > 0:
        result += (r3[0] - r4[0]) * obara_saika(alpha1, l1, r1, alpha2, l2, r2, alpha3, l3 - 1, r3, alpha4, l4, r4)
        result += (1 / (2 * (alpha3 + alpha4))) * (
            obara_saika(alpha1, l1, r1, alpha2, l2, r2, alpha3, l3 - 1, r3, alpha4, l4, r4) +
            (obara_saika(alpha1, l1, r1, alpha2, l2, r2, alpha3, l3 - 2, r3, alpha4, l4, r4) if l3 > 1 else 0.0)
        )
    
    # Recursion over l4
    if l4 > 0:
        result += (r4[0] - r3[0]) * obara_saika(alpha1, l1, r1, alpha2, l2, r2, alpha3, l3, r3, alpha4, l4 - 1, r4)
        result += (1 / (2 * (alpha3 + alpha4))) * (
            obara_saika(alpha1, l1, r1, alpha2, l2, r2, alpha3, l3, r3, alpha4, l4 - 1, r4) +
            (obara_saika(alpha1, l1, r1, alpha2, l2, r2, alpha3, l3, r3, alpha4, l4 - 2, r4) if l4 > 1 else 0.0)
        )
    
    return result

    return result

def construct_matrices():
    S = np.zeros((NUM_BASIS, NUM_BASIS))
    T = np.zeros((NUM_BASIS, NUM_BASIS))
    V = np.zeros((NUM_BASIS, NUM_BASIS))
    E = np.zeros((NUM_BASIS, NUM_BASIS, NUM_BASIS, NUM_BASIS))
    for i in range(NUM_BASIS):
        for j in range(NUM_BASIS):
            S[i, j] = overlap_integral(i, j)
            T[i, j] = kinetic_energy_integral(i, j)
            V[i, j] = nuclear_attraction_integral(i, j, 1)
            print(f"({i}, {j}): ", S[i, j], T[i, j], V[i, j])
            for k in range(NUM_BASIS):
                for l in range(NUM_BASIS):
                    alpha_i, alpha_j, alpha_k, alpha_l = basis[i][3], basis[j][3], basis[k][3], basis[l][3]
                    l_i, l_j, l_k, l_l = np.sum(basis[i][:3]), np.sum(basis[j][:3]), np.sum(basis[k][:3]), np.sum(basis[l][:3])
                    E[i, j, k, l] = obara_saika(alpha_i, l_i, np.zeros(3), alpha_j, l_j, np.zeros(3), alpha_k, l_k, np.zeros(3), alpha_l, l_l, np.zeros(3))
                    print(f"({i}, {j}, {k}, {l}): ", E[i, j, k, l])
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

def SCF():
    S, T, V, E = construct_matrices()
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
    E_tot = 0
    for i in range(NUM_BASIS):
        for j in range(NUM_BASIS):
            E_tot += P[j, i] * (T[i, j] + V[i, j] + F[i, j])
    return w, P, E_tot


MIN = -10
MAX = 10
def get_grid():
    x = np.linspace(MIN, MAX, 17)
    y = np.linspace(MIN, MAX, 17)
    z = np.linspace(MIN, MAX, 17)
    return np.meshgrid(x, y, z)

def main():
    w, P, E_tot = SCF()
    # print ionization energies
    print(w)
    # print electron density
    grid = get_grid()
    electron_density = np.zeros((17, 17, 17))
    for x in range(17):
        for y in range(17):
            for z in range(17):
                for i in range(NUM_BASIS):
                    for j in range(NUM_BASIS):
                        electron_density[x, y, z] += P[i, j] * get_basis(grid[0][x, y, z], grid[1][x, y, z], grid[2][x, y, z], i) * get_basis(grid[0][x, y, z], grid[1][x, y, z], grid[2][x, y, z], j)
    file = open("electron_density.txt", "w")
    print(P)
    for i in range(17):
        for j in range(17):
            for k in range(17):
                file.write(f"{grid[0][i, j, k]} {grid[1][i, j, k]} {grid[2][i, j, k]} {electron_density[i, j, k]}\n")
    file.close()
    E_tot /= 2
    print(E_tot)


main()
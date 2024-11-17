import numpy as np
from scipy.special import factorial
from scipy.special import sph_harm
from scipy import integrate

def get_slater_type_orbital(n, l, m, r, theta, phi, zeta):
    return radial_part(r, n, zeta) * sph_harm(m, l, phi, theta)[-1]

def radial_part(r, n, zeta):
    return get_normalization_constant(zeta, n) * (r ** (n - 1)) * np.exp(-zeta * r)

def get_normalization_constant(zeta, n):
    first_part = (2 * zeta) ** n
    second_part = np.sqrt((2 * zeta) / factorial(2 * n))
    return first_part * second_part

def overlap_integral(n0, n1, l0, l1, m0, m1, zeta0, zeta1):
    if l0 != l1 or m0 != m1:
        return 0
    return factorial(n0 + n1) / ((zeta0 + zeta1) ** (n0 + n1 + 1))

def kinetic_integral(n0, n1, l0, l1, m0, m1, zeta0, zeta1):
    if l0 != l1 or m0 != m1:
        return 0
    constant0 = l1 * (l1 + 1) - n1 * (n1 - 1)
    constant1 = 2 * zeta1 * n1
    constant2 = -2 * zeta1 * zeta1
    integrand = lambda r: np.exp(-(zeta0 + zeta1) * r) * (constant0 * r ** (n0 + n1 - 2) + constant1 * r ** (n0 + n1 - 1) + constant2 * r ** (n0 + n1))
    return integrate.quad(integrand, 0, np.inf) / 2

def nuclear_attraction_integral(n0, n1, l0, l1, m0, m1, zeta0, zeta1):
    if l0 != l1 or m0 != m1:
        return 0
    integrand = lambda r: radial_part(r, n0, zeta0) * radial_part(r, n1, zeta1) * r
    return integrate.quad(integrand, 0, np.inf)

def main():
    orbital = input()
    n = int(orbital[0])
    l = ['s', 'p', 'd', 'f'].index(orbital[1])

main()
import numpy as np
from scipy.special import factorial
from scipy.special import sph_harm
from scipy import integrate
import quantum_numbers

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

def basic_integral(r0, n0, n1, m0, m1, l0, l1, zeta0, zeta1):
    if m0 != m1 or l0 != l1:
        return 0
    integrand = lambda r1: np.conjugate(radial_part(r0 + r1, n0, zeta0)) * radial_part(r0 + r1, n1, zeta1) * r1
    return integrate.quad(integrand, -r0, np.inf)

def get_quantum_numbers(value):
    return quantum_numbers.orbital_dict[value]

def main():
    orbital = input()
    n = int(orbital[0])
    l = ['s', 'p', 'd', 'f'].index(orbital[1])
    basic_integral(0, n, n, 0, 0, l, l, 1, 1)

main()
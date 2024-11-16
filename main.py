import numpy as np
from scipy.special import factorial
from scipy.special import sph_harm

def get_slater_type_orbital(n, l, m, r, theta, phi, zeta):
    return radial_part(r, n, zeta) * sph_harm(m, l, phi, theta)

def radial_part(r, n, zeta):
    return get_normalization_constant(zeta, n) * (r ** (n - 1)) * np.exp(-zeta * r)

def get_normalization_constant(zeta, n):
    first_part = (2 * zeta) ** n
    second_part = np.sqrt((2 * zeta) / factorial(2 * n))
    return first_part * second_part

def get_omega(n, l, s, zeta):
    first_part = (-1 / (4 * (zeta ** 2))) ** s
    second_part = factorial(n - s) / (factorial(s) * factorial(n - l - (2 * s)))
    return first_part * second_part


def main():
    orbital = input()
    n = int(orbital[0])
    l = ['s', 'p', 'd', 'f'].index(orbital[1])

main()
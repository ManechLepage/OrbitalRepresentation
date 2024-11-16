import numpy as np
from scipy.special import factorial
from scipy.special import sph_harm

def get_slater_type_orbital(n, l, m, r, theta, phi, zeta):
    return angular_part(l, m, theta, phi, zeta) * radial_part(r, n, zeta)

def angular_part(l, m, theta, phi, zeta):
    exponential_part = np.exp(1j * m * phi)
    return get_normalization_constant(zeta, l) * exponential_part * P(l, m, np.cos(theta))

def radial_part(r, n, zeta):
    return get_normalization_constant(zeta, n) * (r ** (n - 1)) * np.exp(-zeta * r)

def P(l, m, cos_theta):
    m_abs = abs(m)
    sum_bound = ((l - m_abs) // 2) + (1 / 4) * ((-1) ** (l - m_abs) - 1)
    p_sum = 0

    for k in range(int(sum_bound)):
        p_sum += ((-1) ** k) * (factorial(2 * l - 2 * k) /
                                (factorial(k) * factorial(l - k) * factorial(l - m_abs - 2 * k))) * \
                 (cos_theta ** (l - m_abs - 2 * k))

    first_part = (1 / (2 ** l)) * np.sqrt((2 * l + 1) / 2 * factorial(l - m_abs) / factorial(l + m_abs)) * \
                 ((cos_theta ** 2 + 1) ** (m_abs / 2))

    return p_sum * first_part

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
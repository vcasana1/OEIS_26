#!/usr/bin/env python3
"""Reproduce the decimal enclosure in the paper with mpmath 1.4.1."""

import mpmath as mp


def certified_digits():
    interval = mp.iv
    interval.dps = 150
    terms = 180
    total = interval.mpf(0)
    for k in range(1, terms + 1):
        numerator, denominator = mp.bernfrac(2 * k)
        factorial = mp.factorial(2 * k, exact=True)
        zeta = (
            abs(interval.mpf(numerator)) * (2 * interval.pi) ** (2 * k)
            / (2 * denominator * factorial)
        )
        total += (zeta - 1) / (2 * k - 1)

    partial = interval.pi**2 / 2 - interval.ln(2 * interval.pi) - 1 - 2 * total
    tail = 2 * interval.mpf(4) ** (-terms)
    enclosure = partial - interval.mpf([0, tail.b])
    scaled = enclosure * 10**100

    # Read each binary endpoint at a precision higher than the interval's.
    with mp.workdps(155):
        lower = int(mp.floor(mp.mpf(scaled.a)))
        upper = int(mp.floor(mp.mpf(scaled.b)))
    if lower != upper or len(str(lower)) != 100:
        raise ArithmeticError("The interval does not fix the first 100 decimals.")
    return str(lower)


if __name__ == "__main__":
    print(certified_digits())

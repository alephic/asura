import torch
import numpy as np

# Simplified from https://github.com/avtools-io/itu-r-468-weighting

# DB_GAIN_1KHZ was first determined with the multiplication factor of 8.1333
# (+18.20533583440004 dB). From there on the value was modified to find
# a better one, by converging the distance to 0 at 1000 Hz and 12500 Hz, having
# the same value for both: r468(1000) == r468(12500)
R468_DB_GAIN_1KHZ = 18.246265068039158

# Gain factor, "1khz" option
R468_FACTOR_GAIN_1KHZ = 10 ** (R468_DB_GAIN_1KHZ / 20)

def r468(frequency_hz: torch.Tensor) -> torch.Tensor:
    """Takes a frequency value and returns a weighted gain value.

    For weighting, the ITU-R BS.468-4 standard and the
    SMPTE RP 2054:2010 recommended practice are followed.
    """

    assert torch.all(frequency_hz >= 0.0).item()

    f1 = frequency_hz
    f2 = f1 * f1
    f3 = f1 * f2
    f4 = f2 * f2
    f5 = f1 * f4
    f6 = f3 * f3
    h1 = (
        (-4.7373389813783836e-24 * f6)
        + (2.0438283336061252e-15 * f4)
        - (1.363894795463638e-07 * f2)
        + 1
    )
    h2 = (
        (1.3066122574128241e-19 * f5)
        - (2.1181508875186556e-11 * f3)
        + (0.0005559488023498643 * f1)
    )
    r_itu = (0.0001246332637532143 * f1) / torch.sqrt(h1 * h1 + h2 * h2)

    return R468_FACTOR_GAIN_1KHZ * r_itu

def a_weighting(frequency_hz: torch.Tensor) -> torch.Tensor:
    """Takes a frequency value and returns a weighted gain value.
    
    For weighting, the A-curve is used (https://en.wikipedia.org/wiki/A-weighting)
    """
    f1 = frequency_hz
    f2 = f1 * f1
    f4 = f2 * f2

    r_a = (12194**2*f4) / ((f2 + 20.6**2) * (f2 + 12194**2) * torch.sqrt((f2 + 107.7**2)*(f2 + 737.9**2)))

    return r_a
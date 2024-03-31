from typing import SupportsFloat
import numpy as np
import numpy.typing as npt
from imagetools import delinearize
from trigtools import scalar_cos_pi as cos_pi
from trigtools import scalar_sin_pi as sin_pi


def wavelength_to_lms(Lambda: SupportsFloat
                      ) -> tuple[float, float, float]:
    """Get LMS response values for a given wavelength of light.\n
    Based on smooth fits for the CIE 2006/2015 physiological
    2-degree observer as given in Stockman & Rider (2023).

    Keyword arguments:\n
    Lambda -- A wavelength of light, in nanometers.
              Should be between 350 and 850, inclusive.
    """
    wavelength: float = float(Lambda)
    if wavelength < 350:
        raise ValueError
    if wavelength > 850:
        raise ValueError
    theta_p: float = float((0x129f99d39d7f91/(1 << 52))
                           * np.log(wavelength / 360))
    C1: float = cos_pi(theta_p)
    C2: float = cos_pi(2 * theta_p)
    C3: float = cos_pi(3 * theta_p)
    C4: float = cos_pi(4 * theta_p)
    C5: float = cos_pi(5 * theta_p)
    C6: float = cos_pi(6 * theta_p)
    C7: float = cos_pi(7 * theta_p)
    C8: float = cos_pi(8 * theta_p)
    S1: float = sin_pi(theta_p)
    S2: float = sin_pi(2 * theta_p)
    S3: float = sin_pi(3 * theta_p)
    S4: float = sin_pi(4 * theta_p)
    S5: float = sin_pi(5 * theta_p)
    S6: float = sin_pi(6 * theta_p)
    S7: float = sin_pi(7 * theta_p)
    S8: float = sin_pi(8 * theta_p)
    L_raw: float = (-(0x1576c92146a1a5/(1 << 47))
                    - (0x105143bf727137/(1 << 51)) * C1
                    + (0xe552c2bd7f51f/(1 << 46)) * C2
                    + (0x2071c4fc1df33/(1 << 46)) * C3
                    - (0x1572c6bce8533b/(1 << 48)) * C4
                    - (0xd8ef88b977857/(1 << 50)) * C5
                    + (0xd00c62e4d1a65/(1 << 50)) * C6
                    + (0x1961e4f765fd8b/(1 << 54)) * C7
                    - (0x14508b32ce8965/(1 << 56)) * C8
                    + (0x12fe31b152f3c3/(1 << 46)) * S1
                    + (0x1a4b2702a3486f/(1 << 50)) * S2
                    - (0x136200c9539b89/(1 << 47)) * S3
                    - (0x17c24d099e0e73/(1 << 50)) * S4
                    + (0x9969ad42c3c9f/(1 << 48)) * S5
                    + (0xb87bc3c5bd0e1/(1 << 51)) * S6
                    - (0xb61dc93ea2d3/(1 << 48)) * S7
                    - (0x12aed1394317ad/(1 << 56)) * S8)
    M_raw: float = (-(0x69540181e03f7/(1 << 43))
                    - (0x12a9cdc443914f/(1 << 55)) * C1
                    + (0x131789741d084f/(1 << 44)) * C2
                    + (0x1b5abfb9bed30f/(1 << 50)) * C3
                    - (0x1d9f4c1a8ac5c1/(1 << 46)) * C4
                    - (0x1e60fba8826aa9/(1 << 51)) * C5
                    + (0x9fc8661ae70c1/(1 << 47)) * C6
                    + (0x587ddca4b124d/(1 << 51)) * C7
                    - (0xb502795703f2d/(1 << 52)) * C8
                    + (0x182bb62c77574f/(1 << 44)) * S1
                    + (0x14165cb35f3d7d/(1 << 50)) * S2
                    - (0x340d32f01754b/(1 << 42)) * S3
                    - (0x5c338e6d15ad1/(1 << 48)) * S4
                    + (0xdcb8ac9f2fdb9/(1 << 46)) * S5
                    + (0x1e627e0ef99807/(1 << 52)) * S6
                    - (0x516d490e66cb1/(1 << 48)) * S7
                    - (0x122cd39da16617/(1 << 55)) * S8)
    S_raw: float = (+(0x67b1b4d48882f/(1 << 43))
                    - (0x1939eb6390c91/(1 << 46)) * C1
                    - (0x13baa415f45e0b/(1 << 44)) * C2
                    + (0x13a437a3db3bfb/(1 << 48)) * C3
                    + (0x1e76f123c42a67/(1 << 46)) * C4
                    - (0x115ad4f5903a75/(1 << 49)) * C5
                    - (0x528709741d085/(1 << 46)) * C6
                    + (0x862ec28b2a6b1/(1 << 51)) * C7
                    + (0x1946ae3a3a8e71/(1 << 53)) * C8
                    - (0x189b5c5b4aa971/(1 << 44)) * S1
                    + (0x134ab063e07a29/(1 << 48)) * S2
                    + (0xd6389dbec2481/(1 << 44)) * S3
                    - (0xf2e9c66d373b/(1 << 44)) * S4
                    - (0x1c613bd1676641/(1 << 47)) * S5
                    + (0x1d8c436fc158fb/(1 << 51)) * S6
                    + (0x157670196d8f5/(1 << 46)) * S7
                    - (0x9796bfca85cab/(1 << 54)) * S8)
    theta_mac: float = (wavelength - 375)/175
    mac: float = 0
    if 375 <= wavelength <= 550:
        mac = (+(0x1d2591146b686b/(1 << 41))
               + (0x1780dceaa0a89b/(1 << 44)) * cos_pi(theta_mac)
               - (0xb8e6034fdf66b/(1 << 39)) * cos_pi(2 * theta_mac)
               - (0xb3e81bd5024d5/(1 << 42)) * cos_pi(3 * theta_mac)
               + (0xb5088c2d4969d/(1 << 40)) * cos_pi(4 * theta_mac)
               + (0x75b5f21ec1409/(1 << 42)) * cos_pi(5 * theta_mac)
               - (0x19a9b5dab6d4c5/(1 << 43)) * cos_pi(6 * theta_mac)
               - (0x91863cff074f3/(1 << 44)) * cos_pi(7 * theta_mac)
               + (0x1d09321f5355b3/(1 << 46)) * cos_pi(8 * theta_mac)
               + (0x12c2e17099ef4b/(1 << 48)) * cos_pi(9 * theta_mac)
               - (0x57ee917cea8c5/(1 << 48)) * cos_pi(10 * theta_mac)
               - (0x81c58ac1ad0e9/(1 << 52)) * cos_pi(11 * theta_mac)
               - (0xdc162e13127b3/(1 << 39)) * sin_pi(theta_mac)
               - (0x13e1ba37555abd/(1 << 43)) * sin_pi(2 * theta_mac)
               + (0x44e359906ed85/(1 << 38)) * sin_pi(3 * theta_mac)
               + (0x502a840cef2a3/(1 << 41)) * sin_pi(4 * theta_mac)
               - (0x67e08a68eefe1/(1 << 40)) * sin_pi(5 * theta_mac)
               - (0x11fd65c899694d/(1 << 44)) * sin_pi(6 * theta_mac)
               + (0x2ac146975cf2b/(1 << 41)) * sin_pi(7 * theta_mac)
               + (0xedd845eca56f9/(1 << 46)) * sin_pi(8 * theta_mac)
               - (0x1e62c24f7a158d/(1 << 48)) * sin_pi(9 * theta_mac)
               - (0x42779455b9fd3/(1 << 48)) * sin_pi(10 * theta_mac)
               + (0x41881f62b9633/(1 << 51)) * sin_pi(11 * theta_mac))
    theta_lens: float = (wavelength - 360)/300
    lens: float = 0
    if wavelength <= 660:
        lens = (-(0x13cd5cb0d53119/(1 << 44))
                - (0x46f7bcf9d0e15/(1 << 44)) * cos_pi(theta_lens)
                + (0x1dbdf227ff8f41/(1 << 44)) * cos_pi(2 * theta_lens)
                + (0x802f08a321b1/(1 << 40)) * cos_pi(3 * theta_lens)
                - (0xbde478d793ca1/(1 << 44)) * cos_pi(4 * theta_lens)
                - (0x2277baf6f444d/(1 << 43)) * cos_pi(5 * theta_lens)
                + (0x87074cf819753/(1 << 46)) * cos_pi(6 * theta_lens)
                + (0xdc7201bafb25f/(1 << 48)) * cos_pi(7 * theta_lens)
                - (0x14b5b2de323fff/(1 << 52)) * cos_pi(8 * theta_lens)
                - (0x1cebe4b705d2ab/(1 << 54)) * cos_pi(9 * theta_lens)
                + (0x1276ce3de838af/(1 << 43)) * sin_pi(theta_lens)
                + (0x1d9ba68ca82e85/(1 << 46)) * sin_pi(2 * theta_lens)
                - (0xa3b9b911e70a9/(1 << 43)) * sin_pi(3 * theta_lens)
                - (0x698302ca485b1/(1 << 44)) * sin_pi(4 * theta_lens)
                + (0xb53674d0bb017/(1 << 45)) * sin_pi(5 * theta_lens)
                + (0x11cc5733d4cf59/(1 << 47)) * sin_pi(6 * theta_lens)
                - (0x8d6570e7b1e17/(1 << 48)) * sin_pi(7 * theta_lens)
                - (0xe2e007d5de711/(1 << 50)) * sin_pi(8 * theta_lens)
                + (0x1621416e66f9e7/(1 << 57)) * sin_pi(9 * theta_lens))
    scale: float = 10**(lens + mac)
    try:
        L: float = wavelength * ((1 - 10**(-((10**L_raw)/2)))/scale)
    except BaseException as e:
        print(Lambda, scale, lens, mac, lens+mac)
        raise e
    M: float = wavelength * ((1 - 10**(-((10**M_raw)/2)))/scale)
    S: float = wavelength * ((1 - 10**(-((2 * 10**S_raw)/5)))/scale)
    L_max: float = 0x143cf025b3ed8d/(1 << 44)
    M_max: float = 0x8ee24fceed993/(1 << 43)
    S_max: float = 0xdbaaa42536559/(1 << 46)
    return ((L/L_max, M/M_max, S/S_max))


def lms_to_linear_rgb(l_val: SupportsFloat,
                      m_val: SupportsFloat,
                      s_val: SupportsFloat
                      ) -> tuple[float, float, float]:
    """Convert LMS response values to linearized (s)RGB.

    Keyword arguments:\n
    l_val -- the L cone response\n
    m_val -- the M cone response\n
    s_val -- the S cone response
    """
    L: float = float(l_val)
    M: float = float(m_val)
    S: float = float(s_val)
    R: float = (+(0x5400813f811fd/(1 << 48)) * L
                - (0x147a026cff16b/(1 << 46)) * M
                + (0x37a02de3a8e3b/(1 << 52)) * S)
    G: float = (-(0x12f7e931e0fcfd/(1 << 53)) * L
                + (0x1030e3e0102a41/(1 << 51)) * M
                - (0x117a78798d21b/(1 << 50)) * S)
    B: float = (-(0x1085575fb20ab7/(1 << 57)) * L
                - (0x265d80d4d5c1d/(1 << 52)) * M
                + (0x10860c57d4a42f/(1 << 51)) * S)
    return ((R, G, B))


def lms_to_linear_rgb_optimal(l_val: SupportsFloat,
                              m_val: SupportsFloat,
                              s_val: SupportsFloat,
                              max_scale: SupportsFloat = 4
                              ) -> tuple[float, float, float]:
    """Convert LMS response values to linearized (s)RGB with
    response values scaled such that the resulting colour is
    as close as possible to lying entirely within the (s)RGB
    gamut, ignoring negative values.

    Keyword arguments:\n
    l_val -- the L cone response\n
    m_val -- the M cone response\n
    s_val -- the S cone response\n
    max_scale -- the maximum scaling on the response values
    """
    MAX: float = float(max_scale)
    assert (MAX >= 0)
    L: float = float(l_val)
    M: float = float(m_val)
    S: float = float(s_val)
    R1: float = (+(0x5400813f811fd/(1 << 48)) * L
                 - (0x147a026cff16b/(1 << 46)) * M
                 + (0x37a02de3a8e3b/(1 << 52)) * S)
    G1: float = (-(0x12f7e931e0fcfd/(1 << 53)) * L
                 + (0x1030e3e0102a41/(1 << 51)) * M
                 - (0x117a78798d21b/(1 << 50)) * S)
    B1: float = (-(0x1085575fb20ab7/(1 << 57)) * L
                 - (0x265d80d4d5c1d/(1 << 52)) * M
                 + (0x10860c57d4a42f/(1 << 51)) * S)
    SCALE: float = min(min([1/V for V in [R1, G1, B1] if V > 0]), MAX)
    R: float = R1 * SCALE
    G: float = G1 * SCALE
    B: float = B1 * SCALE
    return ((R, G, B))


def wavelength_to_srgb(Lambda: SupportsFloat
                       ) -> tuple[int, int, int]:
    """Get the (s)RGB colour as close as possible to a given wavelength
    of light.\n
    Based on smooth fits for the CIE 2006/2015 physiological
    2-degree observer as given in Stockman & Rider (2023).

    Keyword arguments:\n
    Lambda -- A wavelength of light, in nanometers.
              Should be between 350 and 850, inclusive.\n
              The smallest integer value for which the output is
                not uniformly zero is 361, giving (1, 0, 0).
              The largest integer value for which the output is
                not uniformly zero is 771, giving (1, 0, 0).
    """
    LMS: tuple[float, float, float] = wavelength_to_lms(Lambda)
    linRGB: tuple[float, float, float] = lms_to_linear_rgb(*LMS)
    linRGB_img: npt.NDArray[np.float64] = np.array([[linRGB]],
                                                   dtype=np.float64)
    sRGB_img: npt.NDArray[np.uint8] = delinearize(linRGB_img)
    sRGB: npt.NDArray[np.uint8] = sRGB_img[0, 0, :]
    R: int = int(sRGB[0])
    G: int = int(sRGB[1])
    B: int = int(sRGB[2])
    return ((R, G, B))


def wavelength_to_srgb_optimal(Lambda: SupportsFloat,
                               max_scale: SupportsFloat = 4
                               ) -> tuple[int, int, int]:
    """Get the (s)RGB colour as close as possible to a given wavelength
    of light with response values scaled such that the resulting colour
    is as close as possible to lying entirely within the (s)RGB gamut,
    ignoring negative values.\n
    Based on smooth fits for the CIE 2006/2015 physiological
    2-degree observer as given in Stockman & Rider (2023).

    Keyword arguments:\n
    Lambda -- A wavelength of light, in nanometers.
              Should be between 350 and 850, inclusive.
    max_scale -- the maximum scaling on the LMS response values
    """
    LMS: tuple[float, float, float] = wavelength_to_lms(Lambda)
    MAX: float = float(max_scale)
    linRGB: tuple[float, float, float] = lms_to_linear_rgb_optimal(*LMS, MAX)
    linRGB_img: npt.NDArray[np.float64] = np.array([[linRGB]],
                                                   dtype=np.float64)
    sRGB_img: npt.NDArray[np.uint8] = delinearize(linRGB_img)
    sRGB: npt.NDArray[np.uint8] = sRGB_img[0, 0, :]
    R: int = int(sRGB[0])
    G: int = int(sRGB[1])
    B: int = int(sRGB[2])
    return ((R, G, B))

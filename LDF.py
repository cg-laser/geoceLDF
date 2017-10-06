import numpy as np
import scipy.special
import pickle
from scipy import interpolate as intp
from scipy.interpolate import BSpline
from radiotools.atmosphere import models as atm
from radiotools import HelperFunctions as hp
import os
atmc = atm.Atmosphere()

dir_path = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(dir_path, "pickle/geo_rcut_b_splines.pickle"), "r") as fin:
    spl_rcut_geo_params, spl_b_geo_params = pickle.load(fin)
    t, c, k = spl_rcut_geo_params
    t = np.append(np.append(np.ones(k) * t[0], t), np.ones(k) * t[-1])
    spl_rcut_geo = BSpline(t, c, k)

    t, c, k = spl_b_geo_params
    t = np.append(np.append(np.ones(k) * t[0], t), np.ones(k) * t[-1])
    spl_b_geo = BSpline(t, c, k)

with open(os.path.join(dir_path, "pickle/geo_sigmaR_spl.pickle"), "r") as fin:
    data = pickle.load(fin)
    t, c, k = data['geo_R_0m']
    t = np.append(np.append(np.ones(k) * t[0], t), np.ones(k) * t[-1])
    spl_geo_R_0m = BSpline(t, c, k)

    t, c, k = data['geo_R_1564m']
    t = np.append(np.append(np.ones(k) * t[0], t), np.ones(k) * t[-1])
    spl_geo_R_1564m = BSpline(t, c, k)

    t, c, k = data['geo_sigma_0m']
    t = np.append(np.append(np.ones(k) * t[0], t), np.ones(k) * t[-1])
    spl_geo_sigma_0m = BSpline(t, c, k)

    t, c, k = data['geo_sigma_1564m']
    t = np.append(np.append(np.ones(k) * t[0], t), np.ones(k) * t[-1])
    spl_geo_sigma_1564m = BSpline(t, c, k)


with open(os.path.join(dir_path, "pickle/ce_sigma_spl.pickle"), "r") as fin:
    data = pickle.load(fin)
    t, c, k = data['ce_sigma_0m']
    t = np.append(np.append(np.ones(k) * t[0], t), np.ones(k) * t[-1])
    spl_ce_sigma_0m = BSpline(t, c, k)

    t, c, k = data['ce_sigma_1564m']
    t = np.append(np.append(np.ones(k) * t[0], t), np.ones(k) * t[-1])
    spl_ce_sigma_1564m = BSpline(t, c, k)


# read in spline parametrization of energy correction factors, i.e. the conversion from the
# fit parameter E to the true radiation energy
with open(os.path.join(dir_path, "pickle/Ecorr.pickle"), "r") as fin:
    data = pickle.load(fin)

    t, c, k = data['geo_Ecorr_1564m']
    t = np.append(np.append(np.ones(k) * t[0], t), np.ones(k) * t[-1])
    spl_geo_Ecorr_1564m = BSpline(t, c, k)

    t, c, k = data['geo_Ecorr_0m']
    t = np.append(np.append(np.ones(k) * t[0], t), np.ones(k) * t[-1])
    spl_geo_Ecorr_0m = BSpline(t, c, k)

    t, c, k = data['ce_Ecorr_1564m']
    t = np.append(np.append(np.ones(k) * t[0], t), np.ones(k) * t[-1])
    spl_ce_Ecorr_1564m = BSpline(t, c, k)

    t, c, k = data['ce_Ecorr_0m']
    t = np.append(np.append(np.ones(k) * t[0], t), np.ones(k) * t[-1])
    spl_ce_Ecorr_0m = BSpline(t, c, k)


def get_p_fixed(rr, dxmax):

    def get_rcut(dxmax, a1=1.98971804e+02, a2=6.75050916e+02, a3=6.35578988e-04, a4=-6.67161546e-07, a5=2.54360599e-10):
        return a1 + a3 * (dxmax - a2) ** 2 + a4 * (dxmax - a2) ** 3 + a5 * (dxmax - a2) ** 4

    def get_b_geo(dxmax, D=5.81868554e+02, a1=6.50455412e-01, b1=-1.76459980e+02, a2=-1.24800432e-01):
        res = np.zeros_like(dxmax)
        res[dxmax < D] = a1 * dxmax[dxmax < D] + b1
        res[dxmax >= D] = a2 * (dxmax[dxmax >= D] - D) + a1 * D + b1
        res[res < 0] = 0
        return res

    return get_p2(rr, get_rcut(dxmax), get_b_geo(dxmax))


# def get_p2(r, rcut, p2):
#     r = np.abs(r)
#     rcut = max(1, np.abs(rcut))
#     b = 1e-3 * p2
#     p_geo = 2. * rcut ** b
#     if np.sum(np.isinf(p_geo)):
#         print rcut, b, p_geo
#     if((type(r) == np.float64) or (type(r) == np.float)):
#         if(r <= 0):
#             return 2.
#         res = p_geo * r ** (-1. * b)
#         if res > 2:
#             return 2.
#         else:
#             return res
#     else:
#         res = np.ones_like(r) * 2.
#         res[r != 0] = p_geo * r[r != 0] ** (-1. * b)
#         res[res > 2] = 2.
#         return res

def get_p2(r, rcut, p2):  # updated version that allows also p > 2
    r = np.abs(r)
    rcut = max(1, np.abs(rcut))
    b = 1e-3 * p2
    p_geo = 2. * rcut ** b
    if np.sum(np.isinf(p_geo)):
        print rcut, b, p_geo
    if((type(r) == np.float64) or (type(r) == np.float)):
        if(r <= rcut):
            return 2.
        res = p_geo * r ** (-1. * b)
        return res
    else:
        res = np.ones_like(r) * 2.
        res[r >= rcut] = p_geo * r[r >= rcut] ** (-1. * b)
        return res


def LDF_vB(x, y, sigma, R, E, p=2.):
    r = (x ** 2 + y ** 2) ** 0.5
    if R < 0:
        norm = np.abs(sigma * np.pi * 2 ** 0.5 * ((scipy.special.erfc(-1. * R * 2 ** 0.5 / (2. * sigma))) * np.pi ** 0.5 * R + 2 ** 0.5 * sigma * np.exp(-R ** 2 / 2. / sigma ** 2)))
        return E / norm * LDF_vB_parts(r, sigma, R, p)
    else:
        norm = 1. / sigma ** 2 * 0.5 * np.exp(0.5 * R ** 2 / sigma ** 2) * sigma / ((scipy.special.erf(0.5 * R * 2 ** 0.5 / sigma) * np.pi ** 0.5 * 2 ** 0.5 * np.exp(0.5 * R ** 2 / sigma ** 2) * R + 2 * sigma) * np.pi)
        return E * norm * (LDF_vB_parts(r, sigma, R, p) + LDF_vB_parts(r, sigma, -R, p))


def LDF_vB_p(x, y, sigma, R, E, rcut=0, b=0, p=None, dxmax=None):
    r = (x ** 2 + y ** 2) ** 0.5
    if dxmax is not None:
        p = get_p_fixed(r, dxmax)
    else:
        if p is None:
            p = get_p2(r, rcut, b)
    return LDF_vB(x, y, sigma, R, E, p)


def LDF_geo(x, y, sigma, R, E, dxmax=None):
    r = (x ** 2 + y ** 2) ** 0.5
    p = 2.
    if dxmax is not None:
        rcut = get_rcut_geo(R, dxmax)
        b = get_b_geo(R, dxmax)
        p = get_p2(r, rcut, b)
    return LDF_vB(x, y, sigma, R, E, p)


def LDF_geo_spl(x, y, sigma, R, E, dxmax=None):
    r = (x ** 2 + y ** 2) ** 0.5
    p = 2.
    if dxmax is not None:
        rcut = get_rcut_geo_spl(dxmax)
        b = get_b_geo_spl(dxmax)
        p = get_p2(r, rcut, b)
    return LDF_vB(x, y, sigma, R, E, p)


def LDF_geo_dxmax(x, y, dxmax, E, obsheight=1564):
    r = (x ** 2 + y ** 2) ** 0.5
    rcut = get_rcut_geo_spl(dxmax)
    b = get_b_geo_spl(dxmax)
    p = get_p2(r, rcut, b)
    Ecorr = 1.
    if(obsheight == 1564):
        R = spl_geo_R_1564m(dxmax)
        sigma = spl_geo_sigma_1564m(dxmax)
        Ecorr = spl_geo_Ecorr_1564m(dxmax)
    elif(obsheight == 0):
        R = spl_geo_R_0m(dxmax)
        sigma = spl_geo_sigma_0m(dxmax)
        Ecorr = spl_geo_Ecorr_0m(dxmax)
    else:
        import sys
        print("requestes observation height of %.0fm is not available" % obsheight)
        sys.exit(-1)

    return LDF_vB(x, y, sigma, R, E, p) / Ecorr


# def LDF_vB_parts(r, E, sigma, R, p_geo=2.):
#     return E * 1e6 / sigma ** 2 * (np.exp(-1. * (np.abs(r - R) / (2 ** 0.5 * sigma)) ** p_geo))


def LDF_vB_parts(r, sigma, R, p=2.):
    return np.exp(-1. * (np.abs(r - R) / (2 ** 0.5 * sigma)) ** p)


def _LDF_vB_parts_normed(r, sigma, R, p=2.):  # only for plotting purposes
    norm = 1. / sigma ** 2 * 0.5 * np.exp(0.5 * R ** 2 / sigma ** 2) * sigma / ((scipy.special.erf(0.5 * R * 2 ** 0.5 / sigma) * np.pi ** 0.5 * 2 ** 0.5 * np.exp(0.5 * R ** 2 / sigma ** 2) * R + 2 * sigma) * np.pi)
    return norm * np.exp(-1. * (np.abs(r - R) / (2 ** 0.5 * sigma)) ** p)


def _LDF_vB_parts_normed2(r, sigma, R, p=2.):  # only for plotting purposes
    norm = np.abs(sigma * np.pi * 2 ** 0.5 * ((scipy.special.erfc(-1. * R * 2 ** 0.5 / (2. * sigma))) * np.pi ** 0.5 * R + 2 ** 0.5 * sigma * np.exp(-R ** 2 / 2. / sigma ** 2)))
    return 1. / norm * np.exp(-1. * (np.abs(r - R) / (2 ** 0.5 * sigma)) ** p)


def my_gamma2(xx, E, sigma, k=1.2, rcut=0, b=0, p=None, k_limit=0):
    if p is None:
        p = get_p2(np.abs(xx), rcut, b)
    if k < k_limit:
        return np.nan
    norm = (k + 1.) / (2. ** k * (2. * k + 2) ** (-0.5 * k)) / (sigma ** (k + 2.)) / (2 * np.pi) / scipy.special.gamma(0.5 * k + 1)
    if np.isnan(norm):
        print k, sigma, scipy.special.gamma(0.5 * k + 1)
        print norm
    # norm = 1
    return norm * E * np.abs(xx) ** k * np.exp(-(np.abs(xx) ** p / (p / (k + 1.) * (sigma) ** p)))


def LDF_vvB3(x, y, sigma, R, E, rcut=0, b=0, p=None):
    r = (x ** 2 + y ** 2) ** 0.5
    if(R < sigma):
        k = -1e-2 * (R - sigma)
        if R > 0:
            sigma = R + sigma
        return my_gamma2(r, E=E, sigma=sigma, k=k, rcut=rcut, b=b, p=p)
    else:
        if p is None:
            p = get_p2(r, rcut, b)
        return E / (sigma * R * np.pi ** 0.5 * 2 ** 0.5) * (LDF_vB_parts(r, sigma, R, p) - LDF_vB_parts(r, sigma, -1. * R, p)) / (2 * np.pi)


def LDF_ce_gamma(x, y, sigma, k, E, rcut=0, b=0, p=None):
    r = (x ** 2 + y ** 2) ** 0.5
    return my_gamma2(r, E=E, sigma=sigma, k=k, rcut=rcut, b=b, p=p, k_limit=0)


def LDF_ce_gamma2(x, y, sigma, k, E, p=None):
    r = (x ** 2 + y ** 2) ** 0.5
    return my_gamma2(r, E=E, sigma=sigma, k=k, p=p, k_limit=0.)


def LDF_ce(x, y, sigma, k, E, dxmax=None):
    r = (x ** 2 + y ** 2) ** 0.5
    rcut = 0
    b = 0
    if dxmax is not None:
        rcut = get_rcut_ce(k, dxmax)
        b = get_b_ce(k, dxmax)
    return my_gamma2(r, E=E, sigma=sigma, k=k, rcut=rcut, b=b)


def LDF_ce_dxmax(x, y, dxmax, E, obsheight=1564):
    r = (x ** 2 + y ** 2) ** 0.5
    k = get_k_ce(dxmax)
    Ecorr = 1.
    if(obsheight == 1564):
        sigma = spl_ce_sigma_1564m(dxmax)
        Ecorr = spl_ce_Ecorr_1564m(dxmax)
    elif(obsheight == 0):
        sigma = spl_ce_sigma_0m(dxmax)
        Ecorr = spl_ce_Ecorr_0m(dxmax)
    else:
        import sys
        print("requestes observation height of %.0fm is not available" % obsheight)
        sys.exit(-1)
    rcut = get_rcut_ce(k, dxmax)
    b = get_b_ce(k, dxmax)
    return my_gamma2(r, E=E, sigma=sigma, k=k, rcut=rcut, b=b) / Ecorr


# # geomagnetic rcut, b dependence:
def get_rcut_geo(R, dxmax, D=6.95619835e+02, a1=-2.91448613e-01, b1=3.68997075e+02, a2=5.21517895e-01):
    if R > 0:
        dxmax = np.array(dxmax)
        res = np.zeros_like(dxmax)
        res[dxmax < D] = a1 * dxmax[dxmax < D] + b1
        res[dxmax >= D] = a2 * (dxmax[dxmax >= D] - D) + a1 * D + b1
        res[res < 0] = 0
        return res
    else:
        return 0


def get_b_geo(R, dxmax):
    if R > 0:
        return 187.2971344
    else:
        D = 2.44572703e+02
        a1 = -2.43759928e-01
        b1 = 1.01603438e+02
        res = np.zeros_like(dxmax)
        res[dxmax < D] = a1 * dxmax[dxmax < D] + b1
        res[dxmax >= D] = a1 * D + b1
        return res
        # return 0


def get_b_geo_spl(dxmax):
    return spl_b_geo(dxmax)


def get_rcut_geo_spl(dxmax):
    """
    get the rcut parameter for the geomagnetic LDF

    the variation of the exponent as a function of distance to the shower axis
    is described with the parameters rcut and b. This function returns a parametriztion
    of rcut as a function of distance to shower maximum using B-splines

    Parameters
    ----------
    dxmax : float
        the density at the position of the air-shower maximum xmax in g/cm^2

    Returns
    -------
    rcut : float
        rcut parameter
    """
    return spl_rcut_geo(dxmax)


# charge-exess rcut, b dependence:
def get_rcut_ce(k, dxmax):
    if k < 1e-5:
        return np.zeros_like(dxmax)
    else:
        b = get_b_ce(k, dxmax)
        p0, p1, p2 = 2.90571462e+01, 1.97413284e-01, 1.80588511e-03
        return 0.5 * (-p1 + ((4 * b - 4 * p0) * p2 + p1 ** 2) ** 0.5) / p2


def get_b_ce(k, dxmax):
    if k < 1e-5:
        return 141.82811531 - 0.23893423 * dxmax
    else:
        return 58.99640913 + 0.32596894 * dxmax


# # parametrizations

def get_k_ce(dxmax, a=3.74875934e+02, b=-3.89726496e+00, c=3.33172563e+00, d=2.86941234e-03):
    t = dxmax - a
    res = b + (c - b) / (1 + np.exp(-d * t))
    if not (isinstance(res, np.float64)):
        res[res < 0] = 0
    else:
        if res < 0:
            res = 0
    return res


def get_sigma_ce_0m(dxmax, a=7.05297446, b=60.61689435, c=88.0811727, d=18):
    res = d / c ** 0.5 * (c + ((dxmax - b) / a) ** 2) ** 0.5
    if not (isinstance(res, np.float64)):
        res[dxmax <= b] = d
    else:
        if dxmax <= b:
            res = d
    return res


def get_sigma_ce_1564m(dxmax, a=5.96339704, b=87.0173615, c=93.13858926, d=18):
    res = d / c ** 0.5 * (c + ((dxmax - b) / a) ** 2) ** 0.5
    if not (isinstance(res, np.float64) or isinstance(res, np.float)):
        res[dxmax <= b] = d
    else:
        if (dxmax <= b):
            res = d
    return res


def get_sigma_geo_0m(dxmax, a=54.25819242, b=5.37127055, c=3.10225619, d=71.32248597, e=597.67805425):
        return b * (c + ((dxmax - e) / a) ** 2) ** 0.5 + d


def get_sigma_geo_1564m(dxmax, a=72.32433156, b=7.46852063, c=1.27128895, d=78.6660653, e=596.54875896):
    return b * (c + ((dxmax - e) / a) ** 2) ** 0.5 + d


def get_R_geo_0m(dxmax, a=303.86227905, b=-481.55459933, c=143.85216778):
    return a + np.log10(dxmax) * b + np.log10(dxmax) ** 2 * c


def get_R_geo_1564m(dxmax, a=-345.16239579, b=-32.88107713, c=67.63247429):
    return a + np.log10(dxmax) * b + np.log10(dxmax) ** 2 * c


def get_a(rho, magnetic_field_strength=0.243):
    """
    returs the relative charge-excess fraction as a function of air density at the shower maximum Xmax

    This prarametrization is from Glaser et al., JCAP 09(2016)024.
    The relative charge-excess fraction is defined as a = sin(alpha) sqrt(Ece/Egeo).

    Parameters
    ----------
    rho : float
        the density at the position of the air-shower maximum xmax in g/
    magnetic_field_strength : float
        the magnetic field strength in Gauss at the position of observation.

    Returns
    -------
    a : float
        relativ charge-excess strength a

    """
    average_density = 648.18353008270035
    a = -0.23604683 + 0.43426141 * np.exp(1.11141046 * 1e-3 * (rho - average_density))
    return a / (magnetic_field_strength / 0.243) ** 0.9


def LDF_geo_ce(x, y, Erad, dxmax, zenith, azimuth, core=np.array([0, 0]),
               obsheight=1564.,
               magnetic_field_vector=np.array([0, .1971, -.1418])):
    """
    returns the energy fluence between 30-80 MHz at a position (x, y) in the vxB-vx(vxB) frame

    This prarametrization is from Glaser et al., JCAP 09(2016)024.
    The relative charge-excess fraction is defined as a = sin(alpha) sqrt(Ece/Egeo).

    Parameters
    ----------
    x : float
        x coordinate in the vxB-vx(vxB) frame where the core position is at the origin
    y : float
        y coordinate in the vxB-vx(vxB) frame where the core position is at the origin
    Erad : float
        radiation energy (in the 30-80 MHz band) in eV
    dxmax: float
        distance from the observation height to the shower maximum Xmax in g/cm^2
    zenith: float
        zenith angle of the air showers incoming direction, 0deg is the zenith
    azimuth: float
        azimuth angle of the air showers incoming direction, 0deg is East, counting counter-clockwise
    core: array
        position of the shower core in the vxB-vx(vxB) frame. Only the x and y coordinate are used.
    obsheight: float
        observation altitude in meters
    magnetic_field_vector: three-vector as numpy.array
        the vector of the local geomagnetic field in Gauss. x is Eastwards, y is Northwards and z is upwards

    Returns
    -------
    f: float
        energy fluence in eV/m^2
    fvB: float
        energy fluence in vxB polarization in eV/m^2
    fvvB: float
        energy fluence in vx(vxB) polarization in eV/m^2
    fgeo: float
        geomagnetic energy fluence in eV/m^2
    fce: float
        charge-excess energy fluence in eV/m^2

    """
    # calculate gemagnetic and charge-excess radation energies from full radiation energy, Dxmax and zenith angles.
    Xatm = atmc.get_atmosphere(zenith, h_low=obsheight)
    xmax = Xatm - dxmax
    rho = atmc.get_density(zenith, xmax)
    a = get_a(rho, magnetic_field_strength=np.linalg.norm(magnetic_field_vector))
    sinalpha = hp.GetSineAngleToLorentzForce(zenith, azimuth, magnetic_field_vector)
    Egeo = Erad / (1 + (a / sinalpha) ** 2)
    Ece = Erad - Egeo
    # print "%.0f %.2g %.2g %.2g %.2f %.2f" % (dxmax, Erad, Egeo, Ece, a, sinalpha)

    x -= core[0]
    y -= core[1]

    # get energy fluence of geomagnetic and charge-excess component
    fce = LDF_ce_dxmax(x, y, dxmax, Ece, obsheight=obsheight)
    fgeo = LDF_geo_dxmax(x, y, dxmax, Egeo, obsheight=obsheight)

    # combine the two emission processes depending on the position relative to the shower axis
    az = np.arctan2(y, x)
    fvB = (fgeo ** 0.5 + fce ** 0.5 * np.cos(az)) ** 2
    fvvB = fce * np.sin(az) ** 2
    f = fvB + fvvB
    return f, fvB, fvvB, fgeo, fce


def LDF_geo_ce2(x, y, Egeo, Ece, dxmax, obsheight=1564.):
    """
    returns the energy fluence between 30-80 MHz at a position (x, y) in the vxB-vx(vxB) frame

    This prarametrization is from Glaser et al., JCAP 09(2016)024.
    The relative charge-excess fraction is defined as a = sin(alpha) sqrt(Ece/Egeo).

    Parameters
    ----------
    x : float
        x coordinate in the vxB-vx(vxB) frame where the core position is at the origin
    y : float
        y coordinate in the vxB-vx(vxB) frame where the core position is at the origin
    Egeo : float
        geomagnetic radiation energy (in the 30-80 MHz band) in eV
    Ece : float
        charge-excess radiation energy (in the 30-80 MHz band) in eV
    dxmax: float
        distance from the observation height to the shower maximum Xmax in g/cm^2
    obsheight: float
        observation altitude in meters

    Returns
    -------
    f: float
        energy fluence in eV/m^2
    fvB: float
        energy fluence in vxB polarization in eV/m^2
    fvvB: float
        energy fluence in vx(vxB) polarization in eV/m^2
    fgeo: float
        geomagnetic energy fluence in eV/m^2
    fce: float
        charge-excess energy fluence in eV/m^2

    """

    # get energy fluence of geomagnetic and charge-excess component
    fgeo = LDF_geo_dxmax(x, y, dxmax, Egeo, obsheight=obsheight)
    fce = LDF_ce_dxmax(x, y, dxmax, Ece, obsheight=obsheight)

    # combine the two emission processes depending on the position relative to the shower axis
    az = np.arctan2(y, x)
    fvB = (fgeo ** 0.5 + fce ** 0.5 * np.cos(az)) ** 2
    fvvB = fce * np.sin(az) ** 2
    f = fvB + fvvB
    return f, fvB, fvvB, fgeo, fce








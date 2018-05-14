import numpy as np
import scipy.special
import pickle
from scipy.interpolate import BSpline
import atmosphere as atm
import os

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


def LDF_geo_ce(x, y, Erad, dxmax, zenith, azimuth, core=np.array([0, 0]),
               obsheight=1564.,
               magnetic_field_vector=np.array([0, .1971, .1418])):
    """
    returns the energy fluence between 30-80 MHz at position (x, y) in the vxB-vx(vxB) frame

    Parametrization with two parameters.

    The function depends on the radiation energy and the distance to the shower maximum.
    The zenith and azimuth angle are just needed to calculate the geomagnetic and
    charge-excess radiation energies from the full radiation energy by using
    the prarametrization of the relative charge-excess strength a as a function of
    air density at the shower maximum is from Glaser et al., JCAP 09(2016)024.
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
    Xatm2 = atm.get_atmosphere(obsheight)
    h2 = atm.get_vertical_height(Xatm2 - dxmax * np.cos(zenith))
    rho = atm.get_density(h2)

    a = get_a(rho, magnetic_field_strength=np.linalg.norm(magnetic_field_vector))
    sinalpha = get_sine_angle_to_lorentz_force(zenith, azimuth, magnetic_field_vector)
    Egeo = Erad / (1 + (a / sinalpha) ** 2)
    Ece = Erad - Egeo

    x2 = x - core[0]
    y2 = y - core[1]

    # get energy fluence of geomagnetic and charge-excess component
    fce = LDF_ce_dxmax(x2, y2, dxmax, Ece, obsheight=obsheight)
    fgeo = LDF_geo_dxmax(x2, y2, dxmax, Egeo, obsheight=obsheight)

    # combine the two emission processes depending on the position relative to the shower axis
    az = np.arctan2(y2, x2)
    fvB = (fgeo ** 0.5 + fce ** 0.5 * np.cos(az)) ** 2
    fvvB = fce * np.sin(az) ** 2
    f = fvB + fvvB
    return f, fvB, fvvB, fgeo, fce


def LDF_geo_ce2(x, y, Egeo, Ece, dxmax, obsheight=1564.):
    """
    returns the energy fluence between 30-80 MHz at a position (x, y) in the vxB-vx(vxB) frame

    Parametrization with three free parameters.

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


def LDF_geo_dxmax(x, y, dxmax, E, obsheight=1564):
    """
    parametrization of the geomagnetic LDF as a function of radiation energy E
    and distance to shower maximum dxmax.

    Parameters
    ----------
    x: float or array
        x position in vxB-vx(vxB) frame
    y: float or array
        y position in vxB-vx(vxB) frame
    E: float
        geomagnetic radiation energy
    dxmax: float (optional)
        distance to shower maximum in g/cm^2
    obsheight: float
        observation height, can be either 1564m (height of the AERA detector) or
        0m (height of the LOFAR detector)

    Returns
    ----------
    f: float
        energy fluence

    """
    r = (x ** 2 + y ** 2) ** 0.5
    rcut = get_rcut_geo_spl(dxmax)
    b = get_b_geo_spl(dxmax)
    p = get_p(r, rcut, b)
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


def LDF_ce_dxmax(x, y, dxmax, E, obsheight=1564):
    """
    parametrization of the charge-excess LDF as a function of radiation energy E
    and distance to shower maximum dxmax.

    Parameters
    ----------
    x: float or array
        x position in vxB-vx(vxB) frame
    y: float or array
        y position in vxB-vx(vxB) frame
    E: float
        charge-excess radiation energy
    dxmax: float (optional)
        distance to shower maximum in g/cm^2
    obsheight: float
        observation height, can be either 1564m (height of the AERA detector) or
        0m (height of the LOFAR detector)

    Returns
    ----------
    f: float
        energy fluence
    """
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


def LDF_geo_spl(x, y, sigma, R, E, dxmax=None):
    """
    parametrization of the geomagnetic LDF as a function of E, sigma and R. If
    dxmax is provided, the parametrization of the exponent p is used.

    Parameters
    ----------
    x: float or array
        x position in vxB-vx(vxB) frame
    y: float or array
        y position in vxB-vx(vxB) frame
    sigma: float
        width of LDF function
    R: float
        radius of Cherenkov ring
    E: float
        geomagnetic radiation energy
    dxmax: float (optional)
        distance to shower maximum in g/cm^2

    Returns
    ----------
    f: float
        energy fluence
    """
    r = (x ** 2 + y ** 2) ** 0.5
    p = 2.
    if dxmax is not None:
        rcut = get_rcut_geo_spl(dxmax)
        b = get_b_geo_spl(dxmax)
        p = get_p(r, rcut, b)
    return LDF_vB(x, y, sigma, R, E, p)


def LDF_ce(x, y, sigma, k, E, dxmax=None):
    """
    parametrization of the charge-excess LDF as a function of E, sigma and k. If
    dxmax is provided, the parametrization of the exponent p is used.

    Parameters
    ----------
    x: float or array
        x position in vxB-vx(vxB) frame
    y: float or array
        y position in vxB-vx(vxB) frame
    sigma: float
        width of LDF function
    k: float
        k parameter of LDF
    E: float
        charge-excess radiation energy
    dxmax: float (optional)
        distance to shower maximum in g/cm^2

    Returns
    ----------
    f: float
        energy fluence
    """
    r = (x ** 2 + y ** 2) ** 0.5
    rcut = 0
    b = 0
    if dxmax is not None:
        rcut = get_rcut_ce(k, dxmax)
        b = get_b_ce(k, dxmax)
    return my_gamma2(r, E=E, sigma=sigma, k=k, rcut=rcut, b=b)


def get_p(r, rcut, p2):
    """
    parametrization of the variation of the exponent as a function of distance r

    Parameters
    ----------
    r: float or array
        distance to the shower axis
    rcut: float
        parameter rcut
    b: float
        parameter b

    Returns
    ----------
    p: float or array
        the exponent p
    """
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


def LDF_vB_parts(r, sigma, R, p=2.):
    return np.exp(-1. * (np.abs(r - R) / (2 ** 0.5 * sigma)) ** p)


def my_gamma2(xx, E, sigma, k=1.2, rcut=0, b=0, p=None, k_limit=0):
    if p is None:
        p = get_p(np.abs(xx), rcut, b)
    if k < k_limit:
        return np.nan
    norm = (k + 1.) / (2. ** k * (2. * k + 2) ** (-0.5 * k)) / (sigma ** (k + 2.)) / (2 * np.pi) / scipy.special.gamma(0.5 * k + 1)
    return norm * E * np.abs(xx) ** k * np.exp(-(np.abs(xx) ** p / (p / (k + 1.) * (sigma) ** p)))


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
        return 146.92691815 - 0.25112664 * dxmax
    else:
        return 55.55667917 + 0.32392104 * dxmax


def get_k_ce(dxmax, a=5.80505613e+02, b=-1.76588481e+00, c=3.12029983e+00, d=3.73038601e-03):
    t = dxmax - a
    res = b + (c - b) / (1 + np.exp(-d * t))
    if not (isinstance(res, np.float64)):
        res[res < 0] = 0
    else:
        if res < 0:
            res = 0
    return res


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


def get_lorentz_force_vector(zenith, azimuth, magnetic_field_vector):
    """
    get the Lorentz force as a cartesian 3-vector
    """
    showerAxis = spherical_to_cartesian(zenith, azimuth)
    magnetic_field_vector_normalized = magnetic_field_vector / np.linalg.norm(magnetic_field_vector)
    return np.cross(showerAxis, magnetic_field_vector_normalized)


def get_sine_angle_to_lorentz_force(zenith, azimuth, magnetic_field_vector=None):
    """
    returns the sine of the angle between shower axis and Lorentz force vector
    """
    # we use the tanspose of the vector or matrix to be able to always use axis=0
    return np.linalg.norm(get_lorentz_force_vector(zenith, azimuth, magnetic_field_vector).T, axis=0)


def spherical_to_cartesian(zenith, azimuth):
    """
    converts zenith and azimuth angle into a cartesian 3-vector
    """
    sinZenith = np.sin(zenith)
    x = sinZenith * np.cos(azimuth)
    y = sinZenith * np.sin(azimuth)
    z = np.cos(zenith)
    if hasattr(zenith, '__len__') and hasattr(azimuth, '__len__'):
        return np.array(zip(x, y, z))
    else:
        return np.array([x, y, z])

import LDF
import numpy as np
import matplotlib.pyplot as plt


Erad = 100e6  # 100 MeV
dxmax = 1000  # in g/cm^2
zenith = np.deg2rad(60)
azimuth = np.deg2rad(270)  # from south
obsheight = 1564.
core = np.array([0, 0, 0])
magnetic_field_vector = np.array([0, .1971, .1418])  # magnetic field at the Pierre Auger Observatory in Gauss

xxx = np.linspace(-700, 700, 1000)
yyy = np.zeros_like(xxx)
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
f, fvB, fvvB, fgeo, fce = LDF.LDF_geo_ce(xxx, yyy, Erad, dxmax, zenith, azimuth, core, obsheight, magnetic_field_vector)  # get energy fluence anlong the vxB axis
ax1.plot(xxx, f, '-C0')
f, fvB, fvvB, fgeo, fce = LDF.LDF_geo_ce(yyy, xxx, Erad, dxmax, zenith, azimuth, core, obsheight, magnetic_field_vector)  # get energy fluence anlong the vx(vxB) axis
ax2.plot(xxx, f, '-C0')

title = r"$E_\mathrm{rad}$ = %.0f MeV, $D_{X_\mathrm{max}}$ = %.0f g/cm$^2$, theta = %.0f$^\circ$, phi = %.0f$^\circ$, h = %.0fm" % (Erad * 1e-6, dxmax, np.rad2deg(zenith),
                                                                                                                                     np.rad2deg(azimuth), obsheight)
fig.suptitle(title)

ax1.set_xlim(-600, 600)
ax2.set_xlim(-600, 600)
ax1.set_ylim(0)

ax1.set_ylabel(r"$f$ [eV/m$^2$]")
ax1.set_xlabel(r"position in $v \times B$ [m]")
ax2.set_xlabel(r"position in $v \times (v \times B)$ [m]")
plt.tight_layout()
fig.subplots_adjust(top=0.93)

plt.show()


from LDF import *

splines = [spl_rcut_geo,
           spl_b_geo,
           spl_geo_R_0m,
           spl_geo_R_1564m,
           spl_geo_sigma_0m,
           spl_geo_sigma_1564m,
           spl_ce_sigma_0m,
           spl_ce_sigma_1564m,
           spl_geo_Ecorr_1564m,
           spl_geo_Ecorr_0m,
           spl_ce_Ecorr_1564m,
           spl_ce_Ecorr_0m]
spline_names = ['spl_rcut_geo',
           'spl_b_geo',
           'spl_geo_R_0m',
           'spl_geo_R_1564m',
           'spl_geo_sigma_0m',
           'spl_geo_sigma_1564m',
           'spl_ce_sigma_0m',
           'spl_ce_sigma_1564m',
           'spl_geo_Ecorr_1564m',
           'spl_geo_Ecorr_0m',
           'spl_ce_Ecorr_1564m',
           'spl_ce_Ecorr_0m']
captions = [r'$r_\mathrm{cut}$ as a function of \dxmax for the geomagnetic function',
            r'$b$ as a function of \dxmax for the geomagnetic function',
            r'$R_\mathrm{geo}$ as a function of \dxmax for an observation altitude of \SI{0}{m asl}',
            r'$R_\mathrm{geo}$ as a function of \dxmax for an observation altitude of \SI{1564}{m asl}',
            r'$\sigma_\mathrm{geo}$ as a function of \dxmax for an observation altitude of \SI{0}{m asl}',
            r'$\sigma_\mathrm{geo}$ as a function of \dxmax for an observation altitude of \SI{1564}{m asl}',
            r'$\sigma_\mathrm{ce}$ as a function of \dxmax for an observation altitude of \SI{0}{m asl}',
            r'$\sigma_\mathrm{ce}$ as a function of \dxmax for an observation altitude of \SI{1564}{m asl}',
            r'the deviation between $E_\mathrm{geo}$ and $E\'_\mathrm{geo}$ for an observation altitude of \SI{1564}{m asl}',
            r'the deviation between $E_\mathrm{geo}$ and $E\'_\mathrm{geo}$ for an observation altitude of \SI{0}{m asl}',
            r'the deviation between $E_\mathrm{ce}$ and $E\'_\mathrm{ce}$ for an observation altitude of \SI{1564}{m asl}',
            r'the deviation between $E_\mathrm{ce}$ and $E\'_\mathrm{ce}$ for an observation altitude of \SI{0}{m asl}']

# for tmp, caption in zip(splines, captions):
#     print 'printing parameters for {}'.format(caption)
#     print 'knots'
#     for t in tmp.t:
#         print "{}, ".format(t)
#     print 'coefficients'
#     for c in tmp.c:
#         print "{}, ".format(c)
#     print
#     print

for i, tmp in enumerate(splines):
    if not os.path.exists("spline_constants"):
        os.mkdir("spline_constants")
    filename = os.path.join("spline_constants", "{}.txt".format(spline_names[i]))
    X = np.array([tmp.t[2:-2], tmp.c]).T
    np.savetxt(filename, X, header='{}\nknots\tcoefficients'.format(captions[i]))

# for tmp, caption in zip(splines, captions):
#     output = "\\begin{table} \n"
#     output += "\\begin{tabular}{c c}\\hline \\hline \n"
#     output += "knots & coefficients \\\\ \\hline \n"
#     for c, t in zip(tmp.t, tmp.c):
#         output += "{:.4g} & {:.4g} \\\\ \n".format(c, t)
#     output += "\\hline \\hline \n"
#     output += "\\end{tabular} \n"
#     output += "\\caption{"
#     output += "Parametrization of " + caption + ".}\n"
#     output += "\\end{table} \n"
#     print output

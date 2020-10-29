from ArrayScript import RealArray
import numpy as np
import sys

if len(sys.argv) !=9:
    print("Syntax is 'python WrapperScript.py Nside M m pixel_offset phi_x phi_y noise nmax'")
else:
    Nside = int(sys.argv[1])
    M = int(sys.argv[2])
    m = int(sys.argv[3])
    pixel_offset = int(sys.argv[4])
    phi_x = float(sys.argv[5])
    phi_y = float(sys.argv[6])
    phi = np.array([phi_x, phi_y])
    noise = float(sys.argv[7])
    nmax = int(sys.argv[8])

    arr = RealArray(Nside, noise, M)
    arr.geometry_error(pixel_offset)
    arr.pointing_error(phi)
    arr.start_data()
    arr.create_fit(m, nmax)
    isolve = arr.itersolve
    fname = '_'.join(sys.argv[1:])

    np.save('./data/'+fname+'_vis', isolve[0])
    np.save('./data/'+fname+'_beam', isolve[1])
    np.save('./data/'+fname+'_chi', isolve[-1])

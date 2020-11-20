from ArrayScript import RealArray
import numpy as np
import sys
import pickle

if len(sys.argv)==8:
    Nside = int(sys.argv[1])
    M = int(sys.argv[2])
    m = int(sys.argv[3])
    pixel_offset = float(sys.argv[4])
    phi_mag = float(sys.argv[5])
    snr = float(sys.argv[6])
    nmax = int(sys.argv[7])
    print(Nside, M, m, pixel_offset, phi_mag, snr, nmax)

    arr = RealArray(Nside, M)
    arr.geometry_error(pixel_offset)
    arr.pointing_error(phi_mag)
    arr.create_beams()
    print("Made beams")
    arr.errorless_data()
    arr.base_noise()
    arr.add_noise(snr)
    arr.create_fit(m, nmax)
    isolve = arr.itersolve
    print(Nside, M, m, pixel_offset, phi_mag, snr, snr_type, nmax)
    fname = '_'.join(sys.argv[1:])
    dat = arr.data
    
    pickle_name = fname+"_pickle.obj"
    pick_file = open('./data/' + pickle_name, 'wb')
    pickle.dump(arr, pick_file)
    pick_file.close()

    np.save('./data/'+fname+'_data', dat)
    np.save('./data/'+fname+'_vis', isolve[0])
    np.save('./data/'+fname+'_beam', isolve[1])
    np.save('./data/'+fname+'_chi', isolve[-1])
    
elif len(sys.argv)==6:
    vis_file = input('Path to saved vis: ')
    beam_file = input('Path to saved beams: ')
    data_file = input('Path to saved data: ')
    fname = '_'.join(sys.argv[1:-1])
    
    bguess = np.load(vis_file)
    ibeam = np.load(beam_file)
    data = np.load(data_file)
    
    Nside = int(sys.argv[1])
    M = int(sys.argv[2])
    m = int(sys.argv[3])
    nmax = int(sys.argv[5])

    arr = RealArray(Nside, noise, M)
    arr.set_data(data)
    arr.create_fit(m, nmax, bguess=bguess, ibeam=ibeam)
    isolve = arr.itersolve
    new_name = fname+'_extension'
    
    np.save('./data/'+new_name+'_vis', isolve[0])
    np.save('./data/'+new_name+'_beam', isolve[1])
    np.save('./data/'+new_name+'_chi', isolve[-1])
else:
    print("Syntax is 'python WrapperScript.py Nside M m pixel_offset phi_mag noise nmax [Load Previous?]'")

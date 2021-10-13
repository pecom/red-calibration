from ArrayScript import RealArray
import numpy as np
import sys
import pickle

Nside = int(sys.argv[1])
geom_err = float(sys.argv[2])
point_err = float(sys.argv[3])
M = 29
m = 3
# snr_vals = [1e-3, 1e-1, 1, 10, 12.5, 25, 50]
snr_vals = [1, 2, 5, 10, 20, 50, 100]
# snr_vals = [1,2]

sample_array = RealArray(Nside, 29)
sample_array.geometry_error(geom_err)
sample_array.pointing_error(point_err)
sample_array.create_beams()
sample_array.errorless_data()
sample_array.base_noise()
datadir = './data/gauss/'
fin_chi2= []
fin_sigmas = []
file_name = 'Nside'+str(Nside)+'_geomerr'+str(geom_err) + '_pointerr'+str(point_err)
good_bins = sample_array.bin_beams(sample_array.fake_beams)
best_bins = sample_array.bin_beams(sample_array.beams)
print("Made good bins")
for snr in snr_vals:
    sample_array.add_noise(snr)
    sample_array.create_fit(3, nmax=10)
    isolve = sample_array.itersolve

    sample_array.gauss_fit(3, good_bins, nmax=10, wien=True, bg=isolve[0], ib=isolve[1])
#     sample_array.compare_to_red()
    
    isolve = sample_array.gauss_itersolve
#     red_chi = sample_array.red_chis  
    
    chi = isolve[2]
    sigmas = isolve[4]
    fin_chi = chi[-1]
    fin_sig = sigmas[-1]
    
    fin_chi2.append(fin_chi)
    fin_sigmas.append(fin_sig)

    fname = "plot_pickle_"+file_name+"_snr"+str(snr) + ".obj"
    pick_file = open(datadir + "pickles/" + fname, 'wb')
    pickle.dump(sample_array, pick_file)
    pick_file.close()
    
    print(Nside, M, m, geom_err, point_err, snr)
    
scores = np.array(fin_chi2)
sigmas = np.array(fin_sigmas)
snrs = np.array(snr_vals)

bnoise = sample_array.bn
errorless = sample_array.errorless


np.save(datadir + file_name + '_scores', scores)
np.save(datadir + file_name + '_sigmas', sigmas)
np.save(datadir + file_name + '_snrs', snrs)
np.save(datadir + file_name + '_errorless', errorless)
np.save(datadir + file_name + '_basenoise', bnoise)
print("All done!")

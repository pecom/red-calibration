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

sample_array = RealArray(Nside, 29)
sample_array.geometry_error(geom_err)
sample_array.pointing_error(point_err)
sample_array.create_beams()
sample_array.errorless_data()
sample_array.base_noise()
datadir = './data/fid_snr/'
fin_scores= []
ns = []
red_scores = []
file_name = 'Nside'+str(Nside)+'_geomerr'+str(geom_err) + '_pointerr'+str(point_err)
for snr in snr_vals:
    sample_array.add_noise(snr)
    sample_array.create_fit(3, 0)
    sample_array.compare_to_red()
    
    isolve = sample_array.itersolve
    red_chi = sample_array.red_chis  
    
    chi = isolve[-1]
    fin_chi = chi[-1]
    iternum = len(chi)
    rchi = red_chi[-1]
    
    fin_scores.append(fin_chi)
    red_scores.append(rchi)
    ns.append(iternum)

    fname = "plot_pickle_"+file_name+"_snr"+str(snr) + ".obj"
    pick_file = open(datadir + "pickles/" + fname, 'wb')
    pickle.dump(sample_array, pick_file)
    pick_file.close()
    
    print(Nside, M, m, geom_err, point_err, snr)
    
scores = np.array(fin_scores)
snrs = np.array(snr_vals)
ns = np.array(ns)
reds = np.array(red_scores)
bnoise = sample_array.bn
errorless = sample_array.errorless


np.save(datadir + file_name + '_scores', scores)
np.save(datadir + file_name + '_snrs', snrs)
np.save(datadir + file_name + '_redscores', reds)
np.save(datadir + file_name + '_ns', ns)
np.save(datadir + file_name + '_errorless', errorless)
np.save(datadir + file_name + '_basenoise', bnoise)
print("All done!")

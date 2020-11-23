from ArrayScript import RealArray
import numpy as np
import sys
import pickle

Nside = int(sys.argv[1])
error_mag = float(sys.argv[2])
M = 29
m = 3
# snr_vals = [1e-3, 1e-1, 1, 10, 12.5, 25, 50]
# snr_vals = [1, 2, 5, 10, 20, 50, 100]
snr_vals = [1, 2, 5]

sample_array = RealArray(Nside, 29)
sample_array.geometry_error(error_mag)
sample_array.pointing_error(error_mag)
sample_array.create_beams()
sample_array.errorless_data()
sample_array.base_noise()
fin_scores= []
ns = []
red_scores = []
file_name = 'Nside'+str(Nside)+'_Error'+str(error_mag)
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
    pick_file = open('./data/' + fname, 'wb')
    pickle.dump(sample_array, pick_file)
    pick_file.close()
    
    print(Nside, M, m, error_mag, error_mag, snr)
    
scores = np.array(fin_scores)
snrs = np.array(snr_vals)
ns = np.array(ns)
reds = np.array(red_scores)
bnoise = sample_array.bn
errorless = sample_array.errorless

np.save('./data/' + file_name + '_scores', scores)
np.save('./data/' + file_name + '_snrs', snrs)
np.save('./data/' + file_name + '_redscores', reds)
np.save('./data/' + file_name + '_ns', ns)
np.save('./data/' + file_name + '_errorless', errorless)
np.save('./data/' + file_name + '_basenoise', bnoise)
print("All done!")

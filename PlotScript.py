from ArrayScript import RealArray
import numpy as np
import sys

Nside = int(sys.argv[1])
error_mag = float(sys.argv[2])
M = 29
m = 3
snr_vals = [1e-3, 1, 10, 12.5, 25, 50, 100]

sample_array = RealArray(Nside, 29)
sample_array.geometry_error(error_mag)
sample_array.pointing_error(error_mag)
sample_array.create_beams()
sample_array.errorless_data()
sample_array.base_noise()
fin_scores= []
ns = []
for snr in snr_vals:
    sample_array.add_noise(snr)
    sample_array.create_fit(3, 0)
    isolve = sample_array.itersolve
    chi = isolve[-1]
    fin_chi = chi[-1]
    iternum = len(chi)
    ns.append(iternum)
    fin_scores.append(fin_chi)
    print(Nside, M, m, error_mag, error_mag, snr)
    
file_name = 'Nside'+str(Nside)+'_Error'+str(error_mag)
scores = np.array(fin_scores)
snrs = np.array(snr_vals)
ns = np.array(ns)
bnoise = sample_array.bn
errorless = sample_array.errorless

np.save('./data/' + file_name + '_scores', scores)
np.save('./data/' + file_name + '_snrs', snrs)
np.save('./data/' + file_name + '_ns', ns)
np.save('./data/' + file_name + '_errorless', errorless)
np.save('./data/' + file_name + '_basenoise', bn)
print("All done!")

from ArrayScript import RealArray
import numpy as np
import sys
import pickle

Nside = int(sys.argv[1])
error_mag = float(sys.argv[2])
M = 29
m = 3
# snr_vals = [1e-3, 1e-1, 1, 10, 12.5, 25, 50]
snr_vals = [1, 2, 5, 10, 20, 50, 100]
# snr = 10

sample_array = RealArray(Nside, 29)
sample_array.geometry_error(error_mag)
sample_array.pointing_error(error_mag)
sample_array.create_beams()

fin_scores= []
chi_diffs = []
red_scores = []
file_start = 'Nside'+str(Nside)+'_Error'+str(error_mag)
for snr in snr_vals:
    nsky_scores = []
    nsky_rscore = []
    for i in range(5):
        file_name = file_start + '_SNR'+str(snr)+'_Nsky'+str(i)
        sample_array.errorless_data()
        sample_array.base_noise()
        sample_array.add_noise(snr)
        if i > 0:
            sample_array.create_fit(3,0, ib=isolve[1])
            sample_array.compare_to_red(gain_guess=chi_gain)
        else:
            sample_array.create_fit(3, 0)
            sample_array.compare_to_red()

        chi_gain = sample_array.red_chig
        isolve = sample_array.itersolve
        red_chi = sample_array.red_chis  
    
        chi = isolve[-1]
        fin_chi = chi[-1]
        rchi = red_chi[-1]

        nsky_rscore.append(rchi)                
        nsky_scores.append(fin_chi)

        fname = "plot_pickle_"+file_name + ".obj"
        pick_file = open('./data/fid_snr/Nsky/pickles/' + fname, 'wb')
        pickle.dump(sample_array, pick_file)
        pick_file.close()

        print(Nside, M, m, error_mag, error_mag, snr)
    
    fin_scores.append(nsky_scores)
    red_scores.append(nsky_rscore)
    
scores = np.array(fin_scores)
redscs = np.array(red_scores)

np.save('./data/fid_snr/Nsky/' + file_name + '_scores', scores)
np.save('./data/fid_snr/Nsky/' + file_name + '_redscores', redscs)
print("All done!")

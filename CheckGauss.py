from ArrayScript import RealArray
import numpy as np
import sys
import pickle

Nside = int(sys.argv[1])
geom_err = float(sys.argv[2])
point_err = float(sys.argv[3])
M = 11
m = 3
# snr_vals = [1e-3, 1e-1, 1, 10, 12.5, 25, 50]
# snr_vals = [1, 2, 5, 10, 20, 50, 100]
snr_vals = [1, 50, 100, 20, 10, 5, 2]
# snr_vals = [100]

sample_array = RealArray(Nside, M)
sample_array.geometry_error(geom_err)
sample_array.pointing_error(point_err)
nm_fac = (2*M - 1)//3//2
tfact = (nm_fac*2) + 1
sfac = (1/tfact)**2 
rms_fac = (m/M)**2
# nm_fact = 1 
# tfact = 1
# sfac = 1
# rms_fac = 1
sample_array.create_beams(s_fac=sfac, v_var=rms_fac)
sample_array.errorless_data(v_var=tfact)
sample_array.base_noise()
datadir = './data/gauss/gtest/'
fin_chi2= []
fin_sigmas = []
file_name = 'run4Nside'+str(Nside)+'_geomerr'+str(geom_err) + '_pointerr'+str(point_err)
good_bins = sample_array.bin_beams(sample_array.fake_beams)
best_bins = sample_array.bin_beams(sample_array.beams)
best_bins *= (1/best_bins[:,1,1])[:,None,None]
tgain = sample_array.overall_gains
tbeam = sample_array.beams/tgain[:,None,None]
tvis = sample_array.vistrue
beam_submit = tbeam
for i, snr in enumerate(snr_vals):
    sample_array.add_noise(snr)
    print(sample_array.data[:10])
    print(np.var(sample_array.data.real))
    # sample_array.create_fit(m, nmax=10, wien=False, bg=tvis, ib=tbeam, gg=tgain, alph=1)
    # isolve = sample_array.itersolve

    if i==0:
        sample_array.gauss_fit(m, good_bins, nmax=3, wien=True, ib=good_bins, gg=tgain, alph=.4)
    else:
        sample_array.gauss_fit(m, good_bins, nmax=3, wien=True, bg=isolve[0], ib=isolve[1], gg=isolve[2], alph=.4)
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


# np.save(datadir + file_name + '_scores', scores)
# np.save(datadir + file_name + '_sigmas', sigmas)
# np.save(datadir + file_name + '_snrs', snrs)
# np.save(datadir + file_name + '_errorless', errorless)
# np.save(datadir + file_name + '_basenoise', bnoise)
print("All done!")

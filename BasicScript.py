#!/usr/bin/env python
# coding: utf-8

import numpy as np
# import matplotlib.pyplot as plt
from scipy import stats
from scipy import sparse
from scipy import optimize
from scipy import signal
from scipy import linalg
from functools import reduce
from collections import defaultdict


class BasicArray:
    
    def __init__(self, Nside):
        self.Nside = Nside
        self.Nant = Nside**2
        self.gains = self.make_gains(Nside)
        self.data = self.make_data(Nside, self.gains)

    def make_guesses(self):
        self.gain_guess = np.random.normal(0, 1, (self.Nant, 2)).view(dtype=np.complex128).flatten()
        vislen = self.get_num_baselines(self.Nside)
        self.vis_guess = np.random.normal(0, 1, (vislen, 2)).view(dtype=np.complex128).flatten()

    def run_min(self, itermax):
        self.make_guesses()
        garr, varr, chiscores, _ = self.chimincal(itermax, self.data[1], self.gain_guess, self.vis_guess, self.data[2], self.data[3], self.data[4])
        self.gain_guess = garr
        self.vis_guess = varr
        self.chiscores = chiscores
        return garr, varr, chiscores

    def get_num_baselines(self, Nside):
        # For square of N telescopes, should be 2N-2*sqrt(N) + 1
        # Fix to a corner and there are N-1 options for unique baselines. 
        # Flip over to other side (over x or y axis) and get another N-1 options
        # Duplicated Q are the pure x-axis and pure y-axis so -2*(sqrt(N)-1)
        # Final +1 is from 0 baseline

        N_bases = 2*Nside**2 - 2*Nside
        return int(N_bases)

    rand_phases = lambda self, x: np.random.uniform(0, 2*np.pi, x)
    zero_weight = lambda self, x, d: x/d if d else 0
    vector_weight = lambda self, x, d: np.divide(x, d, out=np.zeros_like(x, dtype=np.float), where=(d!=0))

    def make_gains(self, Nside):
        # Create complex gains with either (amplitude, phase) or (real, imaginary)
        Nant = Nside**2
        gain_amp = np.random.normal(1, .05, Nant)
        gain_phase = self.rand_phases(Nant)
        tgain = gain_amp*np.exp(1j*gain_phase)    
        return tgain

    def make_data(self, Nside, gains, noise=0.1):
        Nant = Nside**2
        Nbase = self.get_num_baselines(Nside)
        vis_true = np.random.normal(0,1,size=(Nbase,2)).view(np.complex128).flatten() ## size of unique baselines
        ant_i, ant_j, visndx, data = [], [], [], []
        ndx=0
        ndx2base={}
        base2ndx={}
        for i in range(Nant):
            xi,yi=np.unravel_index(i,(Nside,Nside))
            for j in range (i+1,Nant):
                xj,yj=np.unravel_index(j,(Nside,Nside))
                assert (xj>=xi)
                baseline = (xj-xi,yj-yi)
                if baseline in base2ndx:
                    cndx = base2ndx[baseline]
                else:
                    cndx = ndx
                    base2ndx[baseline]=ndx
                    ndx2base[ndx]=baseline
                    ndx+=1
                ant_i.append(i)
                ant_j.append(j)
                visndx.append(cndx)
                data.append(vis_true[cndx]*gains[i]*np.conj(gains[j]))

        assert(ndx==Nbase)
        ant_i = np.array(ant_i)
        ant_j = np.array(ant_j)
        visndx = np.array(visndx)
        data = np.array(data)
        noise = np.random.normal(0,noise,size=(len(data),2)).view(np.complex128).flatten() ## size of unique baselines
        data += noise
        return vis_true, data, ant_i, ant_j, visndx, ndx2base, base2ndx

    def make_pred(self, gains, vis, ant_i, ant_j, visndx):
        gains_i = gains[ant_i]
        cgains_j = np.conj(gains[ant_j])
        pred = gains_i*cgains_j*vis[visndx]
        return pred

    def chi2(self, data, gains, vis, ant_i, ant_j, visndx, noise=0.1):
        pred = self.make_pred(gains, vis, ant_i, ant_j, visndx)
        chi2 = np.abs((data - pred)**2).sum()/(noise**2)
        dof = len(data)*2
        return chi2, dof
    
    def chimincal(self, iter_max, data, g0, v0, ant_i, ant_j, visndx, noise=0.1, delta=0.4, epsilon=1e-5):
        chiscores = []
        garr = g0.copy()
        varr = v0.copy()
        N = 1
        Nant = len(g0)
        Nbase = len(v0)

        chi, dof = self.chi2(data, garr, varr, ant_i, ant_j, visndx)
        chiscores.append(chi/dof)


        for n in range(iter_max):
            gprime = np.zeros(Nant, dtype=np.complex128)
            for i in range(Nant):
                numer = np.complex(0, 0)
                denom = np.complex(0, 0)
                iant = np.where(ant_i == i)[0]
                jant = np.where(ant_j == i)[0]
                numer += (data[iant] * garr[ant_j[iant]]*np.conj(varr[visndx[iant]])).sum()
                denom += (np.abs(garr[ant_j[iant]]*varr[visndx[iant]])**2).sum()
                numer += (np.conj(data[jant]) * garr[ant_i[jant]]*varr[visndx[jant]]).sum()
                denom += (np.abs(garr[ant_i[jant]]*varr[visndx[jant]])**2).sum()
                gprime[i] += self.zero_weight(numer, denom)

            garr = (1 - delta)*garr + delta*gprime

            vprime = np.zeros(Nbase, dtype=np.complex128)
            for i in range(Nbase):
                indxs = np.where(visndx==i)
                numer = (data[indxs]*np.conj(garr[ant_i[indxs]])*garr[ant_j[indxs]]).sum()
                denom = (np.abs(garr[ant_i[indxs]]*np.conj(garr[ant_j[indxs]]))**2).sum()
                vprime[i] = self.zero_weight(numer, denom)

            varr = (1-delta)*varr + delta*vprime

            gscale = N/np.mean(np.abs(garr))
            garr *= gscale
            varr *= (1/gscale)**2

            chi, dof = self.chi2(data, garr, varr, ant_i, ant_j, visndx)
            chiscores.append(chi/dof)
            if n > 20:
                if np.abs(chiscores[-2] - chiscores[-1]) < epsilon:
                    break
        return garr, varr, chiscores, n
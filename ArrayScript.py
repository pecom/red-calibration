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
from BasicScript import BasicArray

class RealArray:

    rand_phases = lambda self, x: np.random.uniform(0, 2*np.pi, x)
    zero_weight = lambda self, x, d: x/d if d else 0
    vector_weight = lambda self, x, d: np.divide(x, d, out=np.zeros_like(x, dtype=np.float), where=(d!=0))
    
    def get_dof(self, Nside, outbeam):
        dof = (Nside**4 - Nside**2) - 2*(2*Nside*3*(2*Nside + Nside - 1) + (outbeam*Nside)**2)
        return dof

    def get_num_baselines(self, Nside):
        N_bases = 2*Nside**2 - 2*Nside
        return int(N_bases)
    
    def get_circle_array(self, rad, Nspacing, tap, gpos):
        x=np.outer(np.linspace(-0.5,+0.5,Nspacing),np.ones(Nspacing))
        y=x.T
        t=np.linspace(0,2.0,100)
        shapefun = lambda t:1-1/(1+np.exp(-2*(t-rad)/tap))
        beam = shapefun(np.sqrt((x - gpos[0])**2+(y - gpos[1])**2))
        return beam

    def get_weighted_array(self, alpha, Nspacing, numdraws=1e5):
        # Overlap a circular beam onto a pixelized grid 
        Nbeam = Nspacing**2
        rmax = alpha*.5
        ndraws = int(numdraws)
        spacing = np.linspace(0,1,Nspacing+1)
        centered_spacing = spacing - .5
        empty_weight_beam = np.zeros((Nspacing, Nspacing), dtype=np.complex128)

        for i in range(Nbeam):
            xi, yi = np.unravel_index(i, (Nspacing, Nspacing))
            draws = np.array([np.random.uniform(centered_spacing[xi], centered_spacing[xi+1], ndraws), np.random.uniform(centered_spacing[yi], centered_spacing[yi+1], ndraws)])
            dist = np.linalg.norm(draws, axis=0)
            empty_weight_beam[xi, yi] = np.sum(dist < rmax)/ndraws
        return empty_weight_beam

    # Helper function to get a (2*br-1, 2*br-1) grid around point x
    def n_mesh(self, br):
        nmesh = lambda x: tuple(np.meshgrid(range(x[0]-br, x[0]+br+1), range(x[1]-br, x[1]+br+1), indexing='ij'))
        return nmesh

    def make_uv_grid(self, Nside):
        uv_size = Nside*2 - 1
        center = (Nside-1,Nside-1)
        npcenter = np.array(center)
        img_size = (uv_size, uv_size)
        random_image = np.random.normal(0, 1, img_size)
        centered_uv = np.fft.fftshift(np.fft.fft2(random_image))
        topleft_uv = np.fft.ifftshift(centered_uv)

        return centered_uv, topleft_uv, npcenter

    def get_new_shape(self, Nside, beam_radius=3):
        # Helper function to get shape of convolved u-v space
        orig_shape = (2*Nside-1, Nside)
        new_shape = tuple([i*beam_radius + (beam_radius - 1) for i in orig_shape])
        return new_shape

    def make_visibilities(self, Nside, beam_radius, vis_var=1):
        n = Nside
        br = beam_radius

        # Useful shapes/sizes to have on hand
        new_shape = self.get_new_shape(n, br)
        new_size = np.prod(new_shape)
        center = (int((new_shape[0]-1)/2), br-1)

        #Matrices of visibiltiies and possible indices
        random_vis = np.random.normal(0,vis_var , (new_shape[0], new_shape[1], 2)).view(np.complex128)
        poss_index = np.arange(new_size).reshape(new_shape)

        #Clear off the leftmost columns to remove redundant conjugate stuff
        poss_index[0:center[0], 0:br] = 0
        poss_index[center[0]:, 0:br-1] = 0

        oversampled_baselines = poss_index.nonzero()[0].shape[0]
        visib = np.zeros(oversampled_baselines, dtype=np.complex128)
        #Baseline - index dictionaries
        new_ndx = 0
        n2b = {}
        b2n = {}
        n2true = {}
        true2n = {}
        for i in range(1, new_size):
            xi,yi = np.unravel_index(i, new_shape)
            if poss_index[xi,yi] == 0:
                continue
            else:
                baseline = (xi-center[0], yi-center[1])
                if baseline[0] < 0:
                    baseline = (-1*baseline[0], -1*baseline[1])

                if baseline in b2n:
                    cndx = new_ndx
                else:
                    cndx = new_ndx
                    b2n[baseline] = new_ndx
                    n2b[new_ndx] = baseline
                    if tuple(np.mod(baseline, br)) == (0,0):
                        modtuple = tuple(np.floor_divide(baseline, br))
                        true2n[modtuple] = new_ndx
                        n2true[new_ndx] = modtuple
                    new_ndx += 1
            visib[cndx] = random_vis[xi,yi]
        return n2b, b2n, n2true, true2n, visib


    def vis_to_grid(self, visib, ndx2base, size):
        # Turn our flat list of visibilities into a grid based on the dictionary
        # Meshes make properly yanking data from a grid straight-forward and also takes into account proper
        # negative indexing which makes it preferable to the flat visibility list.
        
        # Double the size to make sure to avoid rewriting values with negative indexing
        uv_size = (2*size[0], 2*size[1])
        new_grid = np.zeros(uv_size, dtype=np.complex128)
        for i,v in enumerate(visib):
            base = ndx2base[i]
            inv_base = tuple(-1*np.array(base))
            new_grid[base] = v
            new_grid[inv_base] = np.conj(v)
        return new_grid

    def create_fake_flatndx(self, Nside, n_beam):
        Nant = Nside**2
        conv_beamrad = 2*n_beam - 1
        beam_rad = int((n_beam-1)/2)+1
        n2b, b2n, n2t, t2n, _ = self.make_visibilities(Nside, n_beam)
        datandx = []
        ant_i = []
        ant_j = []
        
        if n_beam==1:
            mesh = self.n_mesh(0)
        else:
            mesh = self.n_mesh(beam_rad)
        
        for i in range(Nant):
            xi,yi = np.unravel_index(i, (Nside, Nside))
            for j in range(i+1, Nant):
                ant_i.append(i)
                ant_j.append(j)
                xj,yj = np.unravel_index(j, (Nside, Nside))
                assert (xj>=xi)
                baseline = (xj-xi,yj-yi)
                
                virtual_n = t2n[baseline]
                grid_base = n2b[virtual_n]
                
                virtual_points = mesh(grid_base)
                
                datandx.append(virtual_points)
        flat = self.vector_b2n(b2n, datandx)
        ant_i = np.array(ant_i)
        ant_j = np.array(ant_j)
        return flat, ant_i, ant_j

    def make_data_grid(self, Nside, gains, beams, noise=0.1, vis_var=1, verbose=False):
        
        Nant = Nside**2
        Nbase = self.get_num_baselines(Nside)
        # Get sample beam to understand shape of convolved beams
        samp_beam = signal.convolve(beams[0], beams[0])
        
        #Radius and corresponding mesh
        beam_radius = int((samp_beam.shape[0]-1)/2) + 1
        mesh = self.n_mesh(beam_radius-1)
        
        # Create visibilities
        n2b, b2n, n2t, t2n, visib = self.make_visibilities(Nside, beam_radius, vis_var)
        if verbose:
            print("Made visib")
        
        new_shape = self.get_new_shape(Nside, beam_radius)
        vis_grid = self.vis_to_grid(visib, n2b, new_shape)
        if verbose:
            print("Made grid")
        
        ant_i, ant_j, visndx, data, datandx = [], [], [], [], []
        
        for i in range(Nant):
            xi,yi=np.unravel_index(i,(Nside, Nside))
            for j in range (i+1,Nant):
                conv_beam = signal.convolve(beams[i], np.conj(beams[j][::-1, ::-1]))
                xj,yj=np.unravel_index(j,(Nside, Nside))
                assert (xj>=xi)
                baseline = (xj-xi,yj-yi)
                
                virtual_n = t2n[baseline]
                grid_base = n2b[virtual_n]
                
                virtual_points = mesh(grid_base)
                data_sum = (vis_grid[virtual_points] * gains[i] * np.conj(gains[j]) * conv_beam).sum()
                
                ant_i.append(i)
                ant_j.append(j)
                visndx.append(virtual_n)
                data.append(data_sum)
                datandx.append(virtual_points)
        
        if verbose:
            print("Created data")
        
        ant_i = np.array(ant_i)
        ant_j = np.array(ant_j)
        visndx = np.array(visndx)
        data = np.array(data)
        noise = np.random.normal(0,noise,size=(len(data),2)).view(np.complex128).flatten() ## size of unique baselines
        data += noise
        return visib, data, ant_i, ant_j, visndx, datandx, n2b, b2n, n2t, t2n, vis_grid

    def vector_b2n(self, b2n, datandx):
        # Convert datandx (points on the grid) to flat indices using b2n dictionary
        newndx = []
        for l in datandx:
            x,y = l
            flattened = []
            ndx_size = x.size
            ndx_shape = x.shape
            for i in range(ndx_size):
                point = np.unravel_index(i, ndx_shape)
                k = np.array([x[point], y[point]])
                key = tuple(k)
                if key in b2n:
                    flattened.append(b2n[key])
                else:
                    key = tuple(-1*k)
                    flattened.append(-1*b2n[key])
            newndx.append(flattened)
        newndx = np.array(newndx)
        return newndx


    def conjugate_visib(self, vis, ndxs):
        # Returns the proper list of relevant visibilites given ndxs
        # Negative refers to using a conjugate rather than doubling the size of the array
        flat = []
        for i in ndxs:
            if i >= 0:
                flat.append(vis[i])
            else:
                flat.append(np.conj(vis[-1*i]))
        return np.array(flat)

    def flat_model(self, vis, beams, gains, ant_i, ant_j, flatndx):
        # Make prediction
        postage = np.array([signal.convolve(beams[ant_i[i]], np.conjugate(beams[ant_j[i]][::-1, ::-1])).flatten() for i in range(len(flatndx))])
        gains_i = gains[ant_i]
        gains_j = np.conj(gains[ant_j])
        flatbread = np.array([self.conjugate_visib(vis, v)*postage[i] for i, v in enumerate(flatndx)])
        pred = np.sum(flatbread, axis=1)*gains_i*gains_j
        return pred

    def gen_chi2(self, data, pred, noise=0.1):
        # Generic chi2 definition
        chitwo = np.abs((data - pred)**2).sum()/(noise**2)
        dof = len(data)*2
        return chitwo, dof, chitwo/dof
    
    def vec_chi2(self, data, pred, nv):
        nr = nv.real
        ch2 = lambda d,p : np.sum(((p - d)**2)/(nr**2))
        chitwo = ch2(data.real, pred.real) + ch2(data.imag, pred.imag)
        dof = self.dof
        return chitwo, dof, chitwo/dof
    
#         chitwo = (np.abs((data-pred)**2)/(nv**2)).sum().real
#         dof = self.dof
#         return chitwo, dof, chitwo/dof

    def imag_to_reals(self, vec):
        # Helper function for linear estimator
        reals = vec.real
        imags = vec.imag
        flat_vec = np.concatenate((reals, imags))
        return flat_vec
    
    def reals_to_imag(self, vec):
        vlen = int(len(vec)/2)
        comp_vec = vec[:vlen] + 1j*vec[vlen:]
        return comp_vec
    
    def linear_solver_A(self, beams, gains, data, ant_i, ant_j, fndx):
        # Solve for visibilities using a linear method
        big_ans = self.imag_to_reals(data)
        data_len = len(data)
        v_size = len(set(np.abs(fndx).flatten())) + 1
        postage = np.array([signal.convolve(beams[ant_i[i]], np.conjugate(beams[ant_j[i]][::-1, ::-1])).flatten()*gains[ant_i[i]]*np.conjugate(gains[ant_j[i]]) for i in range(len(fndx))])
        bigA = sparse.lil_matrix((2*data_len, 2*v_size))
        for i,v in enumerate(fndx):
            absv = np.abs(v)
            bigA[i,absv] = postage[i].real
            bigA[i, v_size+absv] = -1*np.sign(v)*postage[i].imag
            bigA[i+data_len, absv] = postage[i].imag
            bigA[i+data_len, v_size+absv] = np.sign(v)*postage[i].real
        bigCSR = bigA.tocsr()
        return bigCSR, big_ans
    
    def vis_solv(self, vis, beams, gains, data, noise, ant_i, ant_j, fndx, fvis):
        bigA, bigB = self.linear_solver_A(beams, gains, data, ant_i, ant_j, fndx)
        At = bigA.T
        sinv = sparse.diags(np.ones(2*fvis))
        inv_noise = 1/noise
        doubled_nv = np.concatenate((inv_noise, inv_noise))
        ninv = sparse.diags(doubled_nv)
        lhs = (sinv + At@ninv@bigA)
        rhs = (At@ninv)@bigB
        wein_m  = self.reals_to_imag(sparse.linalg.spsolve(lhs, rhs))
        return wein_m

    def vis_old_variance(self, vis, beams, gains, data, noise, ant_i, ant_j, sigma_v, fndx):
        bigA, bigB = self.linear_solver_A(beams, gains, data, ant_i, ant_j, fndx)
        diagonal = 2*np.ones(bigA.shape[1])*sigma_v**2
        Sigma = sparse.diags(1/diagonal)
        newA = (bigA.T@bigA + Sigma)
        newB = bigA.T@bigB
        map_sol = sparse.linalg.spsolve(newA, newB)
        comb_sol = self.reals_to_imag(map_sol)
        return comb_sol

    def vis_minimal_variance(self, vis, beams, gains, data, noise, ant_i, ant_j, sigma_v, fndx):
        bigA, bigB = self.linear_solver_A(beams, gains, data, ant_i, ant_j, fndx)
        At = bigA.T
        sinv = sparse.diags(np.ones(bigA.shape[1])*sigma_v**2)
        inv_noise = 1/noise**2
        doubled_nv = np.concatenate((inv_noise, inv_noise))
        ninv = sparse.diags(doubled_nv)

        lhs = (sinv + At@ninv@bigA)
        rhs = (At@ninv)@bigB

        wein_m  = self.reals_to_imag(sparse.linalg.spsolve(lhs, rhs))
        return wein_m

    def old_vis_solver(self, guess, beams, gains, data, ant_i, ant_j, fndx):
        bigA, bigB = self.linear_solver_A(beams, gains, data, ant_i, ant_j, fndx)
        map_sol = sparse.linalg.lsmr(bigA, bigB)[0]
        comb_sol = self.reals_to_imag(map_sol)
        # v_size = len(guess)
        # comb_sol = map_sol[:v_size] + 1j*map_sol[v_size:]
        return comb_sol

    def padded_circulant(self, col):
        c_len = len(col)
        first_row = np.zeros(c_len, dtype=np.complex128)
        first_row[0] = col[0]
        first_col = np.zeros(2*c_len - 1, dtype=np.complex128)
        first_col[0:c_len] = col
        toep_mat = linalg.toeplitz(first_col, first_row)
        return toep_mat

    def generic_block_toep(self, beam_j):
        pad_zero = np.zeros(beam_j.shape[0], dtype=np.complex128)
        nb = beam_j.shape[0]
        finshape = 2*nb - 1
        block_matrix = []
        circ_ndx = linalg.circulant(np.arange(finshape))[:,0:nb]
        for i in range(finshape):
            if i < nb:
                block_matrix.append(self.padded_circulant(beam_j[i]))
            else:
                block_matrix.append(self.padded_circulant(pad_zero))
        blocked = np.array(block_matrix)
        unshape_circ = blocked[circ_ndx]
        shaped_circ = unshape_circ.transpose(0, 2, 1, 3).reshape(finshape**2, nb**2)
        return shaped_circ   

    def beam_solver(self, vis, beam_guess, gains, data, ant_i, ant_j, flatndx, Nside):
        Nant = Nside**2
        n_beam = beam_guess.shape[1]
        fn_bm = 2*n_beam - 1
        matrix_beams = np.zeros((beam_guess.shape[0], fn_bm**2, n_beam**2), dtype=np.complex128)
        
        symbeam = int((n_beam + 1)/2)
        symconv = signal.convolve(np.ones((symbeam, symbeam), dtype=np.complex128), np.ones((symbeam, symbeam), dtype=np.complex128)).flatten()
        center_ofsymbeam = int((n_beam**2 - 1)/2)
        sym_hyper = 50

        for i,v in enumerate(beam_guess):
            matrix_beams[i] = self.generic_block_toep(v)
        
        matrix_beams = np.array(matrix_beams)
        
        new_beams = beam_guess.copy()
        for ant_ndx in range(Nant):
            beam_solver = np.zeros((Nant-1, n_beam**2), dtype=np.complex128)
            rhs_vis = np.zeros(Nant-1, dtype=np.complex128)
            
            ant_filter = ant_i==ant_ndx
            sum_ants = np.sum(ant_filter)
            jant_filter = ant_j==ant_ndx
            
            rhs_vis[:sum_ants] = data[ant_filter]
            rhs_vis[sum_ants:] = np.conjugate(data[jant_filter])                    
            
            for i in range(sum_ants):
                beam_solver[i] = self.conjugate_visib(vis, flatndx[ant_filter][i])[None,:] @ np.conjugate(matrix_beams[ant_j[ant_filter]][i][::-1, ::-1])*gains[ant_i[ant_filter]][i]*np.conjugate(gains[ant_j[ant_filter]][i])
            for j in range(np.sum(jant_filter)):
                beam_solver[sum_ants + j] = (np.conjugate(self.conjugate_visib(vis, flatndx[jant_filter][j])[None,:]) @ np.conjugate(matrix_beams[ant_i[jant_filter]][j]))[::-1,::-1]*gains[ant_j[jant_filter]][j]*np.conjugate(gains[ant_i[jant_filter]][j])

            zerobeam = optimize.lsq_linear(beam_solver, rhs_vis).x
            shaped_beam = zerobeam.reshape((n_beam, n_beam))
            matrix_beams[ant_ndx] = self.generic_block_toep(shaped_beam)
            new_beams[ant_ndx] = shaped_beam
        return new_beams

    def solve_everything(self, iter_max, vis_guess, beam_guess, gains, data, ant_i, ant_j, flatndx, Nside, noise, chi_eps=1, score_stop=.5, wien=True, verbose=True, alpha=1):
        chis = []
        scores = []
        model = self.flat_model(vis_guess, beam_guess, gains, ant_i, ant_j, flatndx)
        chi, _, score = self.vec_chi2(data, model, noise)
        chis.append(chi)
        scores.append(score)
        fvis = len(vis_guess)
        
        n = 0
    
        if iter_max != 0:
            counter = iter_max
        else:
            counter = 10

        while counter >= 0:      
            
            if wien:
                new_vis = self.vis_solv(vis_guess, beam_guess, gains, data, noise, ant_i, ant_j, flatndx, fvis)
            else:
                new_vis = self.old_vis_solver(vis_guess, beam_guess, gains, data, ant_i, ant_j, flatndx)
            
            dvis = new_vis - vis_guess
            vis_guess = vis_guess + alpha*dvis
#             vis_guess = new_vis
            
            # new_beams = self.beam_solver(vis_guess, beam_guess, gains, data, ant_i, ant_j, flatndx, Nside)
            # dbeam = new_beams - beam_guess
            # beam_guess = beam_guess + alpha*dbeam
#           #   beam_guess = new_beams
            
            model = self.flat_model(vis_guess, beam_guess, gains, ant_i, ant_j, flatndx)
            chi, _, score = self.vec_chi2(data, model, noise)
            
            chis.append(chi)
            scores.append(score)
            if (np.abs(chi - chis[-2]) < chi_eps):
                break
            if score < score_stop:
                break
            if verbose:
                if n%100==0:
                    print(n, score)
            if iter_max != 0:
                counter -= 1
            n += 1
            
        chis = np.array(chis)
        scores = np.array(scores)
        print("Final Iteration:", n, score)
        
        return vis_guess, beam_guess, chis, scores

    def scale_phased(self, vis, beams, beam_scale, phase, n2b, n_beam, Nside):
        term_phase = np.array([np.array(n2b[n])/n_beam for n in np.arange(len(n2b))])
        beam_ndx = np.array(np.unravel_index(np.arange(len(beams)), (Nside, Nside)))
        phase_dphi = phase
        
        phase_vis = vis*np.exp(1j*np.dot(term_phase, phase_dphi))

        beam_r = int((n_beam-1)/2)
        beam_mesh = self.n_mesh(beam_r)
        
        phase_beams = []
        for i in range(len(beam_ndx[0])):
            pos_beamy, pos_beamx = np.array(beam_mesh(beam_ndx[:,i]*n_beam))/n_beam
            beam_flipped = np.array([pos_beamy[::-1], pos_beamx[:,::-1]])
            offsets = np.sum(beam_flipped*phase_dphi[:,None,None], axis=0)
            phase_vec = np.exp(1j*offsets)
            phase_beams.append(beams[i]*phase_vec)
        phase_beams = np.array(phase_beams)
        
        fin_beams = phase_beams * beam_scale
        fin_vis = phase_vis * (1/beam_scale)**2
        return fin_vis, fin_beams

    def guess_rms(self, beams, n=100, imax=100, vis_var=1):
        output_m = beams[0].shape[0]
        blen = len(beams)
        fin_size = (2*output_m - 1)**2
        variances = []
        for p in range(n):
            dats = []
            for q in range(imax):
                i,j = np.random.choice(blen, 2)
                post = signal.convolve(beams[i], np.conjugate(beams[j][::-1, ::-1])).flatten()
                fake_data = np.dot(post, np.random.normal(0, vis_var, (fin_size, 2)).view(dtype=np.complex128).flatten())
                dats.append(fake_data)
            dats = np.array(dats)
            variances.append(np.var(dats))
        variances = np.array(variances)
        return np.sqrt(np.mean(variances))

    def __init__(self, Nside, n_beam):
        self.Nside = Nside
        self.Nant = Nside**2
        self.Nbase = self.get_num_baselines(Nside)
        self.n_beam = n_beam
        self.dof = (Nside**2)*(Nside**2 - 1)
        # n_beam = 2*M + 1
        self.gains = np.ones(self.Nant)
        self.gpos = np.zeros((self.Nant, 2), dtype=np.complex128)
        self.point_err = np.ones(self.Nant, dtype=np.complex128)
    
    def create_beams(self, rad=.4, s_fac=1, v_var=1):
        beam_comp_phase = np.exp(1j*self.rand_phases(self.Nant))
        self.overall_gains = beam_comp_phase
        new_beams = np.array([self.get_circle_array(rad, self.n_beam, .05, self.gpos[i])*beam_comp_phase[i]*self.point_err[i]*s_fac for i in range(self.Nant)])
        self.beams = new_beams
        fake_beams = np.array([self.get_circle_array(rad, self.n_beam, .05, [0,0]) * v_var for i in range(self.Nant)])
        self.fake_beams = fake_beams
        rms = self.guess_rms(fake_beams, 100, 75, 1)
        self.rms = rms
        
    def set_beams(self, beams):
        self.beams = beams
    
    def set_chin(self, chin):
        self.chin = chin
    
    def set_data(self, dat):
        self.data = dat

    def geometry_error(self, geom_mag):
        self.gpos = np.random.normal(0, geom_mag, (self.Nant, 2))
        
    def pointing_error(self, phase_mag):
        px = np.random.normal(0, phase_mag, self.Nant)
        py = np.random.normal(0, phase_mag, self.Nant)

        x = np.outer(np.linspace(-0.5,0.5,self.n_beam),np.ones(self.n_beam))
        y = x.T
        
        self.px = px
        self.py = py
        phase_fac = np.zeros((self.Nant, self.n_beam, self.n_beam), dtype=np.complex128)
        for i in range(self.Nant):
            phase_fac[i] = np.exp(1j*((x)*px[i]+(y)*py[i]))
        self.point_err = phase_fac

    def camera_error(self, eps=1e-5):
        self.beams = self.beams + eps*np.random.random((*self.beams.shape, 2)).view(dtype=np.complex128).reshape(self.beams.shape)
        
    def errorless_data(self, v_var=1):
        vistrue, data, ant_i, ant_j, visndx, _, _, _, _, _, _ = self.make_data_grid(self.Nside, self.gains, self.beams, noise=0, vis_var=v_var)
        data_len = len(data)
        self.errorless = data
        self.ant_i = ant_i
        self.ant_j = ant_j
        self.data_len = data_len
        self.visndx = visndx
        self.vistrue = vistrue
        
    def base_noise(self):
        chin = np.ones(self.data_len)*self.rms
        # nvec = np.array([np.random.normal(0, c, 2).view(np.complex128)[0] for c in chin])
        nvec = np.random.normal(0, self.rms, (self.data_len, 2)).view(complex).flatten()
        self.bchin = chin
        self.bn = nvec

    def add_noise(self, snr):
        self.snr = snr
        nvec = self.bn/snr
        self.chin = self.bchin/snr
        self.data = self.errorless + nvec
        
    def compare_to_red(self, gain_guess=None):
        barray = BasicArray(10)
        if gain_guess is not None:
            beam_gain_guess = gain_guess
        else:
            beam_gain_guess = np.random.normal(0, 1, (self.Nant, 2)).view(np.complex128).flatten()
        vis_guess = np.random.normal(0, 1, (self.fakevislen, 2)).view(np.complex128).flatten()
        chig, chiv, c, n = barray.chimincal(400, self.data, beam_gain_guess, vis_guess, self.fanti, self.fantj, self.mid_flat, delta=.4, noise=self.chin)
        print("Red cal: ", c[-1])
        self.red_chig = chig
        self.red_chiv = chiv
        self.red_chis = c

    def create_fit(self, outbeam, nmax=100, wien=True, bg=None, ib=None, gg=None, alph=1):
        bshape = (self.Nant, outbeam, outbeam)
        fakeflat, ant_i, ant_j = self.create_fake_flatndx(self.Nside, outbeam)
        fakevislen = len(set(np.abs(fakeflat).flatten()))+1
        self.fakevislen = fakevislen
        self.fanti = ant_i
        self.fantj = ant_j
        
        dvec_len = len(fakeflat[0])
        mid_ndx = int((dvec_len-1)/2)
        self.mid_flat = fakeflat[:,mid_ndx]

#         print('Guessing visibility')
        if bg is not None:
            self.bad_guess = bg
        else:
            bg = np.random.normal(0, 1, (fakevislen, 2)).view(np.complex128).flatten()
            self.bad_guess = bg
        
#         print('Guessing beams')
        if ib is not None:
            self.improv_beam = ib
        else:
            ib = np.random.normal(0, 1, (*bshape, 2)).view(np.complex128).reshape(bshape)
            self.improv_beam = ib
        
#         print('Guessing gains')
        if gg is not None:
            self.guess_gains = gg
        else:
            gg = np.random.normal(0, 1, (self.Nant, 2)).view(complex).flatten()
            self.guess_gains = gg

        fit_params = fakevislen - 1 + len(self.improv_beam.flatten())
        self.dof = self.Nside**4 - self.Nside**2 - 2*fit_params
        iscore = self.flat_model(self.bad_guess, self.improv_beam, self.guess_gains, ant_i, ant_j, fakeflat)
        print("Original guess, ", self.vec_chi2(self.data, iscore, self.chin))

        self.itersolve = self.solve_everything(nmax, self.bad_guess, self.improv_beam, self.guess_gains, self.data, ant_i, ant_j, fakeflat, self.Nside, self.chin, wien=wien, alpha=alph)
            
    def bin_xs(self, bigM, outsize=3):
        if outsize==3:
            modM = bigM%3
            if modM%2==0:
                central_size = bigM//3
            else:
                central_size = bigM//3+1

            leftovers = (bigM-central_size)//2
            xs = [0]*leftovers + [1]*central_size + [2]*leftovers
            xs = np.array(xs)
        else:
            xs = np.floor(np.linspace(0, outsize, bigM, endpoint=False)).astype(np.int)

        return xs
    
    
    
    
    def bin_beams(self, beams, outsize=3):
        totsize = beams.shape[0]
        bigM = beams.shape[-1]
        xs = self.bin_xs(bigM, outsize)
        binned_beam = np.zeros((totsize, outsize, outsize), dtype=np.complex)
        bin_counts = np.zeros_like(binned_beam)
        for n in range(totsize):
            for i in range(bigM):
                for j in range(bigM):
                    ibin = xs[i]
                    jbin = xs[j]
                    binned_beam[n, ibin, jbin] += beams[n, i,j]
                    bin_counts[n, ibin, jbin] += 1
        final_beams = binned_beam/bin_counts
        return final_beams
    
    def gaussian_chi2(self, data, noise, vis, beams, gains, priors, sigma, ant_i, ant_j, flatndx):
        # Make prediction
        Nant = len(beams)
        beam_shapesize = np.prod(beams[0].shape)
        dof = self.get_dof(int(np.sqrt(Nant)), int(np.sqrt(beam_shapesize)))
        prediction = self.flat_model(vis, beams, gains, ant_i, ant_j, flatndx)
        chi2, _, _ = self.vec_chi2(prediction, data, noise)
        focaccia = []
        for i, b in enumerate(beams):
            focaccia.append((np.abs(b - priors[i])**2)/sigma**2)
        prior_term = np.array(focaccia)
        totchi2 = chi2 + np.sum(prior_term) + Nant*beam_shapesize*np.log(sigma)
        return totchi2
    
    def gaussian_beam_solver4(self, vis, beam_guess, beam_priors, gains, sigma, data, noise, ant_i, ant_j, flatndx, Nside):
        Nant = Nside**2
        n_beam = beam_guess.shape[1]
        fn_bm = 2*n_beam - 1
        ns = noise[0]
        total_beamsize = n_beam**2
        matrix_beams = np.zeros((beam_guess.shape[0], fn_bm**2, total_beamsize), dtype=complex)

        beam_count = (Nant-1)

        for i,v in enumerate(beam_guess):
            matrix_beams[i] = self.generic_block_toep(v) * gains[i]

        matrix_beams = np.array(matrix_beams)

        new_beams = beam_guess.copy()
        beam_solver = np.zeros((beam_count+total_beamsize, total_beamsize), dtype=complex)
        rhs_vis = np.zeros(beam_count + total_beamsize, dtype=complex)

        for ant_ndx in range(Nant):

            Bi = beam_guess[ant_ndx].flatten()
            Bp = beam_priors[ant_ndx].flatten()

            ant_filter = ant_i==ant_ndx
            sum_ants = np.sum(ant_filter)
            jant_filter = ant_j==ant_ndx

            rhs_vis[:sum_ants] = data[ant_filter]/ns
            rhs_vis[sum_ants:-total_beamsize] = np.conjugate(data[jant_filter])/ns

            for i in range(sum_ants):
                beam_solver[i] = (self.conjugate_visib(vis, flatndx[ant_filter][i])[None,:] \
                    @ np.conjugate(matrix_beams[ant_j[ant_filter]][i][::-1, ::-1]))/ns
            for j in range(np.sum(jant_filter)):
                beam_solver[sum_ants + j] = ((np.conjugate(self.conjugate_visib(vis, flatndx[jant_filter][j])[None,:]) \
                    @ np.conjugate(matrix_beams[ant_i[jant_filter]][j]))[::-1,::-1])/ns
                
            # Adding Gaussian prior to system
            rhs_vis[-total_beamsize:] = Bp/sigma
    #         beam_solver[-total_beamsize:] = np.identity(total_beamsize)*np.exp(-1j*np.angle(beam_priors[ant_ndx][1,1]))
            beam_center_gain = beam_guess[ant_ndx][1,1]
            beam_solver[-total_beamsize:] = np.identity(total_beamsize)*np.conj(beam_center_gain)/sigma

            zerobeam = optimize.lsq_linear(beam_solver, rhs_vis).x
            shaped_beam = zerobeam.reshape((n_beam, n_beam))
            matrix_beams[ant_ndx] = self.generic_block_toep(shaped_beam)
            new_beams[ant_ndx] = shaped_beam
        return new_beams
    
    def gaussian_beam_solver5(self, vis, beam_guess, beam_priors, gains, sigma, data, noise, ant_i, ant_j, flatndx, Nside):
        Nant = Nside**2
        n_beam = beam_guess.shape[1]
        fn_bm = 2*n_beam - 1
        ns = noise[0]
        total_beamsize = n_beam**2
        matrix_beams = np.zeros((beam_guess.shape[0], fn_bm**2, total_beamsize), dtype=complex)

        beam_count = (Nant-1)

        for i,v in enumerate(beam_guess):
            matrix_beams[i] = self.generic_block_toep(v) 

        matrix_beams = np.array(matrix_beams)

        new_beams = beam_guess.copy()
        beam_solver = np.zeros((beam_count+total_beamsize, total_beamsize), dtype=complex)
        rhs_vis = np.zeros(beam_count + total_beamsize, dtype=complex)
        c_pix = (total_beamsize-1)//2

        for ant_ndx in range(Nant):

            Bi = beam_guess[ant_ndx].flatten()
            Bp = beam_priors[ant_ndx].flatten()

            ant_filter = ant_i==ant_ndx
            sum_ants = np.sum(ant_filter)
            jant_filter = ant_j==ant_ndx

            rhs_vis[:sum_ants] = data[ant_filter]/ns
            rhs_vis[sum_ants:-total_beamsize] = np.conjugate(data[jant_filter])/ns

            for i in range(sum_ants):
                beam_solver[i] = (self.conjugate_visib(vis, flatndx[ant_filter][i])[None,:] \
                    @ np.conjugate(matrix_beams[ant_j[ant_filter]][i][::-1, ::-1]) \
                    * gains[ant_i[ant_filter]][i]*np.conjugate(gains[ant_j[ant_filter]][i]))/ns
            for j in range(np.sum(jant_filter)):
                beam_solver[sum_ants + j] = ((np.conjugate(self.conjugate_visib(vis, flatndx[jant_filter][j])[None,:]) \
                    @ np.conjugate(matrix_beams[ant_i[jant_filter]][j]))[::-1,::-1] \
                    * gains[ant_j[jant_filter]][j]*np.conjugate(gains[ant_i[jant_filter]][j]))/ns
                
            # Adding Gaussian prior to system
            rhs_vis[-total_beamsize:] = Bp/sigma
    #         beam_solver[-total_beamsize:] = np.identity(total_beamsize)*np.exp(-1j*np.angle(beam_priors[ant_ndx][1,1]))
            beam_solver[-total_beamsize:] = np.identity(total_beamsize)/sigma

            no_cent = np.hstack((beam_solver[:,:c_pix], beam_solver[:,c_pix+1:]))
            cent = beam_solver[:,c_pix]
            mod_rhs = rhs_vis - cent

            zerobeam = optimize.lsq_linear(no_cent, mod_rhs).x
            beam_withcent = np.insert(zerobeam, c_pix, 1)
            shaped_beam = beam_withcent.reshape((n_beam, n_beam))
            matrix_beams[ant_ndx] = self.generic_block_toep(shaped_beam)
            new_beams[ant_ndx] = shaped_beam
        return new_beams
    
    def gaussian_sigma_iter(self, beam_guess, beam_prior):
        bg = beam_guess.flatten()
        bp = beam_prior.flatten()
        newsig = 2*np.std((np.abs(bg) - np.abs(bp)))
        return newsig
    
    def gaussian_sologains(self, vis, beams, garr, flatndx, data, Nside, ant_i, ant_j, delta=.4):
        Nant = Nside**2
        nogain = np.ones_like(garr)
        gainless = self.flat_model(vis, beams, nogain, ant_i, ant_j, flatndx)
        gprime = np.zeros(Nant, dtype=complex)
        for i in range(Nant):
            numer = complex(0, 0)
            denom = complex(0, 0)
            iant = np.where(ant_i == i)[0]
            jant = np.where(ant_j == i)[0]
            numer += (data[iant] * garr[ant_j[iant]]*np.conj(gainless[iant])).sum()
            denom += (np.abs(garr[ant_j[iant]]*gainless[iant])**2).sum()
            numer += (np.conj(data[jant]) * garr[ant_i[jant]]*gainless[jant]).sum()
            denom += (np.abs(garr[ant_i[jant]]*gainless[jant])**2).sum()
            gprime[i] += self.zero_weight(numer, denom)

        fg = (1 - delta)*garr + delta*gprime
        return fg
    
    def gaussian_solve_everything(self, iter_max, vis_guess, beam_guess, gains_guess, bprior, sigma_b, sigma_v, data, ant_i, ant_j, flatndx, Nside, noise, chi_eps=1, score_stop=.8, wiener=True, alpha=1):
        chis = []
        scores = []
        sigmas = []
        Nant = Nside**2
        dof = Nside**4 - Nside**2 - 2*(len(vis_guess.flatten())-1)
        model = self.flat_model(vis_guess, beam_guess, gains_guess, ant_i, ant_j, flatndx)
        chi, _, score = self.vec_chi2(data, model, noise )
        print("Before anything:", chi)
        chis.append(chi)
        scores.append(score)
        sigmas.append(sigma_b)
        fvis = len(vis_guess)
        n = 0
        dchi = 1000

        if iter_max != 0:
            counter = iter_max
        else: 
            counter = 10

        while counter > 0:

            if wiener:
                new_vis = self.vis_minimal_variance(vis_guess, beam_guess, gains_guess, data, noise, ant_i, ant_j, sigma_v, flatndx)
                # new_vis = self.vis_solv(vis_guess, beam_guess, gains_guess, data, noise, ant_i, ant_j, flatndx, fvis)
            else:
                new_vis = self.old_vis_solver(vis_guess, beam_guess, gains_guess, data, ant_i, ant_j, flatndx)

            dvis = new_vis - vis_guess
            vis_guess = vis_guess + alpha*dvis
            # vis_guess = new_vis

            new_beams = self.gaussian_beam_solver5(vis_guess, beam_guess, bprior, gains_guess, sigma_b,
                                                          data, noise, ant_i, ant_j, flatndx, Nside)
            dbeam = new_beams - beam_guess
            beam_guess = beam_guess + alpha*dbeam
            # beam_guess = new_beams
            # 
            new_gains = self.gaussian_sologains(vis_guess, beam_guess, gains_guess, flatndx, data, Nside, ant_i, ant_j, delta=alpha)
            dgains = new_gains - gains_guess
            gains_guess = gains_guess + alpha*dgains
            # # 
            sigma_b = self.gaussian_sigma_iter(beam_guess, bprior)

            # chi  = self.gaussian_chi2(data, noise, vis_guess, beam_guess, gains, bprior, sigma, ant_i, ant_j, flatndx)
            # score = chi/dof
            
            model = self.flat_model(vis_guess, beam_guess, gains_guess, ant_i, ant_j, flatndx)
            chi, _, score = self.vec_chi2(data, model, noise )
            
            chis.append(chi)
            scores.append(score)
            sigmas.append(sigma_b)

            dchi = np.abs(chi - chis[-2])

            if n%5==0:
                print(n, chi, score, sigma_b)
            if (dchi < chi_eps) and (n > 1):
                break
            if iter_max != 0:
                counter -= 1
            n += 1

        chis = np.array(chis)
        scores = np.array(scores)
        sigmas = np.array(sigmas)
        print(n, score, chi, sigma_b)

        return vis_guess, beam_guess, gains_guess, chis, scores, sigmas
    
    
    def gauss_fit(self, outbeam, prior, nmax=100, wien=True, bg=None, ib=None, gg=None, alph=1):
        bshape = (self.Nant, outbeam, outbeam)
        fakeflat, ant_i, ant_j = self.create_fake_flatndx(self.Nside, outbeam)
        fakevislen = len(set(np.abs(fakeflat).flatten()))+1
        self.fakevislen = fakevislen
        self.fanti = ant_i
        self.fantj = ant_j
        
        dvec_len = len(fakeflat[0])
        mid_ndx = int((dvec_len-1)/2)
        self.mid_flat = fakeflat[:,mid_ndx]

#         print('Guessing visibility')
        if bg is not None:
            self.bad_guess = bg
        else:
            bg = np.random.normal(0, 1, (fakevislen, 2)).view(np.complex128).flatten()
            self.bad_guess = bg
        
#         print('Guessing beams')
        if ib is not None:
            self.improv_beam = ib
        else:
            ib = np.random.normal(0, 1, (*bshape, 2)).view(np.complex128).reshape(bshape)
            self.improv_beam = ib
        
#         print('Guessing gains')
        if gg is not None:
            self.guess_gains = gg
        else:
            gg = np.random.normal(0, 1, (self.Nant, 2)).view(complex).flatten()
            self.guess_gains = gg


        fit_params = fakevislen - 1 + len(self.improv_beam.flatten())
        self.dof = self.Nside**4 - self.Nside**2 - 2*fit_params
        iscore = self.flat_model(self.bad_guess, self.improv_beam, self.guess_gains, ant_i, ant_j, fakeflat)
        print("Original guess, ", self.vec_chi2(self.data, iscore, self.chin))
        origstd = np.std((self.improv_beam.flatten()-prior.flatten()))

        self.gauss_itersolve = self.gaussian_solve_everything(nmax, self.bad_guess, self.improv_beam, self.guess_gains, prior, 1, 1, self.data, ant_i, ant_j, fakeflat, self.Nside, self.chin, wiener=wien, alpha=alph)

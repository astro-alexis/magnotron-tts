#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import os
import pandas as pd
import glob as glob
import h5py
import sys
import matplotlib.pyplot as plt
from magnotron_commons import *
import pickle
import corner
import emcee
# MCMC functions such as log-prior, log-likelihood, log posterior, and model
# are described here
#-------------------
# log prior distribution
#-----------------------
def ln_prior(ff,scale):
    # Physically the filling factor must be between 0 and 100%
    # I choose a uniform prior within this range
    # Also the sum of the filling factogc.set_debug(gc.DEBUG_STATS)r cannot be more than 100%
    prior_ff = 0.
    ff_total = 0.
    nff = len(ff)
    for i in range(nff):
        if ff[i] <0. or ff[i] > 100.:
            prior_ff += -100000.
        else:
            prior_ff += np.log(1./100.)
        ff_total += ff[i]
    if ff_total > 100:
        prior_ff += -100000.

    if scale < 0. or scale > 2.:
        prior_scale = -100000.
    else:
        prior_scale = np.log(1./2.)
    output = prior_ff + prior_scale
    return output

# Definition of the model for Y
def ymod(ff,scale,sytmp):
    output = (100.-np.sum(ff))*sytmp[0,:]
    nff = len(ff)
    for i in range(nff):
        output = output + (ff[i] * sytmp[i+1,:])
    output = 0.01 * (output) * scale +1. - scale
    return(output)

# Definition of the log-likelihood. The uncertainties are assumed independant,
# BIG WARNING SIGN: THE MEASUREMENT ERROR YERR IS ASSUMED CONSTANT ACROSS THE SPECTRUM
def lnlike(y,yerr,model_y):
    n = len(y)
    value = -1. * np.sum(((y-model_y)**2.)/(2.*yerr**2.))
    value = value - 0.5 * n * np.log(2.0 * np.pi * yerr[0])
    return value

def chideux(y,yerr,model_y):
    n = len(y)
    value = np.sum(((y-model_y)**2.)/(yerr**2.))
    return value,n

# Definition of the log-posterior
def lnprob(parameters,y,yerr,sytmp):
    ff,scale = parameters[0:-1], parameters[-1]
    return lnlike(y,yerr,ymod(ff,scale,sytmp)) + ln_prior(ff,scale)

# The hard-coded variables should go here:
# --------------------------------------------
plt.ioff()
mf=[0.0,2.0,4.0,6.0,8.0,10.0,12.0,14.0] # magnetic field strengths to be considered
nmf = len(mf)
synthpath = '../../data/synthetic-spectra/' # path to the synthetic spectra generated with Synmast
csvlink = 'https://docs.google.com/spreadsheets/d/1bjMaMhXWs4MK2BCSGerF_NB0rqY2kqwetAtBUF_4feQ/export?format=csv' # link to the CSV input file on Google Sheets
#file_prefix = "naIcor_loggf-7-5_"
#file_prefix = "naI_loggf-7-5_"
#file_prefix = "naI_loggf-7-5_fastconv_"
file_prefix = ""
# MCMC-related variables
#-----------------------
ncpu = 10 # numbers of CPUs available to run the MCMC
ndim = nmf -1 + 1 # number of dimensions = no of mf components, -1 for 0kG, +1 for scaling factor
chlen = 1000 # chain lengths
nwalkers = 60 # number of chains / walkers
ESSlim = 2500 # the minimum number of independent samples we require for each parameter
bidx_aie = 800  # burn-in length, do verify with the output plots that this is all-right!!

# Plotting preferences should go here:
#-------------------------------------
plt.rcParams['figure.figsize'] = [14, 4]
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'Roboto'


# Some welcoming print on stdout
#------------------------------
print('')
print('[info] starting magnotron.py | MCMC fitting of Zeeman broadening')
print('[info] v0.2 2019-01-30 | @astro_alexis | alexis.lavail@physics.uu.se')
print('[info] loading the input configuration .csv file')
# Loading the input CSV file into the input parameter dataframe
inputpar = magnotron_readcsv(csvlink)  # loads the config file with input parameters

# Optional run of crunch1.
#--------------------------------------------------------------------------------------------
#c1 = crunch1(inputpar,synthpath,mf)

# Optional run of crunch2. 
#--------------------------------------------------------------------------------------------
#c2 = crunch2(inputpar)

obslist = range(len(inputpar.uuid))    # if you want to loop on all the row of the input configuration file
obsid = int(sys.argv[1])
obslist=[obsid]

for iobs in obslist:
    f = h5py.File("synth-spectra-2.h5", "r")    # open the hdf5 file with synthetic spectra
    sp = f.get(inputpar.loc[iobs,'uuid'])   # loads the relevant dataset
    wave,obs,synth = np.squeeze(sp[0,0,:]),np.squeeze(sp[0,1,:]),np.squeeze(sp[:,2,:])  # isolates wavelength, observed intensity, synthetic intensity arrays
    f.close()   # close the hdf5 file

    print('')
    print ('[info] working on UUID={} | STAR={} | VSIN={}.'.format(inputpar.loc[iobs,'uuid'],inputpar.loc[iobs,'star'],inputpar.loc[iobs,'vsini']))

    # taking care of the wlmin and wlmax columns
    buff0 = inputpar.loc[iobs,('wlmin')]
    buff1 = inputpar.loc[iobs,('wlmax')]
    buff0 = buff0.split(';')    # splitting the string at the semicolons
    buff1 = buff1.split(';')    # splitting the string at the semicolons
    line = np.zeros([2,len(buff0)]) # creating an array to store wlmin (line[0,:]) and wlmax (line[1,:] values)
    line[0,:] = np.array(buff0, dtype=np.float64)   # storing wlmin values
    line[1,:] = np.array(buff1, dtype=np.float64)   # storing wlmax values
    # WLSEL is selecting the indices corresponding to the wavelength windows to be fitted
    wlsel = magnotron_wl_select(wave,line)
    x = np.squeeze(np.arange(len(wlsel)))
    y = np.squeeze(np.take(obs,wlsel,axis=0))
    # BIG SIGN: THE MEASUREMENT UNCERTAINTIES YERR IS ASSUMED CONSTANT ACROSS THE SPECTRUM
    yerr = (np.zeros_like(y) +1. ) / inputpar.loc[iobs,'snr'] # Uniform measurement uncertainty over the entire spectrum
    sy = np.squeeze(np.take(synth,wlsel,axis=1))
    aic = np.zeros(nmf-1)
    bic = np.zeros(nmf-1)
    chisqr = np.zeros(nmf-1)

    #for i_mf in [nmf-4]
    for i_mf in range(nmf-1):
        dsn = "i"+str(i_mf)
        ii = i_mf+2
        ndim = ii
        npoints = 0.
        # Opening the h5 file with starting values for the filling factor array
        #----------------------------------------------------------------------
        hff = h5py.File('ff.h5','r')
        ffgen = hff.get(dsn).value
        hff.close()
        ffgen
        ffs = ffgen.shape
        ffi = np.arange(ffs[1])
        np.random.shuffle(ffi)
        ffstart = ffgen[1:,ffi[0:nwalkers]]
        scalestart = 0.8 + 0.1*np.random.randn(nwalkers)
        posstart = np.vstack((ffstart,scalestart))
        posstart = posstart.T
        # Start of the Affine Invarient Ensemble MCMC part
        #-------------------------------------------------
        AIE_sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(y, yerr, sy[0:ii,:]),threads=ncpu) # initialize sampler
        pos = posstart
        # run MCMC sampler
        pos,prob,state = AIE_sampler.run_mcmc(pos, chlen)

        # Plotting the chains for each parameter and the burnin-length to check it's acceptable
        for iii in range(0,ii): # iterate over all the model parameters indexed by iii
            medmed = np.median(AIE_sampler.chain[:,bidx_aie:chlen,iii])
            plt.scatter(np.tile(range(0,chlen),nwalkers), AIE_sampler.chain[:,0:chlen,iii],color="k")
            plt.axhline(y=medmed, xmin=0, xmax=1,  color='#f03b20', ls ='--')
            plt.axvline(x=bidx_aie, ymin=0, ymax=1,  color='#f03b20', ls ='--')
            plt.savefig('plots/'+file_prefix+'burnin-mcmc_{}_iter{}_{}'.format(inputpar.loc[iobs,'uuid'],str(i_mf),str(iii)))
            plt.close('all')


        #print('AIE: production run')
        ESSmin=0.0 # initializing this variable here

        while ESSmin<ESSlim:
            # run MCMC sampler
            pos,prob,state = AIE_sampler.run_mcmc(None, chlen) # Now use "None" position to continue from earlier last position (in the burnin run above)
            try:
                tau = AIE_sampler.get_autocorr_time(c=2)
            except:
                #print('Auto-correlation not possible to estimate yet (not enough samples), running the MCMC again...')
                zzz = 0
            else:
                ESSmin = 100000000000000.0 # just some very large number to initialize
                for i in range(0,ndim): # iterate over all the model parameters indexed by i
                    #We print out the auto-correlation time (a k a length)
                    #print('[info] Auto-correlation time: ' + repr(tau))
                    # We then estimate the number of independent samples
                    numel = AIE_sampler.chain[:,bidx_aie:,0].size # the total number of chain elements
                    ESS = numel/(2.0*tau[i]) # estimate the number of independent samples (see Sharma 3.7.1)
                    #print('Estimated number of independent samples: ' + repr(ESS))
                    ESSmin = min([ESSmin, ESS])
                #print('[info] ESSmin = ' + repr(ESSmin))
            #print("End of loop iteration")
        print('[info] ESSmin = ' + repr(ESSmin))
        print('[info] Mean acceptance rate: {:3f}'.format(np.mean(AIE_sampler.acceptance_fraction)*100.))
        accfrac = np.mean(AIE_sampler.acceptance_fraction)*100.
        samples = AIE_sampler.chain[:,bidx_aie:, :]
        samples = samples.reshape((-1,ndim))
        # Compute parameter means and medians
        par_mean = np.mean(samples[:,:],axis=0)
        par_median = np.median(samples[:,:],axis=0)
        # Compute parameter confidence regions by looking at the quantiles
        Bsamples = np.zeros_like(samples[:,0])
        for i in range(ndim-1 ):
            Bsamples += samples[:,i]*mf[i+1]*0.01

        BB= np.percentile(Bsamples,[0.3,50,99.7],axis=0) # computing the different percentiles

        # Plotting the observed spectrum with the best fit synth spectrum
        plt.title('{} | {} | <B> = {:.2f} kG'.format(inputpar.loc[iobs,'uuid'],inputpar.loc[iobs,'star'],BB[1]))
        plt.errorbar(x,y,yerr=yerr,fmt=".")
        plt.plot(ymod(par_median[0:-1],par_median[-1],sy[0:ii,:]))
        plt.savefig('plots/'+file_prefix+'bestfit_{}_iter{}'.format(inputpar.loc[iobs,'uuid'], str(i_mf)))
        plt.close('all')

        # Printing the <B> results with 3-sigma confidence
        print("<B> (3-sigma confidence interval) {} kG" .format(BB))

        # Plotting the histogram of the samples for <B>
        plt.title('{} | {}'.format(inputpar.loc[iobs,'uuid'],inputpar.loc[iobs,'star']))
        plt.hist(Bsamples, bins='auto')
        plt.axvline(x=BB[0],color='red',linestyle=':')
        plt.axvline(x=BB[1],color='orange',linestyle='-',linewidth=2)
        plt.axvline(x=BB[2],color='red',linestyle=':')
        plt.xlabel("<B> (kG)")
        plt.savefig('plots/'+file_prefix+'Bhisto_{}_iter{}'.format(inputpar.loc[iobs,'uuid'],str(i_mf)))
        plt.close('all')

        pickle_filename = 'results/'+file_prefix+'samples_{}_iter{}'.format(inputpar.loc[iobs,'uuid'],str(i_mf))
        with open(pickle_filename+'.pickle', 'wb') as handle:
            pickle.dump(samples, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # Plotting the corner plots
        #fig = corner.corner(samples, quantiles=[0.16, 0.50, 0.84], truths=par_median)
        #plt.savefig('plots/'+file_prefix+'corner_{}_iter{}'.format(inputpar.loc[iobs,'uuid'],str(i_mf)))
        #plt.close('all')

        # Computing the information criteria: AIC and BIC
        aic[i_mf] = -2.* lnlike(y,yerr,ymod(par_median[0:-1],par_median[-1],sy[0:ii,:])) + 2. * ii
        bic[i_mf] = -2.* lnlike(y,yerr,ymod(par_median[0:-1],par_median[-1],sy[0:ii,:])) + ii * np.log(len(yerr))
        chisqr[i_mf],npoints = chideux(y,yerr,ymod(par_median[0:-1],par_median[-1],sy[0:ii,:]))
        chisqr[i_mf] = chisqr[i_mf] / (npoints - ndim - 1)

        outputfilename = 'results/'+file_prefix+'log_{}'.format(inputpar.loc[iobs,'uuid'])
        if i_mf == 0:
            f= open(outputfilename,"w")
            f.write('iter\tbmin\tbbest\tbmax\tchi2\taic\tbic\tacceptance-fraction\tESSmin\n')
            f.write('{}\t{}\t{}\t{}\t{:.1f}\t{:.1f}\t{}\t{}\t{}\n'.format(str(i_mf),BB[0],BB[1],BB[2],chisqr[i_mf],aic[i_mf],bic[i_mf], accfrac, ESSmin))
            f.close()
        else:
            f= open(outputfilename,"a")
            f.write('{}\t{}\t{}\t{}\t{:.1f}\t{:.1f}\t{}\t{}\t{}\n'.format(str(i_mf),BB[0],BB[1],BB[2],chisqr[i_mf],aic[i_mf],bic[i_mf], accfrac, ESSmin))
            f.close()

        txtf = 'results/'+file_prefix+str(inputpar.loc[iobs,('uuid')])+'-spec-'+str(inputpar.loc[iobs,('star')])+'_iter'+str(i_mf)+'.ascii'    # creating the filename for the ascii output file with spectra
        spec = ymod(par_median[0:-1],par_median[-1],synth[0:ii,:])
        np.savetxt(txtf,np.transpose([wave[:],obs[:], spec]), fmt='%1.5f')    # saving the ascii output file
	# Important: resetting the AIE sampler, otherwise memory leaks quick!
        AIE_sampler.reset()
        del pos
        del prob
        del state
        del samples
        del Bsamples
        print('#{} AIC = {:.1f}'.format(i_mf+1,aic[i_mf]))
        print('#{} BIC = {:.1f}'.format(i_mf+1,bic[i_mf]))
        print('#{} chi2 = {:.1f}'.format(i_mf+1,chisqr[i_mf]))

    # end of iteration loop

    sp = None
    del sp

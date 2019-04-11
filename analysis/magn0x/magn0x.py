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

# A simple chi-square function
def chideux(y,yerr,model_y):
    n = len(y)
    value = np.sum(((y-model_y)**2.)/(yerr**2.))
    return value,n

# The hard-coded variables should go here:
# --------------------------------------------
mf=[0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0] # magnetic field strengths to be considered
nmf = len(mf) # number of magnetic field values
csvlink = 'https://docs.google.com/spreadsheets/d/1bjMaMhXWs4MK2BCSGerF_NB0rqY2kqwetAtBUF_4feQ/export?format=csv' # link to the CSV input file on Google Sheets
scl = np.arange(200)/100.0  # initializing the scale array
file_prefix = ""    # a string that will be prefixed to output plots and files

# Plotting preferences should go here:
#-------------------------------------
plt.rcParams['figure.figsize'] = [14, 4]
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'Roboto'


# Some welcoming print on stdout
#------------------------------
print('')
print('[info] starting magn0x.py | dumb full-grid-search fitting of Zeeman broadening')
print('[info] v1.0 2018-02-08 | alexis.lavail@physics.uu.se')
print('[info] loading the input configuration .csv file')
# Loading the input CSV file into the input parameter dataframe
inputpar = magnotron_readcsv(csvlink)  # loads the config file with input parameters
outputpar = inputpar
outputpar['bmin'] = 0.
outputpar['bbest'] = 0.
outputpar['bmax'] = 0.
outputpar['chisquare'] = 0.
outputpar['scale'] = 0.
print ('[info] loading the filling factor file')
ff = magnotron_loadff('ff0x.p')  # loads the filling factor combinations

obslist = range(len(inputpar.uuid))
iobs = int(sys.argv[1])

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


nff = len(ff[0,:])  # number of filling factor combinations
chi = np.zeros([nff,nmf]) # init the chisquare array
scale = np.zeros([nff,nmf])   # init scaling factor array
modBarr = np.zeros([nff,nmf]) # init <B> array
dof = len(wlsel)-5  # degree of freedom = number of wavelength points - 1 (filling factor) -1 (scaling factor) -2 (m.f) -1 (there's always -1 anyway that you don't know why)
dofl = float(dof)   # converts from int to flt
b_counter = 0
for imf in range(nmf):
    for ii in range(nff): # entering the main loop on the filling factor combinations
    # for each combination: (1) compute the corresponding magnetic spectrum, (2) find the optimal scaling factor, (3) compute the chi-square then
        sc,ch = magnotron_findscale(0.01*(sy[0,:] * ff[0,ii] + sy[imf,:] * ff[1,ii]),y,scl,inputpar.snr[iobs])
        scale[ii,imf] = sc  # scale
        chi[ii,imf] = ch    # chisquare
        modBarr[ii,imf] = 0.01 * (ff[1,ii]*mf[imf])  # <B>

modB,wbest = magnotron_bstat(modBarr,chi,dof)
minchi = np.amin(chi)   # minimum chi-square value

spec = (0.01*(synth[0,:] * ff[0][wbest[0]] + synth[wbest[1],:] * ff[1][wbest[0]] + synth[wbest[1],:])) * scale[wbest] +1. - scale[wbest]   # spectrum for the solution
ffw = ff[0,:].argmax()  # finds the index of the non-magnetic solution
spec0 = synth[0,:] * scale[ffw][0] +1. -scale[ffw][0] # best-scaled non-magnetic spectrum

# assigning the 3 values of modB into the output parameter data structure
outputpar.loc[iobs,('bbest')], outputpar.loc[iobs,('bmin')], outputpar.loc[iobs,('bmax')] = modB[:]

# assigning the scale and chi2 values to the output parameter data structure
outputpar.loc[iobs, ('scale')], outputpar.loc[iobs, ('chisquare')]  = scale[wbest],chi[wbest]/dofl

txtf = 'results/'+file_prefix+str(inputpar.loc[iobs,('uuid')])+'-'+str(inputpar.loc[iobs,('star')])+'.ascii'    # creating the filename for the ascii output file with spectra
np.savetxt(txtf,np.transpose([wave[:],obs[:], spec, spec0]), fmt='%1.5f')    # saving the ascii output file
print('[info] done with the analysis. <B> = [{:.2f},{:1.2f},{:1.2f}] kG. chi2 = {:1.2f}'.format(modB[0],modB[1],modB[2],chi[wbest]/dofl))   # little print with results
logf = open('results/'+file_prefix+'magn0x.log',"a")
outstring = str(inputpar.loc[iobs,('uuid')]) + (f"\t") + str(inputpar.loc[iobs,('star')])
outstring += (f"\t{modB[0]:.2f}") + (f" ({chi[wbest]/dofl:.2f})\n")
logf.write(outstring)
logf.close()

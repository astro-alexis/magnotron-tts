import numpy as np
import os as os
import sys
import glob as glob # glob.glob I like how this sounds
import pandas as pd
import requests
import pickle
import emcee
import h5py
#import matplotlib.pyplot as plt
#---------------------------------------------------------------


# The OBSPATH (path to the observed spectrum file) can point to several files, separated by ':'
# magnotron_split_obspath takes the OBSPATH in input, splits the thing, and returns an array
def magnotron_split_obspath(obspath):
    splitobspath = obspath.split("+")
    return splitobspath

# MAGNOTRON_OBSREAD opens the array of inputfile (2 column ascii text file) and
# puts the wavelength and spectrum into the array obs
def magnotron_obsread(inputfile):
    ii = 0  # initializes counter
    for file in inputfile:  # loops over the inputfile array
        my_path = os.path.abspath(os.path.dirname(__file__))
        filepath = os.path.join(my_path, file)
        filepath = filepath.strip()
        if os.path.exists(filepath) != True:    #checking that the file is present
            print("[error] magnotron_obsread(): The spectrum file is not found at {}".format(filepath))  #if not, print an error message
            print(os.path.exists(filepath))
            sys.exit(5) # and exit the program

        if ii == 0: # if first file then create the obs array
            obs = np.loadtxt(file, unpack=True) # unpacks the file into obs
            ii += 1 # increments counter
        else:   # if not first file, unpacks file into buffer
            buff = np.loadtxt(filepath, unpack=True)
            buff = buff.reshape((2,-1))
            obs = np.concatenate((obs,buff),axis=1) # and concatenate buffer into already existing obs
            ii += 1
    return obs

# MAGNOTRON_WL_WELECT takes into input the 1D array of synthetic wavelength, and the wave_min wave_max array
# returned from magnotron_get_line. It returns the subset of wavlength point inside the wavelength regions
def magnotron_wl_select(synth,wl):
    for i in range(len(wl[0,:])):   # loop over the different wavelength regions
        if i == 0:
            wls = np.argwhere((synth >= wl[0,i]) & (synth <= wl[1,i]))
        else:
            buff = np.argwhere((synth >= wl[0,i]) & (synth <= wl[1,i]))
            wls = np.concatenate((wls,buff),axis=0)
    return(wls)

# this routine makes sure that the 2d array synth and obs are interpolated on the same support (wavelength grid)
# in practice the synthetic spectrum is interpolated on the obs spectrum.
def magnotron_equal_support(synth,obs):
    from scipy import interpolate
    from scipy.interpolate import interp1d
    minsynth = np.amin(synth[0,:])  # min wavelength of synthetic spectrum
    maxsynth = np.amax(synth[0,:])  # max wavelength of synthetic spectrum
    minobs = np.amin(obs[0,:])  # min wavelength of observed spectrum
    maxobs = np.amax(obs[0,:])  # max wavelength of observed spectrum
    minglob = np.amax([minsynth,minobs])    # finds the minimum of the shared wavelength domain between synth and obs
    maxglob = np.amin([maxsynth,maxobs])    # finds the maximum of the shared wavelength domain between synth and obs
    w_s = np.argwhere((synth[0,:] >= minglob) & (synth[0,:] <= maxglob))    # indices of the shared wavelength domain in the synth file
    w_s.sort()  # sorts it
    fs = interpolate.splrep(synth[0,w_s],synth[1,w_s],s=0)  # spline interpolation of the synthetic spectrum
    w_obs = np.argwhere((obs[0,:] >= minglob) & (obs[0,:] <= maxglob))  # indices of the shared wavelength domain in the obs file
    w_obs.sort()    # sorts it
    y_obs = obs[1,w_obs]
    x_obs = obs[0,w_obs]
    y_synth = interpolate.splev(x_obs, fs, der=0)   # interpolates the synth with the observed spectrum wavelength grid
    output = np.stack((x_obs,y_obs,y_synth))
    output = output.reshape([3,-1])
    return(output)

def magnotron_readcsv(link):
    response = requests.get(link)
    with open("input.csv", "wb") as text_file:
        text_file.write(response.content)
    df = pd.read_csv('input.csv', dtype={'uuid': object, 'star': object, 'vsini': np.float64, 'snr':np.float64, 'rv': np.float64, 'instr': np.float64, 'obspath': object,'logg': object ,'teff': object, 'wlmin': object, 'wlmax': object, 'rowid':np.float64})
    return(df)

def magnotron_clean(files):
    for f in files:
        if os.path.exists(f):
            os.remove(f)
    return(0)

# MAGNOTRON_SYNTHREAD reads the .prf files outputted from the s3div.Linux program. These are the disk-integrated and vsini-vmacro-instr broadened synthetic spectra
def magnotron_synthread(file):
    my_path = os.path.abspath(os.path.dirname(__file__))
    filepath = os.path.join(my_path, file)
    filepath = filepath.strip()
    if os.path.isfile(filepath) != True:    #checking that the file is present
        print("[error] magnotron_synthread(): The synthetic spectrum file is not found at {}".format(filepath))  #if not, print an error message
        sys.exit(1) # and exit the program

    with open(filepath) as f:   #opens the file
        first_line = f.readline()   # reads the first line which is a string containing a number of lines (to skip),  '-' and some text
        a = first_line.split('-')   # splits the first line using '-' as a delimiter, in order to get extract the number of lines to skip
        nl = int(a[0])  # converts the number of lines to skip into integer
        s = np.genfromtxt(filepath,skip_header=nl+2,skip_footer=2,unpack=True)  # skips nl + 2 lines to get to the spectrum. skips the last 2 lines which are text
    return(s)   # returns the 2D array [wave,spectrum]

# CRUNCH1 is optionally called at the beginning of the main routine
# It goes through the input parameters database, isolates a unique
# combination of stellar parameters, and computes a synthethic
# stellar spectra for each combination of stellar parameters, and
# for the given magnetic field array. It saves the resulting
# stellar templates in a HDF5 file that can be used by CRUNCH2
#-------------------------------------------------------------
def crunch1(inputpar,synthpath,mf,output="synth-spectra-1.h5"):
    f = h5py.File(output, "w")    # opens a HDF5 file
    stellar_par = inputpar[['vsini','logg','teff']].drop_duplicates()   # selects the unique combination of stellar parameters vsini / teff / log(g)
    n_stellar_par = len(stellar_par.logg)   # counts such unique combinations
    print("[info] preparing the grid of magnetic synthetic spectra")
    print("[info] synthpath: {}".format(synthpath))
    print("[info] the unique set of stellar parameters is:")
    print("-------------------------")
    print(stellar_par)
    print("-------------------------")
    nmf = len(mf)
    # Now looping on the different parameter combinations
    for ipar in stellar_par.index:
        for imf in range(len(mf)):  # looping on the different magnetic field strength selected
            str_mf = '%04.1f' % mf[imf]     # converting the mf value to a flot such as 10.0, or 07.2 (4 characters including decimal point)
            string = 'g'+stellar_par.loc[ipar,'logg']+'_'+stellar_par.loc[ipar,'teff']+'*'
            string += str_mf +'*.mout'
            if not (glob.glob(synthpath+string)):   # checking if there is a synthetic spectra with the right filename
                print("ERROR: FILE NOT FOUND {}".format(string))
                sys.exit(1) # and exit the program

            vsini = inputpar.loc[ipar,('vsini')]      # getting vsini from input parameters
            instr = inputpar.loc[ipar,('instr')]      # and the spectral resolution
            str_vsini = '%05.2f' % vsini  # converts vsini to a string
            str_instr = '%.1f' % instr # converts spectral resolution to a string

            synthfile = glob.glob(synthpath+string)    # filepath to the magnetic synthetic spectrum
            cs2= 's3div.Linux '+synthfile[0] + ' s.prf ' + str_vsini + ' 0. 1000000. ' +str_instr    # command-line to apply different broadenings to the synthetic spectrum
            os.popen(cs2).read()    # running the command and saving the convolved spectrum into a file
            tsynth = magnotron_synthread('s.prf')
            if imf == 0:
                nsp = len(tsynth[0,:])  # number of points in the spectrum
                synth = np.zeros([nmf,2,nsp]) # creates an array to store the magnetic synthetic spectra
            synth[imf,:,:] = tsynth
            cs2 = 'rm s.prf'
            os.popen(cs2).read()
            status = 0
            print("crunching | vsini = {} km/s | log(g)= {} | t_eff = {} K| <B> = {} kG".format(str_vsini,stellar_par.loc[ipar,'logg'],stellar_par.loc[ipar,'teff'], str_mf))
        datasetname = 'dataset'+str_vsini+'_g'+stellar_par.loc[ipar,'logg']+'_'+stellar_par.loc[ipar,'teff']
        print("datasetname: {}".format(datasetname))
        f.create_dataset(datasetname, data=synth, dtype=np.float64)
        print("saving the synthetic spectrum in the dataset: {}".format(datasetname))
    f.close()
    # end of loop
    #
    return 0
# END OF CRUNCH1
#---------------

# CRUNCH2 takes the input parameter database as input. It loads the output
# file generated by CRUNCH1. For each observed spectra, it will interpolate
# the relevant synthetic spectra onto the same wavelength base.
# The synth array containing the spectra is returned into a dataset
# within the output HDF5 file.
#-----------------------------
def crunch2(inputpar,input="synth-spectra-1.h5", output="synth-spectra-2.h5"):
    f2 = h5py.File(output, "w")    # opens a HDF5 file
    with h5py.File(input,'r') as hf1:
        for iobs in inputpar.index:
            obspath = magnotron_split_obspath(inputpar.loc[iobs,('obspath')])
            obs = magnotron_obsread(obspath)  # reading in the observed spectrum: 2D array with (1) wave (2) normalized intensity
            vsini = inputpar.loc[iobs,('vsini')]      # getting vsini from input parameters
            instr = inputpar.loc[iobs,('instr')]      # and the spectral resolution
            str_vsini = '%05.2f' % vsini  # converts vsini to a string
            dsname = 'dataset'+str_vsini+'_g'+inputpar.loc[iobs,'logg']+'_'+inputpar.loc[iobs,'teff']
            synth = hf1.get(dsname)
            ns = synth.shape[0]
            for imf in range(ns):
                sp_int = magnotron_equal_support(np.squeeze(synth[imf,:,:]),obs)
                if imf == 0:
                    sp = np.zeros([ns,3,len(sp_int[0,:])])
                sp[imf,:,:] = sp_int
            dsname2 = inputpar.loc[iobs,'uuid']
            f2.create_dataset(dsname2, data=sp, dtype=np.float64)
    f2.close()
    return 0
# END OF CRUNCH2
#---------------

# CRUNCH1vsini is optionally called at the beginning of the main routine
# It goes through the input parameters database, isolates a unique
# combination of stellar parameters, and computes a synthethic
# stellar spectra for each combination of stellar parameters, and
# for the given magnetic field array. It computes spectra for the target vsini,
# + [-2,-1.5, -1, -0.5, 0, 0.5, 1., 1.5, 2.0].
# It saves the resulting stellar templates in a HDF5 file that can be used by CRUNCH2
#-------------------------------------------------------------

def crunch1vsini(inputpar,synthpath,mf,output="synth-spectra-1-vsini.h5"):
    f = h5py.File(output, "w")    # opens a HDF5 file
    stellar_par = inputpar[['vsini','logg','teff']].drop_duplicates()   # selects the unique combination of stellar parameters vsini / teff / log(g)
    n_stellar_par = len(stellar_par.logg)   # counts such unique combinations
    delta_vsini = [-2,-1., 0., 1., 2.0]
    print("[info] preparing the grid of magnetic synthetic spectra")
    print("[info] synthpath: {}".format(synthpath))
    print("[info] the unique set of stellar parameters is:")
    print("-------------------------")
    print(stellar_par)
    print("-------------------------")
    nmf = len(mf)
    # Now looping on the different parameter combinations
    for ipar in stellar_par.index:
        for imf in range(len(mf)):  # looping on the different magnetic field strength selected
            str_mf = '%04.1f' % mf[imf]     # converting the mf value to a flot such as 10.0, or 07.2 (4 characters including decimal point)
            string = 'g'+stellar_par.loc[ipar,'logg']+'_'+stellar_par.loc[ipar,'teff']+'*'
            string += str_mf +'*.mout'
            if not (glob.glob(synthpath+string)):   # checking if there is a synthetic spectra with the right filename
                print("ERROR: FILE NOT FOUND {}".format(string))
                sys.exit(1) # and exit the program

            for ivsini in range(len(delta_vsini)):
                vsini = inputpar.loc[ipar,('vsini')] + delta_vsini[ivsini]      # getting vsini from input parameters
                instr = inputpar.loc[ipar,('instr')]      # and the spectral resolution
                str_vsini = '%05.2f' % vsini  # converts vsini to a string
                str_instr = '%.1f' % instr # converts spectral resolution to a string

                synthfile = glob.glob(synthpath+string)    # filepath to the magnetic synthetic spectrum
                cs2= 's3div.Linux '+synthfile[0] + ' s.prf ' + str_vsini + ' 0. 1000000. ' +str_instr    # command-line to apply different broadenings to the synthetic spectrum
                os.popen(cs2).read()    # running the command and saving the convolved spectrum into a file
                tsynth = magnotron_synthread('s.prf')
                if imf == 0 and ivsini == 0:
                    nsp = len(tsynth[0,:])  # number of points in the spectrum
                    synth = np.zeros([nmf,len(delta_vsini), 2,nsp]) # creates an array to store the magnetic synthetic spectra
                    #print(synth.shape)
                    #print(tsynth.shape)
                synth[imf,ivsini,:,:] = tsynth
                cs2 = 'rm s.prf'
                os.popen(cs2).read()
                status = 0
                print("crunching | vsini = {} km/s | log(g)= {} | t_eff = {} K| <B> = {} kG".format(str_vsini,stellar_par.loc[ipar,'logg'],stellar_par.loc[ipar,'teff'], str_mf))
        str_vsini = '%05.2f' % inputpar.loc[ipar,('vsini')]
        datasetname = 'dataset'+str_vsini+'_g'+stellar_par.loc[ipar,'logg']+'_'+stellar_par.loc[ipar,'teff']
        print("datasetname: {}".format(datasetname))
        f.create_dataset(datasetname, data=synth, dtype=np.float64)
        print("saving the synthetic spectrum in the dataset: {}".format(datasetname))
    f.close()
    # end of loop
    #
    return 0
# END OF CRUNCH1
#---------------

# CRUNCH2vsini takes the input parameter database as input. It loads the output
# file generated by CRUNCH1. For each observed spectra, it will interpolate
# the relevant synthetic spectra onto the same wavelength base.
# The synth array containing the spectra is returned into a dataset
# within the output HDF5 file.
#-----------------------------
def crunch2vsini(inputpar,input="synth-spectra-1-vsini.h5", output="synth-spectra-2-vsini.h5"):
    f2 = h5py.File(output, "w")    # opens a HDF5 file
    with h5py.File(input,'r') as hf1:
        for iobs in inputpar.index:
            obspath = magnotron_split_obspath(inputpar.loc[iobs,('obspath')])
            obs = magnotron_obsread(obspath)  # reading in the observed spectrum: 2D array with (1) wave (2) normalized intensity
            wwuu, iiuu = np.unique(obs[0,:], return_index=True)
            obs = obs[:,iiuu]

            vsini = inputpar.loc[iobs,('vsini')]      # getting vsini from input parameters
            str_vsini = '%05.2f' % vsini  # converts vsini to a string
            dsname = 'dataset'+str_vsini+'_g'+inputpar.loc[iobs,'logg']+'_'+inputpar.loc[iobs,'teff']
            synth = hf1.get(dsname).value
            ns = synth.shape[0]
            nv = synth.shape[1]
            no = synth.shape[2]
            nsp = synth.shape[3]

            for imf in range(ns):
                for ivsini in range(nv):
                    sp_int = magnotron_equal_support(np.squeeze(synth[imf,ivsini,:,:]),obs)
                    if imf == 0 and ivsini == 0:
                        sp = np.zeros([ns,nv,3,len(sp_int[0,:])])
                    sp[imf,ivsini,:,:] = sp_int
            dsname2 = inputpar.loc[iobs,'uuid']
            print(sp.shape)
            f2.create_dataset(dsname2, data=sp, dtype=np.float64)
    f2.close()
    return 0
# END OF CRUNCH2
#---------------

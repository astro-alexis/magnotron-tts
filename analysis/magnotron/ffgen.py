# ffgen0 saves arrays of filling factor that is an input for the magnotron procedure
# the final arrays ffX are  X x N array, each slice of X elements is a combination of filling factors which sum is 100
# the final array is saved into a pickle file.

# Only change the ff_increment variable, which sets the increment between two possible filling factor values.
# for exemple ff_increment = 25 means ff0 = [0,25,50,75,100]
import h5py
import numpy as np
import os as os

print('[info] ffgen.py | generates filling factor combinations for magnotron')
print('[info] v1.0 | 2019-01-29 | alexis.lavail@physics.uu.se')
# Variables

ff_increment = 1
ff_max = 100

# Creating the filling factors arrays
ff0 = np.arange(ff_max/ff_increment+1, dtype=np.float64) * ff_increment
ff1 = np.arange(ff_max/ff_increment+1, dtype=np.float64) * ff_increment
ff2 = np.arange(ff_max/ff_increment+1, dtype=np.float64) * ff_increment
ff3 = np.arange(ff_max/ff_increment+1, dtype=np.float64) * ff_increment
ff4 = np.arange(ff_max/ff_increment+1, dtype=np.float64) * ff_increment
ff5 = np.arange(ff_max/ff_increment+1, dtype=np.float64) * ff_increment
ff6 = np.arange(ff_max/ff_increment+1, dtype=np.float64) * ff_increment
ff7 = np.arange(ff_max/ff_increment+1, dtype=np.float64) * ff_increment

#-----
# B = [0,2] kG
# ffxx will store the final array of filling factor
ff = np.empty((0,2), np.float64)
print('[info] entering the main loop')
# looping through all the filling factor possibilities and checking in which cases
# the sum of all filling factor is 100%: is yes -> store into ff array

for ii0 in range(len(ff0)):
    for ii1 in range(len(ff1)):
        if ff0[ii0] + ff1[ii1] == 100:
            ff  = np.append(ff, np.array([[ff0[ii0],ff1[ii1]]]), axis=0)
# the main loop is done.
ff = ff.T
print('[result] generated a final array with shape {}'.format(ff.shape))
# Storing the ff array into a HDF5 file now
if os.path.exists('./ff.h5'):
    os.remove('./ff.h5')
    print('[info] removed an existing "ff.h5" file')

f2 = h5py.File("ff.h5", "w")    # opens a HDF5 file
dsname = "i0"
f2.create_dataset(dsname, data=ff, dtype=np.float64)
f2.close()

#-----
# B = [0,2,4] kG
# ffxx will store the final array of filling factor
ff = np.empty((0,3), np.float64)
print('[info] entering the main loop')
# looping through all the filling factor possibilities and checking in which cases
# the sum of all filling factor is 100%: is yes -> store into ff array

for ii0 in range(len(ff0)):
    for ii1 in range(len(ff1)):
        for ii2 in range(len(ff2)):
            if ff0[ii0] + ff1[ii1] + ff2[ii2] == 100:
                ff  = np.append(ff, np.array([[ff0[ii0],ff1[ii1],ff2[ii2]]]), axis=0)
# the main loop is done.
ff = ff.T
print('[result] generated a final array with shape {}'.format(ff.shape))
# Storing the ff array into a HDF5 file now

f2 = h5py.File("ff.h5", "a")    # opens a HDF5 file
dsname = "i1"
f2.create_dataset(dsname, data=ff, dtype=np.float64)
f2.close()


ff_increment = 10
ff_max = 100

# Creating the filling factors arrays
ff0 = np.arange(ff_max/ff_increment+1, dtype=np.float64) * ff_increment
ff1 = np.arange(ff_max/ff_increment+1, dtype=np.float64) * ff_increment
ff2 = np.arange(ff_max/ff_increment+1, dtype=np.float64) * ff_increment
ff3 = np.arange(ff_max/ff_increment+1, dtype=np.float64) * ff_increment
ff4 = np.arange(ff_max/ff_increment+1, dtype=np.float64) * ff_increment
ff5 = np.arange(ff_max/ff_increment+1, dtype=np.float64) * ff_increment
ff6 = np.arange(ff_max/ff_increment+1, dtype=np.float64) * ff_increment
ff7 = np.arange(ff_max/ff_increment+1, dtype=np.float64) * ff_increment



#-----
# B = [0,2,4,6] kG
# ffxx will store the final array of filling factor
ff = np.empty((0,4), np.float64)
print('[info] entering the main loop')
# looping through all the filling factor possibilities and checking in which cases
# the sum of all filling factor is 100%: is yes -> store into ff array

for ii0 in range(len(ff0)):
    for ii1 in range(len(ff1)):
        for ii2 in range(len(ff2)):
            for ii3 in range(len(ff3)):
                if ff0[ii0] + ff1[ii1] + ff2[ii2] + ff3[ii3] == 100:
                    ff  = np.append(ff, np.array([[ff0[ii0],ff1[ii1],ff2[ii2],ff3[ii3]]]), axis=0)
# the main loop is done.
ff = ff.T
print('[result] generated a final array with shape {}'.format(ff.shape))
# Storing the ff array into a HDF5 file now

f2 = h5py.File("ff.h5", "a")    # opens a HDF5 file
dsname = "i2"
f2.create_dataset(dsname, data=ff, dtype=np.float64)
f2.close()

#-----
# B = [0,2,4,6,8] kG
# ffxx will store the final array of filling factor
ff = np.empty((0,5), np.float64)
print('[info] entering the main loop')
# looping through all the filling factor possibilities and checking in which cases
# the sum of all filling factor is 100%: is yes -> store into ff array

for ii0 in range(len(ff0)):
    for ii1 in range(len(ff1)):
        for ii2 in range(len(ff2)):
            for ii3 in range(len(ff3)):
                for ii4 in range(len(ff4)):
                    if ff0[ii0] + ff1[ii1] + ff2[ii2] + ff3[ii3] + ff4[ii4] == 100:
                        ff  = np.append(ff, np.array([[ff0[ii0],ff1[ii1],ff2[ii2],ff3[ii3],ff4[ii4]]]), axis=0)
# the main loop is done.
ff = ff.T
print('[result] generated a final array with shape {}'.format(ff.shape))
# Storing the ff array into a HDF5 file now

f2 = h5py.File("ff.h5", "a")    # opens a HDF5 file
dsname = "i3"
f2.create_dataset(dsname, data=ff, dtype=np.float64)
f2.close()

ff_increment = 20
ff_max = 100

# Creating the filling factors arrays
ff0 = np.arange(ff_max/ff_increment+1, dtype=np.float64) * ff_increment
ff1 = np.arange(ff_max/ff_increment+1, dtype=np.float64) * ff_increment
ff2 = np.arange(ff_max/ff_increment+1, dtype=np.float64) * ff_increment
ff3 = np.arange(ff_max/ff_increment+1, dtype=np.float64) * ff_increment
ff4 = np.arange(ff_max/ff_increment+1, dtype=np.float64) * ff_increment
ff5 = np.arange(ff_max/ff_increment+1, dtype=np.float64) * ff_increment
ff6 = np.arange(ff_max/ff_increment+1, dtype=np.float64) * ff_increment
ff7 = np.arange(ff_max/ff_increment+1, dtype=np.float64) * ff_increment



#-----
# B = [0,2,4,6,8,10] kG
# ffxx will store the final array of filling factor
ff = np.empty((0,6), np.float64)
print('[info] entering the main loop')
# looping through all the filling factor possibilities and checking in which cases
# the sum of all filling factor is 100%: is yes -> store into ff array

for ii0 in range(len(ff0)):
    for ii1 in range(len(ff1)):
        for ii2 in range(len(ff2)):
            for ii3 in range(len(ff3)):
                for ii4 in range(len(ff4)):
                    for ii5 in range(len(ff5)):
                        if ff0[ii0] + ff1[ii1] + ff2[ii2] + ff3[ii3] + ff4[ii4] + ff5[ii5] == 100:
                            ff  = np.append(ff, np.array([[ff0[ii0],ff1[ii1],ff2[ii2],ff3[ii3],ff4[ii4],ff5[ii5]]]), axis=0)
# the main loop is done.
ff = ff.T
print('[result] generated a final array with shape {}'.format(ff.shape))
# Storing the ff array into a HDF5 file now

f2 = h5py.File("ff.h5", "a")    # opens a HDF5 file
dsname = "i4"
f2.create_dataset(dsname, data=ff, dtype=np.float64)
f2.close()

#-----
# B = [0,2,4,6,8,10,12] kG
# ffxx will store the final array of filling factor
ff = np.empty((0,7), np.float64)
print('[info] entering the main loop')
# looping through all the filling factor possibilities and checking in which cases
# the sum of all filling factor is 100%: is yes -> store into ff array

for ii0 in range(len(ff0)):
    for ii1 in range(len(ff1)):
        for ii2 in range(len(ff2)):
            for ii3 in range(len(ff3)):
                for ii4 in range(len(ff4)):
                    for ii5 in range(len(ff5)):
                        for ii6 in range(len(ff6)):
                            if ff0[ii0] + ff1[ii1] + ff2[ii2] + ff3[ii3] + ff4[ii4] + ff5[ii5] + ff6[ii6] == 100:
                                ff  = np.append(ff, np.array([[ff0[ii0],ff1[ii1],ff2[ii2],ff3[ii3],ff4[ii4],ff5[ii5],ff6[ii6]]]), axis=0)
# the main loop is done.
ff = ff.T
print('[result] generated a final array with shape {}'.format(ff.shape))
# Storing the ff array into a HDF5 file now

f2 = h5py.File("ff.h5", "a")    # opens a HDF5 file
dsname = "i5"
f2.create_dataset(dsname, data=ff, dtype=np.float64)
f2.close()

#-----
# B = [0,2,4,6,8,10,12,14] kG
# ffxx will store the final array of filling factor
ff = np.empty((0,8), np.float64)
print('[info] entering the main loop')
# looping through all the filling factor possibilities and checking in which cases
# the sum of all filling factor is 100%: is yes -> store into ff array

for ii0 in range(len(ff0)):
    for ii1 in range(len(ff1)):
        for ii2 in range(len(ff2)):
            for ii3 in range(len(ff3)):
                for ii4 in range(len(ff4)):
                    for ii5 in range(len(ff5)):
                        for ii6 in range(len(ff6)):
                            for ii7 in range(len(ff7)):
                                if ff0[ii0] + ff1[ii1] + ff2[ii2] + ff3[ii3] + ff4[ii4] + ff5[ii5] + ff6[ii6] + ff7[ii7] == 100:
                                    ff  = np.append(ff, np.array([[ff0[ii0],ff1[ii1],ff2[ii2],ff3[ii3],ff4[ii4],ff5[ii5],ff6[ii6],ff7[ii7]]]), axis=0)
# the main loop is done.
ff = ff.T
print('[result] generated a final array with shape {}'.format(ff.shape))
# Storing the ff array into a HDF5 file now

f2 = h5py.File("ff.h5", "a")    # opens a HDF5 file
dsname = "i6"
f2.create_dataset(dsname, data=ff, dtype=np.float64)
f2.close()

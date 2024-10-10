import numpy as np
import pandas as pd
import os
from functions import *

J = 1

T = 2/np.log(1+np.sqrt(2)) # critical temperature for the 2d Ising model

neq = int(5e4) # number of equilibration steps 
nMCsteps = int(neq+int(5e3))
nsamples = 100
nstepssamples = (nMCsteps - neq)//nsamples # number of time steps between samples

N = 320 # side of the square lattice

print('Simulating a %d x %d 2d Ising model with nn interaction at criticality'%(N,N))
print('Number of Monte Carlo steps: %d' % nMCsteps)
print('of which the number of equilibration steps is: %d' % neq)
print('and the number of time steps between samples is: %d' % nstepssamples + '; i.e. the number of samples is: %d' % nsamples)

dataPath = 'configs'

# Check if a directory named 'configs' exists, if not create it
if not os.path.exists(dataPath):
    os.makedirs(dataPath)

# Bool variable to select whether to re-perform the whole data production
reTake = False

if reTake or len(os.listdir(dataPath))==0:
    # Counter for saved configurations
    k = 0
else:
    k = np.max([int((f.split('_')[-1]).split('.')[0]) for f in os.listdir(dataPath)])+1

print('Beginning simulation')# Generate initial configuration
config = configuration(N)

for t in range(nMCsteps):
    # Print the progress of the simulation
    if t>0 and t%(nMCsteps//10) == 0:
        print('\tPercentage of progress of the simulation',int(t/nMCsteps*100),'%')
    config = MCWolffMoveNumba(T,config)
    if (t > neq) and (t-neq)%nstepssamples == 0:
        pd.DataFrame(config).to_csv(dataPath+'/config_N%d_%d.csv'%(N,k),header=None,index=False)
        k += 1
pd.DataFrame(config).to_csv(dataPath+'/config_N%d_%d.csv'%(N,k),header=None,index=False)        
print('\tPercentage of progress of the simulation 100 %')
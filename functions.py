import numpy as np
from matplotlib import pyplot as plt
from numba import njit

# Generate an initial configuration for the 2d Ising system
def configuration(N):
    # Generate a random configuration of spins, where 1 is up and -1 is down
    return 2*np.random.randint(2, size=(N,N), dtype=np.int8)-1

# Calculate nearest neighbours energy term for a given configuration
# This involves products of two spins, therefore it's an even coupling
def nNbEnergy(config,J=1):
    energy = 0
    N = np.shape(config)[0]
    for i in range(N):
        for j in range(N):
            s = config[i,j] # spin at site (i,j)
            Nb = config[(i+1)%N, j] + config[i,(j+1)%N] + config[(i-1)%N, j] + config[i,(j-1)%N] # Sum of the nearest neighbors; % takes care of the PBC
            energy += -J*Nb*s
    return energy/2

# Calculate next-to-nearest neighbours energy term for a given configuration
# This involves products of two spins, therefore it's an even coupling
def nnNbEnergy(config,J=1):
    energy = 0
    N = np.shape(config)[0]
    for i in range(N):
        for j in range(N):
            s = config[i,j] # spin at site (i,j)
            nNb = config[(i+1)%N, (j+1)%N] + config[(i-1)%N, (j+1)%N] + config[(i+1)%N, (j-1)%N] + config[(i-1)%N, (j-1)%N] # Sum of the next-to-nearest neighbors; % takes care of the PBC
            energy += -J*nNb*s
    return energy/2

# Calculate next-to-next-to-nearest neighbours energy term for a given configuration
# This involves products of two spins, therefore it's an even coupling
def nnnNbEnergy(config,J=1):
    energy = 0
    N = np.shape(config)[0]
    for i in range(N):
        for j in range(N):
            s = config[i,j] # spin at site (i,j)
            nnNb = config[(i+2)%N, j] + config[i,(j+2)%N] + config[(i-2)%N, j] + config[i,(j-2)%N] # Sum of the nearest neighbors; % takes care of the PBC
            energy += -J*nnNb*s
    return energy/2

# Calculate 2x2 square blocks/plaquettes energy term for a given configuration
def square2x2BlocksEnergy(config,J=1):
    energy = 0
    N = np.shape(config)[0]
    for i in range(N):
        for j in range(N):
            s = config[i,j] # spin at site (i,j)
            block = config[i,j] + config[(i+1)%N, j] + config[i,(j+1)%N] + config[(i+1)%N, (j+1)%N] # Sum of the spins in the 2x2 block; % takes care of the PBC
            energy += -J*block*s
    return energy

# Calculate external field energy term for a given configuration
def externalFieldEnergy(config,h=1):
    return -h*np.sum(config)

# Calculate the energy term due to 3-spins row interactions for a given configuration
def row3SpinsEnergy(config,J=1):
    energy = 0
    N = np.shape(config)[0]
    for i in range(N):
        for j in range(N):
            s1 = config[i,j] # spin at site (i,j)
            s2 = config[i,(j+1)%N] # spin at site (i,j+1)
            s3 = config[i,(j+2)%N] # spin at site (i,j+2)
            energy += -J*s1*s2*s3
    return energy/3

# Calculate the energy term due to 3-spins column interactions for a given configuration
def column3SpinsEnergy(config,J=1):
    energy = 0
    N = np.shape(config)[0]
    for i in range(N):
        for j in range(N):
            s1 = config[i,j] # spin at site (i,j)
            s2 = config[(i+1)%N, j] # spin at site (i+1,j)
            s3 = config[(i+2)%N, j] # spin at site (i+2,j)
            energy += -J*s1*s2*s3
    return energy/3

# Execute a single spin-flip MC move using the Metropolis algorithm
def MCMetropolisMove(T,config,params):
    # Monte Carlo move using Metropolis algorithm
    # Choose a random spin
    N = np.shape(config)[0]
    i = np.random.randint(N)
    j = np.random.randint(N)
    s = config[i,j]
    nb = config[(i+1)%N, j] + config[i,(j+1)%N] + config[(i-1)%N, j] + config[i,(j-1)%N] # sum of the spins of the nearest neighbors; % takes care of the PBC
    dE = 2*s*(params[0]*nb + params[1]) # change in energy if the spin is flipped
    if dE < 0:
        s *= -1
    elif np.random.random() < np.exp(-dE/T):
        s *= -1
    config[i,j] = s
    return config

# Execute a cluster-flip MC move using the Wolff algorithm
def MCWolffMove(T,config,J=1):
    # Choose a random spin
    N = np.shape(config)[0]
    i = np.random.randint(N)
    j = np.random.randint(N)
    # print('Random site:',i,j)
    s = config[i,j]
    # Generate the cluster
    cluster = [[i,j]]
    # print('Beginning cluster generation')
    cluster = WolffCluster(T,config,cluster,J)
    # print('Cluster of %d sites generated'%len(cluster))
    # print(cluster)
    # Flip the spins in the cluster
    for k,site in enumerate(cluster):
        config[site[0],site[1]] *= -1
        # print('Site %d with coordinates %d,%d flipped'%((k+1),site[0],site[1]))
        # print(config[site[0],site[1]])
    # plt.figure(figsize=(8,8))
    # plt.imshow(config, cmap='hot', interpolation='nearest')
    # plt.colorbar()
    return config

# Build the Wolff cluster starting from a random site
def WolffCluster(T,config,cluster,J=1):
    N = np.shape(config)[0]
    to_visit = cluster.copy()
    # print('\tSites to be visited:',to_visit)
    while len(to_visit)>0:
        newSites = []
        # Select a site to visit, i.e. a site belonging to the border of the cluster
        i = to_visit[0][0]
        j = to_visit[0][1]
        # print('Site to visit:',i,j)
        # Find its nearest neighbors
        nb = [[(i+1)%N, j], [i, (j+1)%N], [(i-1)%N, j], [i, (j-1)%N]]
        # print('Nearest neighbors:',nb)
        # Add them to the cluster according to the Wolff algorithm
        for n in nb:
            # print('\tNn to vist:',n)
            if n not in cluster and config[i,j] == config[n[0],n[1]] and np.random.random() < 1 - np.exp(-2*J/T):
                newSites.append(n)
                # print('\t\tSite added to cluster:',n)
        # print('\tSites to be visited before cluster update',to_visit)
        cluster += newSites
        # print('\tSites to be visited after cluster update',to_visit)
        # Remove the site from the list of sites to visit
        to_visit = to_visit[1:] + newSites
        # print('\tSites to be visited:',to_visit)
    return cluster

# Execute a cluster-flip MC move using the Wolff algorithm; version for usage with numba
@njit
def MCWolffMoveNumba(T,config,J=1):
    # Generate the cluster
    cluster = WolffClusterNumba(T,config,J)
    # Flip the spins
    config *= cluster
    return config

# Build the Wolff cluster starting from a random site; version for usage with numba
@njit
def WolffClusterNumba(T,config,J=1):
    N = np.shape(config)[0]
    cluster = np.ones((N,N), dtype = np.int8)
    visited = np.zeros((N,N), dtype = np.int8)
    toVisit = [[np.random.randint(N),np.random.randint(N)]]
    while len(toVisit)>0:
        i = toVisit[0][0]
        j = toVisit[0][1]
        if visited[i,j] == 0:
            visited[i,j] = 1
            nb = [[(i+1)%N, j], [i, (j+1)%N], [(i-1)%N, j], [i, (j-1)%N]]
            for n in nb:
                if config[i,j] == config[n[0],n[1]] and np.random.random() < 1 - np.exp(-2*J/T):
                    cluster[n[0],n[1]] = -1
                    toVisit.append(n)
        toVisit = toVisit[1:]
    return cluster

# Perform the RG decimation of the configuration
def decimation(config,Ndec):
    # Decimate the configuration by a factor of Ndec; to each block-spin thus generated assign the value of the majority of the spins in the block
    N = np.shape(config)[0]
    newconfig = np.zeros((N//Ndec,N//Ndec))
    for i in range(0,N,Ndec):
        ii = i//Ndec 
        for j in range(0,N,Ndec):
            jj = j//Ndec
            newconfig[ii,jj] = np.sign(np.sum(config[i:i+Ndec,j:j+Ndec]))
    return newconfig

# Calculate the covariances between two matrices with shape (nSamples,nVariables)
def mycov(x,y):
    nsamples = np.shape(x)[0]
    xCentered = x - np.mean(x,axis=0)
    yCentered = y - np.mean(y,axis=0)
    return np.dot(xCentered.T,yCentered)/(nsamples-1)


# Plot a configuration of the 2D Ising model
def spinConfPlot(config):
    fig, ax = plt.subplots(figsize=(8,8))
    cmap = plt.matplotlib.colors.ListedColormap(['blue', 'red'])
    im = ax.imshow(config, cmap=cmap)
    cbar = fig.colorbar(im, ax=ax, location='right', shrink=0.5)
    cbar.set_ticks([-1,1])
    ax.set_xticks([])
    ax.set_yticks([])
    return fig, ax

# Get the index of a simulation
def getSimIdx(sim):
    return int(sim.split('_')[-1].split('.')[0])

# Get the index of a simulation and pad it with zeros
def getSimIdxPadded(sim):
    intIdx = getSimIdx(sim)
    return '{:03}'.format(intIdx)
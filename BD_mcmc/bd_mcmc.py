import os

os.environ["OMP_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import corner
import emcee

from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import KDTree
from multiprocessing import Pool

def interpolated_mag(theta, mag_table):
    Teff, log_g, Z, d = theta
    
    # Find the nearest point in the grid
    kdtree = KDTree(mag_table[['Teff', 'log_g', 'Z']])
    dist, points = kdtree.query(theta[:3], 1)
    
    mag = mag_table.iloc[points][['F115W', 'F150W', 'F277W', 'F444W']].values
    
    # Convert fluxes to magnitudes, mag_table is AB magnitude at 10 pc
    Mag = mag + 5 * np.log10(d / 10)
    
    return Mag

def log_likelihood(theta, mag_table, y, yerr):
    model_m = interpolated_mag(theta, mag_table)
    return -0.5 * np.sum(((np.array(y) - np.array(model_m)) / np.array(yerr)) ** 2)  # likelihood function

def log_prior(theta):
    Teff, log_g, Z, d = theta
    if 400 < Teff < 2400 and 2.9 < log_g < 5.6 and 0.4 < Z < 1.6 and 1000 < d < 6000:  # set a reasonable range of parameters
        return 0.0
    return -np.inf

def log_probability(theta, mag_table, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):  # if the parameter is not reasonable, return -inf (0 possibility)
        return -np.inf
    return lp + log_likelihood(theta, mag_table, y, yerr)

file_dir = '/cluster/home/yuanchen/JWST/BD_mcmc/'

mag_table = pd.read_feather(file_dir+'interpolated_ABmag.feather')
observation = pd.read_csv(file_dir+'final.csv')

magnitudes = observation[['MAG_AUTO_F115W', 'MAG_AUTO_F150W', 'MAG_AUTO_F277W', 'MAG_AUTO_F444W']].values
magerr = observation[['MAGERR_AUTO_F115W', 'MAGERR_AUTO_F150W', 'MAGERR_AUTO_F277W', 'MAGERR_AUTO_F444W']].values
Mobs, Merr = lambda x: magnitudes[x-1], lambda x: magerr[x-1]

# Set up argument parser
parser = argparse.ArgumentParser(description='Run a Python script with SLURM.')
parser.add_argument('n', type=int, help='Set the value of n.')

# Parse the arguments
args = parser.parse_args()

# Set n based on the command-line argument
n = args.n

# set initial guess, please fit the prior range
theta = 1000, 3.75, 1, 3000
coefficients = np.array([500, 1.75, 0.6, 2000])
nwalkers = 100
ndim = len(theta)
pos = np.zeros((nwalkers, ndim))

# setup walkers, more wolkers, less iterations
for i in range(ndim):
    pos.T[i] = theta[i] + coefficients[i] * np.random.uniform(-1, 1, nwalkers)

# run the mcmc fitting
with Pool() as pool:
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability, pool=pool, args=(mag_table, Mobs(n), Merr(n))
    ) 
    sampler.run_mcmc(pos, 5000, progress=True)

# check convergent time
fig, axes = plt.subplots(4, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()       # get 32 values for each time (within 5000)
labels = ["Teff", "log_g", "Z", "d"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number")

N = str(n).zfill(2)
plt.savefig(file_dir+f'result2_v2/BD_{N}chain.png')

# refine results and drop data affected by initial guess
flat_samples = sampler.get_chain(discard=1000, thin=10, flat=True)

data = pd.DataFrame(flat_samples, columns=['Teff', 'log_g', '[M/H]', 'd'])

data.to_csv(file_dir+f'result2_v2/BD{N}_mcmc_alldata.csv', index=False)

# # Remove the burn-in period
# flat_samples = sampler.get_chain(discard=6000, thin=10, flat=True)
# # print(flat_samples.shape)

# for i in range(ndim):
#     mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
#     q = np.diff(mcmc)
#     print(mcmc[1], q[0], q[1])

# # Plot the chains
# fig, axes = plt.subplots(4, figsize=(10, 7), sharex=True)
# samples = sampler.get_chain()       # get 32 values for each time (within 5000)
# labels = ["Teff", "log_g", "Z", "d"]
# for i in range(ndim):
#     ax = axes[i]
#     ax.plot(samples[:, :, i], "k", alpha=0.3)
#     ax.set_xlim(0, len(samples))
#     ax.set_ylabel(labels[i])
#     ax.yaxis.set_label_coords(-0.1, 0.5)

# axes[-1].set_xlabel("step number")
# if n<9:
#     plt.savefig(file_dir+f'results/BD_0{n}chain.png')
# else:
#     plt.savefig(file_dir+f'results/BD_{n}chain.png')

# plot_ranges = [(840, 1000), (3.2, 6), (0.3, 1.6), (400, 800)]


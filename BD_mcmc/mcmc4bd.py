import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import astropy.units as u
import emcee
from scipy.interpolate import interp1d
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import KDTree

def interpolated_mag(theta, mag_table):
    
    ''' With the input parameters, find the nearest point in the grid and return the BD model magnitudes
    Returns:
    --------
    Mag: numpy array
        Magnitudes in F115W, F150W, F277W, F444W
        
    Parameters:
    -----------
    theta: numpy array
        Teff, log_g, Z, d
    mag_table: pandas DataFrame
        BD model magnitudes table
    ''' 
    
    Teff, log_g, Z, d = theta
    
    # Find the nearest point in the grid
    kdtree = KDTree(mag_table[['Teff','log_g','Z']])
    dist, points=kdtree.query(theta[:3],1)
    
    mag = mag_table.iloc[points][['F115W', 'F150W', 'F277W', 'F444W']].values
    
    # Convert fluxes to magnitudes, mag_table is AB magnitude at 10 pc
        # mag = log10(flux) * -2.5
        # log10(flux/d**2) = log10(flux) - 2*log10(d)
        # mag = -2.5*np.log10(flux) + 5*np.log10(d)
    Mag = mag + 5*np.log10(d/10)
    
    return Mag

def log_likelihood(theta, mag_table, y, yerr):
    Teff, log_g, Z, d = theta
    model_m = interpolated_mag(theta, mag_table)
    return -0.5 * np.sum(((np.array(y) - np.array(model_m))/ np.array(yerr)) ** 2 ) # likelihood function

def log_prior(theta):
    Teff, log_g, Z, d = theta
    if 400 < Teff < 2400 and 2.9 < log_g < 5.6 and 0.4 < Z < 1.6 and 9 < d < 3000: # set a reasonable range of parameters
        return 0.0
    return -np.inf

def log_probability(theta, mag_table, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp): # if the parameter is not reasonable, return -inf (0 possibility)
        return -np.inf
    return lp + log_likelihood(theta, mag_table, y, yerr) # set the probability as maximum likelihood 

def read_flux_table(flux_table):
    
    file = flux_table
    with open(file, 'r') as f:
        lines = f.readlines()
        
    data = []
    header = lines[6].split()
    for i in range(len(lines)-8):
        data.append([float(x) for x in lines[i+8].split()])
    
    data = pd.DataFrame(data, columns=header)
        
    return data
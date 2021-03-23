#!/usr/bin/env python3

#Default Modules:
import numpy as np
import pandas as pd
import scipy 
import matplotlib.pyplot as plt
import os, glob, sys
from tqdm import tqdm
from astropy import units as u

plt.style.use("/home/custom_style1.mplstyle")

#Other Modules:

from scipy.stats import cauchy

##### Author: Alex Polanski #####


if __name__ == "__main__":

    # input stuff here:
    
    if '-num' in sys.argv:
        p = sys.argv.index('-num')
        num = int(sys.argv[p+1])
    if '-samples' in sys.argv:
        p = sys.argv.index('-samples')
        samples = int(sys.argv[p+1])
    if '-upper' in sys.argv:
        p = sys.argv.index('-upper')
        upper = int(sys.argv[p+1])
    if '-lower' in sys.argv:
        p = sys.argv.index('-lower')
        lower = int(sys.argv[p+1])


    # Initiliaze the Rayleigh parameters with a uniform distribution
    
    x0s = np.random.uniform(lower,upper,num)
    
    # Loop through the values, draw from a Lorentz distribution and store the samples in a list
    
    rayleigh_samples = []

    for i in range(num):
        sample = np.random.rayleigh(x0s[i],size=samples)

        rayleigh_samples.append(sample)
    

    rayleigh_samples = np.array(rayleigh_samples)

    file_name = 'rayleigh-samples_' + f'{num}-experiments_' + f'{samples}-samples' + '.csv'
    
    np.savetxt(f'./{file_name}', rayleigh_samples, delimiter=',')   

    

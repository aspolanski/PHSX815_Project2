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

##### Author: Alex Polanski #####

def rayleigh(x, sig):

    # Probability dist for Rayleigh

    coef = x / sig**2

    ex = np.exp(-x**2 / 2* sig**2)

    return( coef * ex )



if __name__ == "__main__":

    # Input stuff

    
    if '-file' in sys.argv:
        p = sys.argv.index('-file')
        file_loc = sys.argv[p+1]

    if '-fixed_sigma' in sys.argv:
        p = sys.argv.index('-fixed_sigma')
        sigma = float(sys.argv[p+1])

    samples = np.loadtxt(file_loc,delimiter=',')

    #print(samples)
    llr_posterior = []
    llr_fixed = []
    
    weights = np.ones_like(samples[:,0])/len(samples[:,0])
    weights = np.ones_like(samples.flatten())/len(samples.flatten())
    hist, bins = np.histogram(samples.flatten(), weights=weights, bins=50)


    # calculate the probabilities for this distribution using 50 bins

    for i in range(np.shape(samples)[0]):
        
        hist_idx = np.digitize(samples[i],bins[:-1]) - 1

        probs = hist[hist_idx] # these are the probabilities for the native distribution
        
        probs_rayleigh = rayleigh(samples[i], sigma) # these are probabilites that the samples come from a fixed Rayleigh distribtion
    
        # Generate samples from the fixed Rayleigh distribution with the same paramter as above

        samples_null = np.random.rayleigh(sigma,np.shape(samples)[1])

        # Get the probabilities for these samples

        prob_null = hist[np.digitize(samples_null,bins[:-1]) - 1]
    
        prob_null_fixed = rayleigh(samples_null, sigma)

        # Calculate the log likelihood ratios

        llr = np.sum(np.log(probs_rayleigh/probs))
        
        llr2 = np.sum(np.log(prob_null_fixed/prob_null))

        llr_posterior.append(llr)
        llr_fixed.append(llr2)


    # Calculate lambda alpha

    sorted_llr_post = np.sort(llr_posterior)

    idx = np.rint( 0.95 * len(sorted_llr_post) )

    lam_alpha = sorted_llr_post[int(idx)]

    # Calculate beta
    sorted_llr_fixed = np.sort(llr_fixed)

    absolute_val_array = np.abs(sorted_llr_fixed - lam_alpha)

    smallest_difference_index = absolute_val_array.argmin()

    beta = smallest_difference_index/len(sorted_llr_fixed)

    print(f" The false negative rate is {beta}")

    weights_post = np.ones_like(llr_posterior)/len(llr_posterior)
    weights_fixed = np.ones_like(llr_fixed)/len(llr_fixed)

    fig, ax = plt.subplots()
    
    ax.hist(llr_posterior, weights=weights_post,bins=50,label='Null')
    ax.hist(llr_fixed, weights=weights_fixed,bins=50,alpha=0.5, label='Alternative')
    ax.vlines(lam_alpha,0.0,1.0, linestyles='dashed',colors='black', label = r'$\lambda_{\alpha}$')
    ax.set_yscale('log')
    ax.set_xlabel('$\lambda = \log{L(H1)/L(H0)}$')
    ax.set_ylabel('Probability')
    ax.text(0.01,0.75, r'$\alpha = 0.05$' + '\n' + r'$\beta$' + f'= {beta}', transform=ax.transAxes, fontsize=29) 
    ax.legend(loc='upper left')
    ax.set_title(f'{np.shape(samples)[1]} samples/experiment')
    plt.tight_layout()
    plt.show()

    

    



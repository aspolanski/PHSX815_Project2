# PHSX815_Project2: Weird Rayleigh Distributions

This package compares samples drawn from a Rayleigh distribution with a fixed parameter to samples drawn from a Rayleigh distribution with parameter drawn from a uniform distribution.

## Descriptions of Included Scripts:

* *rayleigh.py*: This script generates samples drawn from a Rayleigh distribution while first determining the parameter by drawing it from a uniform distribution. The samples are saved as .csv files. User inputs are the number of experiments (-num), the number of samples per experiment (-samples), and the upper/lower bounds of the uniform distribution (-upper/-lower).

* *rayleigh_analysis.py*: This script analyzes the results of the previous by loading in the .csv and finding the probability of each drawing each of those samples. These are then compared via log-likelihood ratio to the probability of getting those samples assuming a fixed-parameter Rayleigh distribution. This is repeated with samples drawn from that fixed-parameter Rayleigh distribution. The output is a plot showing the overlap of the two LLR distributions and the resulting false-negative rate. User inputs are the location of the .csv file (-file) and the value of the fixed parameter (-fixed_sigma).

## Usage 

To generate samples:

```python
python rayleigh -num 10000 -samples 2 -upper 2 -lower 5
```

To analyze:

```python
python rayleigh_analysis -file rayleigh-samples_10000-experiments_100-samples.csv -fixed_sigma 1.5
``` 


## Dependencies

This code requires:

* Python 3.7.3
* Scipy v1.6.0
* Numpy v1.19.2
* MatplotLib v3.3.2
* TQDM v4.56.0



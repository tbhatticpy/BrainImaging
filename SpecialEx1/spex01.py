# -*- coding: utf-8 -*-


"""
This special exercise will expand the analysis of the 4 student EEG datasets begun in exercise 5. 

You will need the "cross_functions" module that was provided for ex. 5.

If you want, you can deliver the solutions for ex. 5 and this special python exercise
 together in a single script or notebook.

The total amount of points from this part is 50 + up to 5 bonus points.
50 points can be gotten from the essay task, the maximum total score is 100.

Completing all special exercises can give you a bonus on the exam grade.
In order to get the bonus, you'll need to pass the exam by itself and also
 get at least a mean of 50% completion from the 3 special exercises together.
That means, completing only one of the special exercises will not do any good, 
 and completing only two can only give a maximum bonus of 0.33 on the exam grade.
Completing all 3 special exercises with 100% will improve the exam grade by 1.0.

Packages to install: 
    MNE
    fooof - for fitting peaks and 1/f slopes
    statsmodels
    functools 
    
Message if you have problems installing any of these packages.

"""





#%% Import modules and functions

from fooof import FOOOF, FOOOFGroup
from fooof import Bands
from fooof.analysis import get_band_peak_fg, get_band_peak_fm
from mne.preprocessing import ICA
from mne.time_frequency import tfr_array_morlet

import numpy as np
import matplotlib.pyplot as plt
import mne

plt.style.use('default')

import sys
sys.path.append('L:/nttk-data3/palva/Felix/Python37/__exercises/ex05/')
sys.path.append('L:/nttk-data3/palva/Felix/Python37/Utilities/')

from cross_functions import cplv, dfa, get_dfa_parameters

# Define canonical frequency bands
bands = Bands({'theta' : [4, 8],
               'alpha' : [8, 15],
               'beta' : [15, 30]})  








#%% Task 1 (8 points)

# We'll now compute the scaling exponents of the time series' long-range temporal correlations.
# We'll use detrended fluctuation analysis as described in Hardstone et al., 2012, Front. Physiol.
# Use the get_dfa_parameters() function to get the appropiate window lengths (for 4-minute time series) for DFA analysis.
# Use the dfa() function to fit DFA from the amplitudes of the alpha band time series.
# The function returns four arrays, representing fluctuations, slopes, dfa_exponents, and residuals.
# Do the DFA for all 4-min eyes-closed periods and 4-min eyes-open periods separately.
# Save the individual channels' exponents.
# Report for each subject and both eyes-open and -closed data, the mean (over all channels) DFA exponents.








#%% Task 2 (8 points)

# For one of the subject datasets, do the following, for 4 different channels:
# Make appropiate plots of the fluctuations as a function of window length,
# and add the DFA fit.





    
    

#%% Task 3 (10 points)

# Now, use FOOOFGroup to make FOOOF fits for all channels individually in all 8 spectra. 
# Use the same parameters as in ex. 5 task 4.
# Extract the alpha peak frequency and each 1/f slope for each channel.
# For all 8 spectra, take and report the mean over channels.







#%% Task 4 (4 points)

# Compare the results from Task 3 with the values from exercise 5 task 4 
# (where you first averaged the spectra and then fitted FOOOF)
# Do you see any differences? What could be the reasons?



#%% Task 5 (8 points) 

# Now, let's look at the relationship between synchrony and LRTC scaling exponents.
# Make scatterplots of each channel's mean alpha-band phase synchrony (using the better metric)
# and alpha-band DFA exponent for each subject and condition (eyes-open & eyes-closed).
# Add linear fits and compute the correlations.

  
    
    
#%% Task 6 (4 points)

# If the relationship between synchrony and LRTC exponents that has been proposed in Fusca et al., 2023, Nat Comm.
# within the "critical brain dynamics" framework is true, which of these 8 analyzed states might be subcritical, critical, or supercritical?

# Or do you have any other possible interpretations? (not neccessary, but feel free to speculate here)

# Note for the EEG participants: 
# The results from these findings are not to be taken as evidence of any clinical condition.
# Remember that these are estimates, that we don't fully understand critical 
# dynamics and that characteristics of critical dynamics, and most likely 
# also the parameters that are best for healthy critical brain function, vary between individuals.
    
    
    
#%% Task 7 (8 points)

# Now let's also look at the relationship between 1/f slopes and alpha-band scaling exponents.
# Make scatterplots of slopes vs DFA exponents, add linear fits and compute correlations.
# If you earlier removed bad channels from the data when you computed the slopes, remove the same channels also from the DFA results.





#%% Task 8 (up to 5 bonus points)   

# Describe the relationships that you observe between 1/f slopes and alpha-band LRTC exponents.
# Do you have any idea what these relationships might possibly indicate for (critical) brain dynamics?
# This is a novel research question, so feel free to speculate.
# You can also search on Google Scholar if you can find any articles
# dealing with this question, or related ones. If you base your ideas on articles, cite them.
# There is not really a "correct answer" here.
# The main point is to think about what a relationship between these observables might mean.





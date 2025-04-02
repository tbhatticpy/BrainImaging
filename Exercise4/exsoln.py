
# -*- coding: utf-8 -*-
"""

"""
import mne
import numpy as np
import matplotlib.pyplot as plt

from numpy import random
from scipy import signal


plt.style.use('default')

#%% Task 1 (4 points)

# Create some oscillation time series as simple sine or cosine waves.
# Use frequencies 6, 10, and 45 Hz.
# Length should be 2 sec and sampling frequency 1000 Hz.
# Plot them individually and their sum

fs = 1000

times = np.linspace(0, 2, fs, endpoint=False)
alpha = np.sin(10  * np.pi * times)
theta = np.cos( 6  * np.pi * times)
gamma = np.sin(45  * np.pi * times)

sum_data = alpha+theta+gamma

fig, axes = plt.subplots(4,1,figsize=[12,11])
axes[0].plot(theta,'b');axes[0].set_title('Theta')
axes[1].plot(alpha,'r');axes[1].set_title('Alpha')
axes[2].plot(gamma,'g');axes[2].set_title('Gamma')
axes[3].plot(sum_data,'k');axes[3].set_title('Sum')
axes[3].set_xlabel("Time (ms)")
plt.show()

#%% Task 2 (4 points)

# Now, get the PSD (power spectral density) of the summed signal with the Welch method.
# Implementations can be found e.g. in scipy.signal or MNE.
# Play around with the nperseg parameter until you can see all oscillations as peaks.
# What do you notice?

f, psd = signal.welch(sum_data, fs, nperseg=2000)

fig, axes = plt.subplots(2,1)
axes[0].plot(times, sum_data)
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Amplitude")
axes[0].set_title("Simulated time series")

axes[1].loglog(f, psd)
axes[1].set_xlabel("Frequency (Hz)")
axes[1].set_ylabel("Power (V^2/Hz)")
axes[1].set_title("Welch PSD")

plt.tight_layout()
plt.show()

print("The spectral resolution increases for higher values of nperseg.")
print("Also, the spectral resolution is generally better for higher frequencies.")




#%% Task 3 (5 points)

# Now, let's create some white noise (several ways to do this is numpy or scipy)
# Again, use f_samp = 1000 Hz and length of 2 sec.
# Plot both the signal and its PSD (computed with Welch).

fs     = 1000 # Sampling frequency
length = 20 # Time in seconds

t, dt = np.linspace(0, length, fs*length, retstep=True)
data = random.normal(size=fs*length)

f, psd = signal.welch(data, fs, nperseg=256*4)

fig, axes = plt.subplots(2,1)
axes[0].plot(t, data)
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Amplitude")
axes[0].set_title("Simulated white noise")
axes[0].set_xlim([6, 10])

axes[1].loglog(f, psd)
axes[1].set_xlabel("Frequency (Hz)")
axes[1].set_ylabel("Power (V^2/Hz)")
axes[1].set_title("Welch PSD")

fig.tight_layout()
fig.show()



#%% Task 4 (5 points)

# Create a Morlet wavelet (e.g. with scipy.signal or PyWavelets) at 10 Hz.
# Filter the white noise data by convolution with this wavelet.
# Plot 2 seconds of both real and imaginary part of the filtered time series on the same plot.
# Compute and plot the PSD for the real part of the filtered time series. 
# What do you notice when you change the width parameter of the wavelet?

freq = 10
w = 9

widths = w*fs / (2*freq*np.pi)
wavelet_transform = signal.cwt(data, signal.morlet2, [widths], w=w)
sim_data =  data + wavelet_transform.real[0]


f, psd = signal.welch(wavelet_transform.imag, fs, nperseg=256*4)

fig, axes = plt.subplots(2,1)
axes[0].plot(t, wavelet_transform.real[0])
axes[0].plot(t, wavelet_transform.imag[0])
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Amplitude")
axes[0].set_title("Simulated brain activity")
axes[0].set_xlim([6, 8])

axes[1].loglog(f, psd[0])
axes[1].set_xlabel("Frequency (Hz)")
axes[1].set_ylabel("Power (V^2/Hz)")
axes[1].set_title("Welch PSD")

fig.tight_layout()
fig.show()

print("The spectral resolution increases with the width parameter.")


#%% Task 5 (4 bonus points)

# Let's create a time-frequency plot.
# Create an array of 50 (ideally log-spaced) frequencies from 1 to 30 Hz.
# Create another white noise time series and add it to your artifical 10-Hz 
# time series from the previous task.
# Compute the convolution of the summed signal with the array of frequencies.
# (you can do this at once e.g. with signal.cwt)
# You should get a 2D data array.
# Use this to create a time-frequency plot of the time series amplitude. 
# Label your axes correctly.
# Describe what you see.

w = 7
freqs = np.geomspace(1, 30, 50) # logarithmically evenly spaced frequencies

widths = w*fs / (2*freqs*np.pi)
cwtm = signal.cwt(data+sim_data, signal.morlet2, widths, w=w)

fig, ax = plt.subplots()
ax.pcolormesh(t, freqs, np.abs(cwtm))
ax.set_xlabel("Time (s)")
ax.set_ylabel("Frequency (Hz)")
ax.set_title("Time-frequency plot of simulated data")



print("The intensity is highest around 10 Hz and oscillates also on slower time scales (few seconds).") 
print("The actual peak frequency also fluctuates over time.")



#%% Task 6 (5 points)

 
# For this following tasks, we will use the MNE library.
# If you have problems installing and/or using MNE library, please let me know.
# All the following tasks here can be done using MNE functions.
# Use the online documentation at mne.tools.stable to figure out how to use them.
# You can search for functions directly on the website; also googling them often works.

# We are gonna look at two EEG data files that were recorded last week at Galuske lab from students of this course.
# Download the files from Moodle.
# Load the two raw student data files into variables raw1, raw2 with mne.io.read_brainvision()
# Use the preload option.
# Get the sampling frequency and the indices of the channels marked as 'misc' from the info metadata.
# Plot the power sepctra in the range to 120 Hz. 
# What do you notice regarding peaks, noise? 
# How do the two datasets differ?


file1 = 'g:/My Drive/TU Darmstadt 2024/Galuske/EEG data/VP01/VP1_BrainImaging.vhdr'
file2 = 'g:/My Drive/TU Darmstadt 2024/Galuske/EEG data/VP02/VP2_BrainImaging.vhdr'

raw1 = mne.io.read_raw_brainvision(file1,preload=True)
raw2 = mne.io.read_raw_brainvision(file2,preload=True)

fig1 = raw1.compute_psd(fmax=120).plot()
fig2 = raw2.compute_psd(fmax=120).plot()

misc1_inds = mne.channel_indices_by_type(raw1.info)['misc']
misc2_inds = mne.channel_indices_by_type(raw2.info)['misc']

sfreq = raw1.info['sfreq']


print("In VP1, there is a strong alpha peak at around 9 Hz.")
print("In VP2, there are two peaks at around 8 and 11 Hz.")
print("In VP2, line noise is visible at 50 Hz in most channels.")
print("In VP1, there are more peaks in the gamma range.")

print("In both subjects, there is noise visible (elevated PSD) in gamma band for several channels.")
print("The spatial placement of electrodes on the head differs between the two subjects.")




#%% Task 7 (2 points)

# For each dataset, plot the sensor locations with names.

raw1.plot_sensors(show_names=True)
raw2.plot_sensors(show_names=True)


#%% Task 8 (4 points)

# Then, use .plot() on your raw objects to open interactive plots. 
# If the plot isn't interactive, try changing your matplotlib backend.
# https://matplotlib.org/stable/users/explain/figure/backends.html
# In an interactive plot, you can scroll through the whole time series with arrow buttons.
# If you can't get interactive plots to work, you can also specify start and 
# duration for static plots with the plot() function.
# 
# In which subject can you easily identify periods with eyes open and eyes closed?
# Why?
# Make a plot or screenshot of 10 sec eyes-open brain activity, and one of 10 sec eyes-closed.

raw1.plot()
raw1.plot(highpass=1,lowpass=99)
raw2.plot()
raw2.plot(highpass=1,lowpass=99)

print("In VP1, eye blinks are very prominent in frontal channels the first 6 minutes, indicating open eyes.")
print("In VP1, alpha rhythm is more pronounced in occipital channels in the later (eyes-closed) part of the recording.")


raw1.plot(start=30,  duration=10, highpass=1, lowpass=99) 
raw1.plot(start=400, duration=10, highpass=1, lowpass=99)  




#%% Task 9 (4 points)

# Make copies of the raw data objects.
# To these copies, apply a notch filter to remove line noise, 
# and bandpass-filter them in the range 1 to 100 Hz.
# Plot the power spectra of the filtered data and describe what has changed.

filt1 = raw1.copy().notch_filter(50).filter(l_freq=1,h_freq=100)
filt2 = raw2.copy().notch_filter(50).filter(l_freq=1,h_freq=100)

fig1n = filt1.compute_psd(fmax=120).plot()
fig2n = filt2.compute_psd(fmax=120).plot()

print("Both spectra now have a downward peak at 50 Hz, indicating PSD content at this frequency was removed.")
print("Both spectra now decay in power at frequencies over 100 Hz.")



#%% Task 10 (3 points)

# Plot and inspect the filtered time series. What has changed?

filt1.plot()
filt2.plot()

print("The slow drift (<1Hz) has been removed.")
print("Filtered series look smoother due to removal of HF (> 100 Hz) noise.")



#%% Task 11 (4 bonus points)

# Identify for one subject periods in the data when they had their eyes closed and open.
# Plot power spectra for these periods.
# What difference is observable in the power spectra? 
# What is this effect called?

filt1.compute_psd(fmax=120,tmax=360).plot()
filt1.compute_psd(fmax=120,tmin=360,tmax=720).plot()

print("In VP1, the alpha band power is much higher during eyes-closed resting state.")
print("This is called the Berger effect.")


#%% Task 12 (4 points)

# Now let's look at some typical artefacts in EEG (or MEG) data.
# For the first subject, plot ECG and EOG events as described on 
# https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html
# Where are these artifacts mostly located?

from mne.preprocessing import ICA, create_ecg_epochs, create_eog_epochs

ecg_evoked = create_ecg_epochs(filt1,ch_name='ECG').average()
ecg_evoked.apply_baseline(baseline=(None, -0.2))
ecg_evoked.plot_joint()

eog_evoked = create_eog_epochs(filt1,ch_name='Fp1').average()
eog_evoked.apply_baseline(baseline=(None, -0.2))
eog_evoked.plot_joint()


print("The hearbeat-related artifact is located in left posterior region mainly.")
print("The eyeblink-related artifact is located in the frontal regions/electrodes, close to the eyes.")



#%% Task 13 (10 points)

# Following the instruction on the same tutorial page,
# compute ICA with 16 components and fit to the filtered data of subject 1
# This may take a while depending on your computer.
# Plot the components and their source time series (but applied to the filtered data)
# Which components look obviously like they represent artefacts, and what kind?
# Exclude the artifactual components and apply the ICA to a copy of the raw data.
# Did the ICA do what it was supposed to?

ica = ICA(n_components=16, max_iter="auto", random_state=97)
ica.fit(filt1)
ica

ica.plot_sources(filt1, show_scrollbars=False)

ica.plot_components()


print("The first component (ICA000) is frontal and represents eye-blinks.")
print("The second component is frontal, but as a left-right dipole and represents eye movements.")
print("Although there was some ECG-related signal visible with create_ecg_epochs, there is no obvious ECG-related component here.")
print("While ICA005 has a spatial distribution that would be typical of ECG-artifacts, no heartbeat peaks are visible on its time series.")


ica.exclude=[0,1]
reconst_raw1 = raw1.copy()
ica.apply(reconst_raw1)
reconst_raw1

raw1.plot(highpass=1,lowpass=99)
reconst_raw1.plot(highpass=1,lowpass=99)

print("After reconstruction, we can see that ICA has reduced eye-blink and -movement artefacts.")

"""










#%%


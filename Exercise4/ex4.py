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

sampling_freq = 1000
duration = 2

t =  np.linspace(0,duration, sampling_freq, endpoint=False)
#print(t.shape)
alpha = np.sin(10  * np.pi * t)
theta = np.cos( 6  * np.pi * t)
gamma = np.sin(45  * np.pi * t)
sine_oscillations = alpha+theta+gamma

fig, axes = plt.subplots(4,1,figsize=[12,11])
axes[0].plot(theta,'b');axes[0].set_title('Theta')
axes[1].plot(alpha,'r');axes[1].set_title('Alpha')
axes[2].plot(gamma,'g');axes[2].set_title('Gamma')
axes[3].plot(sine_oscillations,'k');axes[3].set_title('Sum')
axes[3].set_xlabel("Time (ms)")
plt.show()


#%% Task 2 (4 points)

# Now, get the PSD (power spectral density) of the summed signal with the Welch method.
# Implementations can be found e.g. in scipy.signal or MNE.
# Play around with the nperseg parameter until you can see all oscillations as peaks.
# What do you notice?

#nperseg = 256
#nperseg = 512
nperseg = 2000
freqs, psd = signal.welch(sine_oscillations,sampling_freq,nperseg=nperseg)
plt.loglog(freqs, psd )
plt.title(f'Power Spectral Density (PSD) using Welch Method (nperseg = {nperseg})')
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD')
plt.grid()
plt.show()

peaks, _ = signal.find_peaks(psd, height=0)
peak_freqs = freqs[peaks]
print("Frequencies of the peaks (Hz):", peak_freqs)

#Better frequency resolution is obtained using higher values of nperseg. 1024 values a decent estimate of the frequencies of the peaks in the PSD.
#When I run the code with nperseg = 2000, which is the max allowed value for this signal length, it gives a bunch of additonal peaks. This is possibly because of the
#noise artefacts in the signal at such a high resolution.

#%% Task 3 (5 points)

# Now, let's create some white noise (several ways to do this is numpy or scipy)
# Again, use f_samp = 1000 Hz and length of 2 sec.
# Plot both the signal and its PSD (computed with Welch).

np.random.seed(42)

white_noise = np.random.normal(0,1, int(duration*sampling_freq))

plt.figure(figsize=(10, 6))
plt.plot(t, white_noise)
plt.title('White Noise Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid()
plt.show()

freqs_noise, psd_noise = signal.welch(white_noise, fs=sampling_freq, nperseg=nperseg)

plt.figure(figsize=(10, 6))
plt.semilogy(freqs_noise, psd_noise)
plt.title('PSD of White Noise')
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD')
plt.grid()
plt.show()

#%% Task 4 (5 points)

# Create a Morlet wavelet (e.g. with scipy.signal or PyWavelets) at 10 Hz.
# Filter the white noise data by convolution with this wavelet.
# Plot 2 seconds of both real and imaginary part of the filtered time series on the same plot.
# Compute and plot the PSD for the real part of the filtered time series. 
# What do you notice when you change the width parameter of the wavelet?

center_freq = 10
width = 5
#scaling factor s should be 2 according to the formula f = 2*s*w*r / M in the documentation
M = int(duration * sampling_freq)
s = (center_freq * M) / (2 * width * sampling_freq)
print(s)
morlet_wavelet = signal.morlet(M=M, w=width, s = s)

filtered_signal = np.convolve(white_noise, morlet_wavelet, mode='same')

real_part = np.real(filtered_signal)
imag_part = np.imag(filtered_signal)

plt.figure(figsize=(10, 6))
plt.plot(t, real_part, label='Real Part')
plt.plot(t, imag_part, label='Imaginary Part')
plt.title('Filtered Signal (Real and Imaginary Parts)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()
plt.show()

freqs_filtered, psd_filtered = signal.welch(real_part, fs=sampling_freq, nperseg=nperseg)

plt.figure(figsize=(10, 6))
plt.semilogy(freqs_filtered, psd_filtered)
plt.title('PSD of the Real Part of the Filtered Signal')
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD')
plt.grid()
plt.show()

peak_index = np.argmax(psd_filtered)
peak_frequency = freqs_filtered[peak_index]

# Print the peak frequency
print(f"The peak frequency in the PSD is approximately: {peak_frequency:.2f} Hz")

#Changing the width controls the trade-off between time and frequency resolution.
#Small width -> less frequency resolution but narrow time domain wavelet
#Larger width -> Sharper peak at center frequency but wider wavelet in time domain

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

log_spaced_frequencies = np.logspace(np.log10(1), np.log10(30), 50)

np.random.seed(43)
new_white_noise = np.random.normal(0, 1, int(duration * sampling_freq))
summed_signal = real_part + new_white_noise

wavelet_widths = np.round(sampling_freq / (log_spaced_frequencies * 2 * np.pi)).astype(int)
cwt_result = signal.cwt(summed_signal, signal.morlet2, wavelet_widths)
amplitude = np.abs(cwt_result)

plt.figure(figsize=(12, 6))
plt.imshow(
    amplitude,
    aspect='auto',
    extent=[0, duration, log_spaced_frequencies[-1], log_spaced_frequencies[0]],
    cmap='viridis'
)
plt.colorbar(label='Amplitude')
plt.title('Time-Frequency Representation of the Summed Signal')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.gca().invert_yaxis()
plt.show()


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

raw1_path = 'EEG data_1-4/VP01/VP1_BrainImaging.vhdr'
raw2_path = 'EEG data_1-4/VP02/VP2_BrainImaging.vhdr'
raw1 = mne.io.read_raw_brainvision(raw1_path, preload=True, verbose=False)
raw2 = mne.io.read_raw_brainvision(raw2_path, preload=True, verbose=False)
#print(raw1.info)
#print(raw2.info)

sfreq1 = raw1.info['sfreq']
sfreq2 = raw2.info['sfreq']
misc_channels1 = mne.pick_types(raw1.info, misc=True)
misc_channels2 = mne.pick_types(raw2.info, misc=True)

print(f"Dataset 1: Sampling frequency = {sfreq1} Hz, Misc channels = {misc_channels1}")
print(f"Dataset 2: Sampling frequency = {sfreq2} Hz, Misc channels = {misc_channels2}")

raw1.compute_psd(fmax=120, verbose=False).plot(show=False)
plt.title('Power Spectra - Dataset 1')
plt.show(block=True)

raw2.compute_psd(fmax=120, verbose=False).plot(show=False)
plt.title('Power Spectra - Dataset 2')
plt.show(block=True)

#Both datasets have clear peaks at lower frequencies and low power at gamma+ freqs.
#Dataset 1 has stronger peaks. Although it shows more noise at higher frequencies than Dataset 2

# %% Task 7 (2 points)

# For each dataset, plot the sensor locations with names.
raw1.plot_sensors(show_names=True, show=False)
plt.title('Sensor Locations - Dataset 1')
plt.show(block=True)

raw2.plot_sensors(show_names=True, show=False)
plt.title('Sensor Locations - Dataset 2')
plt.show(block=True)

# %% Task 8 (4 points)

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


raw1.plot(show=False)
plt.show(block=True)

raw2.plot(show=False)
plt.show(block=True)


#In the interactive plots, looking at the occipital (subject 1) and parietal (subject 2) channels helps identify eye open and eye closed activity
#I believe it is rather easier to identify in the first subject due to the presence of occipital channels.
#I have made the following function to plot power spectra of both subjects for alpha frequencies. Higher activity in alpha freq means eye-closed period.
#For subject 1: 400-800 seconds is eye closed period.
#For subject 2: 350-600 seconds seem to be the eye closed period.


def plot_alpha_power(raw, channels, alpha_band=(8, 12), window_length=2, step_size=0.5, title=None):
    with mne.utils.use_log_level("error"):
        picks = mne.pick_channels(raw.info['ch_names'], include=channels)

        sfreq = raw.info['sfreq']
        window_samples = int(window_length * sfreq)
        step_samples = int(step_size * sfreq)
        n_samples = raw.n_times
        n_windows = (n_samples - window_samples) // step_samples + 1

        alpha_power = []
        for i in range(n_windows):
            start = i * step_samples
            stop = start + window_samples
            data = raw.get_data(picks=picks, start=start, stop=stop)
            psd, freqs = mne.time_frequency.psd_array_welch(data, sfreq=sfreq, fmin=alpha_band[0], fmax=alpha_band[1], n_overlap=0)
            alpha_power.append(psd.mean())

    time_axis = np.arange(0, n_windows) * step_size

    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, alpha_power, label=f'Alpha Power ({alpha_band[0]}–{alpha_band[1]} Hz)')
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Power')
    plt.legend()
    plt.grid()
    plt.show()

alpha_band = (8, 12)

occipital_channels = ['O1', 'O2', 'Oz']
plot_alpha_power(raw1, channels=occipital_channels, alpha_band=alpha_band, title = 'Alpha Power Spectra - Subject 1')

parietal_channels = ['P5', 'P1', 'P2', 'P6', 'POz']
plot_alpha_power(raw2, channels=parietal_channels, alpha_band=alpha_band, title = 'Alpha Power Spectra - Subject 2')


def plot_10s_segment(raw, start_time, title):
    raw.plot(start=start_time, duration=10, title=title, show=False)
    plt.show(block=True)

eye_open = 100
eye_closed = 500

#subject 1
plot_10s_segment(raw1, eye_open, title='Eyes-Open Brain Activity - Subject 1')
plot_10s_segment(raw1, eye_closed, title='Eyes-Closed Brain Activity - Subject 1')
#subject 2
plot_10s_segment(raw2, eye_open, title='Eyes-Open Brain Activity - Subject 2')
plot_10s_segment(raw2, eye_closed, title='Eyes-Closed Brain Activity - Subject 2')


# %% Task 9 (4 points)

# Make copies of the raw data objects.
# To these copies, apply a notch filter to remove line noise,
# and bandpass-filter them in the range 1 to 100 Hz.
# Plot the power spectra of the filtered data and describe what has changed.
def filter_psd(raw, notch_freq=50, bandpass_range=(1, 100), title=None):
    raw_filtered = raw.copy()
    raw_filtered.notch_filter(freqs=notch_freq, verbose=False)
    raw_filtered.filter(l_freq=bandpass_range[0], h_freq=bandpass_range[1], verbose=False)
    psd = raw_filtered.compute_psd(fmax=100, verbose=False)
    psd.plot(show=False)
    plt.title(title)
    plt.tight_layout()
    plt.show(block=True)
    return raw_filtered

raw1_filtered = filter_psd(raw1, title="Power Spectrum - Filtered Subject 1")

raw2_filtered = filter_psd(raw2, title="Power Spectrum - Filtered Subject 2")



# %% Task 10 (3 points)

# Plot and inspect the filtered time series. What has changed?
raw1_filtered.plot(start=100,duration=10,title='Filtered Time series - Subject 1', show = False)
plt.show(block=True)
raw2_filtered.plot(start=100,duration=10,title='Filtered Time series - Subject 2', show = False)
plt.show(block=True)


#With the removal of the line noise, the sharp oscillations around 50 Hz have been removed in the filtered time series.
#Slow frequency drift seems to be removed and filtered series shows smoother waveform due to the bandpass filter.
#Neural oscillations are not only intact but seem rather clearer.


# %% Task 11 (4 bonus points)

# Identify for one subject periods in the data when they had their eyes closed and open.
# Plot power spectra for these periods.
# What difference is observable in the power spectra?
# What is this effect called?

sfreq = raw1.info['sfreq']

eyes_closed_data = raw1.copy().crop(tmin=500, tmax=510).get_data()
psd_closed, freqs_closed = mne.time_frequency.psd_array_welch(eyes_closed_data,sfreq, fmax=50, verbose=False)

eyes_open_data = raw1.copy().crop(tmin=100, tmax=110).get_data()
psd_open, freqs_open = mne.time_frequency.psd_array_welch(eyes_open_data,sfreq, fmax=50, verbose=False)

psd_closed_mean = psd_closed.mean(axis=0)
psd_open_mean = psd_open.mean(axis=0)

plt.figure(figsize=(10, 6))
plt.semilogy(freqs_closed, psd_closed_mean, label='Eyes Closed', alpha=0.8)
plt.semilogy(freqs_open, psd_open_mean, label='Eyes Open', alpha=0.8)
plt.title('Power Spectra - Eyes Closed vs Eyes Open')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density')
plt.legend()
plt.grid()
plt.show()


#For eyes closed, we can see a strong peak at 8 Hz (in Alpha range 8-12 Hz).
#For eyes open, we can see some reduction in the Alpha power.
#This is known as Alpha Blocking, where visual stimuli reduce alpha rhythm while eyes are open.


# %% Task 12 (4 points)

# Now let's look at some typical artefacts in EEG (or MEG) data.
# For the first subject, plot ECG and EOG events as described on
# https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html
# Where are these artifacts mostly located?

raw1_filtered = mne.set_bipolar_reference(
    raw1_filtered, anode='Fp1', cathode='Fp2', ch_name='EOG_simulated'
)

ecg_epochs = mne.preprocessing.create_ecg_epochs(raw1_filtered, ch_name='ECG')
ecg_evoked = ecg_epochs.average()
ecg_evoked.apply_baseline(baseline=(None, -0.2))
ecg_evoked.plot_joint(title="ECG Artifacts")

eog_epochs = mne.preprocessing.create_eog_epochs(raw1_filtered, ch_name='EOG_simulated')
eog_evoked = eog_epochs.average()
eog_evoked.apply_baseline(baseline=(None, -0.2))
eog_evoked.plot_joint(title="Simulated EOG Artifacts")


#The EOG artifacts from eye blinking are present in the frontal regions.
#The ECG artifacts, mainly from heartbeat, are prominent in the central and posterior regions, as seen in the topomaps at 0.006s and 0.157s.


# %% Task 13 (10 points)

# Following the instruction on the same tutorial page,
# compute ICA with 16 components and fit to the filtered data of subject 1
# This may take a while depending on your computer.
# Plot the components and their source time series (but applied to the filtered data)
# Which components look obviously like they represent artefacts, and what kind?
# Exclude the artifactual components and apply the ICA to a copy of the raw data.
# Did the ICA do what it was supposed to?

ica = mne.preprocessing.ICA(n_components=16, max_iter="auto", random_state=97)
ica.fit(raw1_filtered)

ica.plot_components(title="ICA Components (Scalp Maps)")  #visualize scalp maps
ica.plot_sources(raw1_filtered, show_scrollbars=False, title="ICA Component Time Series", show=False)  #time series
plt.show(block=True)

eog_inds, eog_scores = ica.find_bads_eog(raw1_filtered, ch_name='EOG_simulated')
print(f"EOG-related components: {eog_inds}")

ecg_inds, ecg_scores = ica.find_bads_ecg(raw1_filtered, ch_name='ECG')
print(f"ECG-related components: {ecg_inds}")

ica.exclude = eog_inds + ecg_inds
print(f"Components marked for exclusion: {ica.exclude}")

reconst_raw1 = raw1_filtered.copy()
ica.apply(reconst_raw1)

raw1_filtered.plot(title="Original Data", show=False)
plt.show(block=True)

reconst_raw1.plot(title="Reconstructed Data After ICA", show=False)
plt.show(block=True)


#ICA000 seems to have a large ECG artefact.
#Main EOG related artefact is simulated in ICA001 due to eye blink.
#ICA003, 005, 013, 014 are major ECG related artefact, likely due to heartbeat or some muscle artefact.
#After reconstruction, we can see ICA has reduced eye blink artefacts and also minimized some ECG related artefacts.

# %%








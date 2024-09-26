import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from fooof import FOOOF

# Function to simulate power spectrum using the FOOOF model
def fooof_spectrum(frequencies, a, c, w, b, k, e):
    num_components = len(a)
    num_freqs = len(frequencies)
    power_spectrum = np.zeros(num_freqs)

    # Add oscillatory components to the power spectrum
    for i in range(num_components):
        component = a[i] * np.exp(-((frequencies - c[i]) ** 2) / (2 * w[i] ** 2))
        power_spectrum += component

    # Add aperiodic component to the power spectrum
    aperiodic_component = b - np.log(k + frequencies ** e)
    power_spectrum += aperiodic_component

    return power_spectrum

def generate_pink_noise(n_samples):
    freqs = np.fft.rfftfreq(n_samples)
    freqs[0] = 1  # Avoid division by zero at f = 0
    spectrum = 1 / np.sqrt(freqs)
    random_phases = np.exp(1j * 2 * np.pi * np.random.random(len(freqs)))
    noise = np.fft.irfft(spectrum * random_phases)
    return noise

# Frequency range from 0.1 to 45 Hz
frequencies = np.linspace(0.1, 45, 1000)

# Parameters for oscillatory components (example with 1 peak)
c = [8.217776732] 
a = [9.23687153] 
w = [2.503498526] 
b = 1.415934959 
e = 1.846200706

# Parameters for the aperiodic component
k = 0

# Compute power spectrum
fooof_psd = fooof_spectrum(frequencies, a, c, w, b, k, e)

# Parameters for pink noise generation
n_samples = len(frequencies)*2-1
sampling_rate = 1000

# Generate pink noise
pink_noise = generate_pink_noise(n_samples)

# Compute the PSD of pink noise using Welch's method
freqs_welch, pink_noise_psd = welch(pink_noise, fs=sampling_rate, nperseg=n_samples)

# Interpolate the pink_noise_psd to match the length of fooof_psd
pink_noise_psd_interp = np.interp(frequencies, freqs_welch, pink_noise_psd)

# Increase noise level by a factor (adjust this to increase/decrease noise)
noise_factor = 5000 
pink_noise_psd_scaled = pink_noise_psd_interp * noise_factor

# Add the pink noise PSD to the FOOOF-generated PSD
combined_psd = fooof_psd + pink_noise_psd_scaled

# Plot the individual components and the combined PSD
plt.figure(figsize=(10, 6))
plt.plot(frequencies, fooof_psd, label='FOOOF PSD', color='blue')
plt.plot(frequencies, pink_noise_psd_scaled, label='Pink Noise PSD', color='orange')
plt.plot(frequencies, combined_psd, label='Combined PSD', color='green', linestyle='--')

plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density')
plt.title('Simulated Power Spectrum with FOOOF and Pink Noise')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('simulate_fooof.png')

# Evaluation: Simple spectrum summary statistics
def evaluate_spectrum(spectrum):
    return {
        'mean_power': np.mean(spectrum),
        'std_power': np.std(spectrum),
        'max_power': np.max(spectrum)
    }

# Evaluate the original and noisy power spectra
noisy_eval = evaluate_spectrum(combined_psd)
print(f"Noisy Power Spectrum Evaluation: {noisy_eval}")

# Test 
print(combined_psd)

count = 0
combined_psd = np.array(combined_psd)  # Chuyển đổi sang numpy array
if np.any(np.isnan(combined_psd)) or np.any(np.isinf(combined_psd)):
    count += 1
print(count)

# Re-evaluate by fooof
freq_range = [0.1, 45]
fm = FOOOF(peak_width_limits=[0.5, 8.0], max_n_peaks=1, min_peak_height=0.1,
        peak_threshold=2.0, aperiodic_mode='fixed')
fm.report(frequencies, 10 ** fooof_psd, freq_range)
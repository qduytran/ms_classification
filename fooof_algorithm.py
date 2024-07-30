import numpy as np
from fooof import FOOOF

def fooof_tool(frequencies, psd):
    all_features = []
    for i in range(19):
        fm_periodic = FOOOF(peak_width_limits=[0.05, 20], max_n_peaks=1, min_peak_height=0.01, 
                   peak_threshold=-5, aperiodic_mode='fixed')
        fm_aperiodic = FOOOF(peak_width_limits=[0.05, 20], max_n_peaks=99999999999, min_peak_height=0.01, 
                   peak_threshold=-10, aperiodic_mode='fixed')
        freq_range_periodic = [4, 16]
        freq_range_aperiodic = [0.5, 40]
        fm_periodic.report(frequencies[i], psd[i][0], freq_range_periodic)
        fm_aperiodic.report(frequencies[i], psd[i][0], freq_range_aperiodic)
        aperiodic_params = fm_aperiodic.get_params('aperiodic_params')
        peak_params = fm_periodic.get_params('peak_params')
        peak_params = peak_params.flatten()
        if np.isnan(peak_params).any():
            peak_params = np.array([0, 0, 0])
        features = np.concatenate((peak_params, aperiodic_params))
        all_features.extend(features)
    return all_features 

def fooof_tool_from_mat_file(frequencies, psd):
    all_features = []
    for i in range(19):
        fm_periodic = FOOOF(peak_width_limits=[0.05, 20], max_n_peaks=1, min_peak_height=0.01, 
                   peak_threshold=-5, aperiodic_mode='fixed')
        fm_aperiodic = FOOOF(peak_width_limits=[0.05, 20], max_n_peaks=99999999999, min_peak_height=0.01, 
                   peak_threshold=-10, aperiodic_mode='fixed')
        freq_range_periodic = [4, 16]
        freq_range_aperiodic = [0.5, 40]
        fm_periodic.report(frequencies[0], psd[i], freq_range_periodic)
        fm_aperiodic.report(frequencies[0], psd[i], freq_range_aperiodic)
        aperiodic_params = fm_aperiodic.get_params('aperiodic_params')
        peak_params = fm_periodic.get_params('peak_params')
        peak_params = peak_params.flatten()
        if np.isnan(peak_params).any():
            peak_params = np.array([0, 0, 0])
        features = np.concatenate((peak_params, aperiodic_params))
        all_features.extend(features)
    return all_features 

def fooof_tool_drop_bandwidth_feature(frequencies, psd):
    all_features = []
    for i in range(19):
        fm_periodic = FOOOF(peak_width_limits=[0.05, 20], max_n_peaks=1, min_peak_height=0.01, 
                   peak_threshold=-5, aperiodic_mode='fixed')
        fm_aperiodic = FOOOF(peak_width_limits=[0.05, 20], max_n_peaks=99999999999, min_peak_height=0.01, 
                   peak_threshold=-10, aperiodic_mode='fixed')
        freq_range_periodic = [4, 16]
        freq_range_aperiodic = [0.5, 40]
        fm_periodic.report(frequencies[i], psd[i][0], freq_range_periodic)
        fm_aperiodic.report(frequencies[i], psd[i][0], freq_range_aperiodic)
        aperiodic_params = fm_aperiodic.get_params('aperiodic_params')
        peak_params = fm_periodic.get_params('peak_params')
        peak_params = peak_params.flatten()
        peak_params = peak_params[:2]
        if np.isnan(peak_params).any():
            peak_params = np.array([0, 0, 0])
        features = np.concatenate((peak_params, aperiodic_params))
        all_features.extend(features)
    return all_features 

def fooof_tool_drop_periodic_features(frequencies, psd):
    all_features = []
    for i in range(19):
        fm_aperiodic = FOOOF(peak_width_limits=[0.05, 20], max_n_peaks=99999999999, min_peak_height=0.01, 
                   peak_threshold=-10, aperiodic_mode='fixed')
        freq_range_aperiodic = [0.5, 40]
        fm_aperiodic.report(frequencies[i], psd[i][0], freq_range_aperiodic)
        aperiodic_params = fm_aperiodic.get_params('aperiodic_params')
        features = np.concatenate((aperiodic_params))
        all_features.extend(features)
    return all_features 
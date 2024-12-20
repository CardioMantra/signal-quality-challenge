import wfdb
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter("ignore", UserWarning)
from tqdm.notebook import tqdm
import neurokit2 as nk
import numpy as np
import multiprocessing
import sys, os
import sklearn
import sklearn.ensemble
import sklearn.svm
import datasets, sqis, featurization

def plot_signal(datfile, SAMPLE_START, SAMPLE_SIZE, CHANNEL, extract_annotation=False):
    record = wfdb.rdrecord(datfile)

    # Get data and annotations for the samples selected below.
    SAMPLE_END = SAMPLE_START + SAMPLE_SIZE
    channel = record.p_signal[SAMPLE_START:SAMPLE_END, CHANNEL]

    # Plot the heart beats. Time scale is number of readings divided by sampling frequency.
    times = (np.arange(SAMPLE_START, SAMPLE_END, dtype='float')) / record.fs
    plt.figure(figsize=(20,10))
    plt.plot(times, channel)

    if extract_annotation:
        # Extract annotations.
        annotation = wfdb.rdann(datfile, 'atr')

        where = np.logical_and(annotation.sample >= SAMPLE_START, annotation.sample < SAMPLE_END)
        annotation_symbol = np.array(annotation.symbol)[where]
        annotimes = annotation.sample[where] / record.fs

        # Plot the Annotations
        plt.scatter(annotimes, np.ones_like(annotimes) * channel.max() * 1.4, c='r')
        for idx in range(len(annotimes)):
            plt.annotate(annotation_symbol[idx], xy = (annotimes[idx], channel.max() * 1.3))

    plt.xlim([SAMPLE_START / record.fs, (SAMPLE_END / record.fs) + 1])
    plt.xlabel('Offset (Seconds from start)')
    plt.ylabel(record.sig_name[CHANNEL])
    plt.grid()
    plt.show()


def get_features(subject_dict, sampling_rate=125, window_size=10):
    X_features_dict = {
        'zhao2018': [],
        'orphanidou2015': [],
        'li2007': [],
        'clifford2012': [],
        'li2014': [],
        'behar2013': [],
        'average_qrs': [],
        'geometric': [],
        'all': [],
        'y_list': []
    }

    ## Calculate multi-lead features
    ## Remove baseline wader and dc offset with highpass Butterworth. Also remove powerline interference (50hz).
    ecg_cleaned_list = [
        nk.ecg_clean(subject_dict[channel]['data'], sampling_rate=sampling_rate, method="neurokit")
        for channel in subject_dict.keys()
        ]

    i_sqi = sqis.i_sqi(ecg_cleaned_list, sampling_rate)
    """sqi that measures inter-channel signal quality. 
    Calculated as the ratio of the number of matched beats (Nmatched) to 
    all detected beats (Nall) between a given lead and all other synchronous ECG."""

    pca_sqi = sqis.pca_sqi(np.array(ecg_cleaned_list).T) # 12 features
    """PCA sqi of the input signals"""
    i_sqi = 0
    pca_sqi = 0

    ## Calculate single-lead features
    for i, channel in enumerate(subject_dict.keys()):
        ecg_raw = subject_dict[channel]['data']
        ecg_cleaned = ecg_cleaned_list[i]
        X_features_dict['y_list'].append(subject_dict[channel]['label'])

        ## Find peaks indices
        peaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate, method='kalidas2017')[1]['ECG_R_Peaks']

        ## Featurize ecgs
        ecg_features = featurization.featurize_ecg(window=ecg_cleaned, sampling_rate=sampling_rate)

        ## Obtain ECG sqis for single channel
        orphanidou2015_sqi = sqis.orphanidou2015_sqi(ecg_cleaned, sampling_rate, show=False)
        averageQRS_sqi = sqis.averageQRS_sqi(ecg_cleaned, sampling_rate)
        zhao2018_sqi = sqis.zhao2018_sqi(ecg_cleaned, sampling_rate)
        p_sqi = sqis.p_sqi(ecg_cleaned, sampling_rate, window=window_size, num_spectrum=[5, 15], dem_spectrum=[5, 40])
        bas_sqi = sqis.bas_sqi(ecg_cleaned, sampling_rate, window=window_size, num_spectrum=[0, 1], dem_spectrum=[0, 40])
        c_sqi = sqis.c_sqi(ecg_cleaned, sampling_rate)
        q_sqi = sqis.q_sqi(ecg_cleaned, sampling_rate, matching_qrs_frames_tolerance=50)
        b_sqi = sqis.q_sqi(ecg_cleaned, sampling_rate, method='b_sqi')
        bs_sqi = sqis.bs_sqi(ecg_cleaned, peaks, sampling_rate)
        e_sqi = sqis.e_sqi(ecg_cleaned, peaks, sampling_rate)
        hf_sqi = sqis.hf_sqi(ecg_raw, peaks, sampling_rate)
        rsd_sqi = sqis.rsd_sqi(ecg_cleaned, peaks, sampling_rate)
        k_sqi = sqis.k_sqi(ecg_cleaned, kurtosis_method='fisher')
        s_sqi = sqis.s_sqi(ecg_cleaned)
        pur_sqi = sqis.pur_sqi(ecg_cleaned)
        ent_sqi = sqis.ent_sqi(ecg_cleaned)
        zc_sqi = sqis.zc_sqi(ecg_cleaned)
        f_sqi = sqis.f_sqi(ecg_cleaned, window_size=3, threshold=1e-7)

        X_features_dict['zhao2018'].append([zhao2018_sqi])
        X_features_dict['orphanidou2015'].append([orphanidou2015_sqi])
        X_features_dict['li2007'].append([i_sqi, b_sqi, p_sqi, k_sqi])
        X_features_dict['clifford2012'].append([i_sqi, b_sqi, p_sqi, k_sqi, s_sqi, f_sqi, bas_sqi])
        X_features_dict['li2014'].append([i_sqi, b_sqi, p_sqi, k_sqi, s_sqi, f_sqi, bas_sqi, bs_sqi, e_sqi, hf_sqi, pur_sqi, rsd_sqi, ent_sqi])
        X_features_dict['behar2013'].append([k_sqi, s_sqi, p_sqi, b_sqi, i_sqi, pca_sqi])
        X_features_dict['average_qrs'].append([averageQRS_sqi])
        X_features_dict['geometric'].append(ecg_features)
        X_features_dict['all'].append(ecg_features + [i_sqi, pca_sqi, p_sqi, bas_sqi, c_sqi, b_sqi, q_sqi, bs_sqi, e_sqi, hf_sqi, rsd_sqi, k_sqi, s_sqi, \
            pur_sqi, ent_sqi, zc_sqi, f_sqi, np.nanmean(ecg_cleaned), np.nanstd(ecg_cleaned), np.nanmax(ecg_cleaned), np.nanmin(ecg_cleaned)])

    return X_features_dict

def generate_features_dict(output_dict, X_features_dict):
    with multiprocessing.Pool(processes=10) as pool:
        X_features_dicts = list(tqdm(pool.imap(get_features, [output_dict[subject] for subject in output_dict.keys()]), total=len(output_dict.keys())))
        for i, d in enumerate(X_features_dicts):
            # print(f"i: {i}, d:{d}")
            for key in d.keys():
                # print(f"key: {key}")
                X_features_dict[key].extend(d[key])

            X_features_dict['subject'].extend([i for _ in range(len(d['y_list']))])

    return X_features_dict


"""
Created on 16.20 at 31/01/2022
@author: blia
"""
import matplotlib.pyplot as plt 
import numpy as np
import os
from torch import zero_ 


def calc_std_multiple_spectra(x, k):
    """Args:
    x: [num_spectra, wavenumber]
    k: int
    """
    _std = []
    for i in range(np.shape(x)[1] - k):
        v = np.std(np.diff(x[:, i:(i+k)], axis=-1), axis=-1)
        _std.append(v)
    _std = np.transpose(_std, (1, 0))
    _std = np.concatenate([np.zeros([len(x), k//2]), _std, np.zeros([len(x), k//2])], axis=-1)
    return _std


def aug_spec(spec, k, norm_std, sers_shape, repeats, norm_mean=0, val=False):
    """Augment the spectra
    Args:
        spec: [imh, imw, wavenumber]
        k: window size
        norm_std: the standard deviation for the normal distribution
        sers_shape: [imh, imw, wave]
        repeats: int
    """
    spec_reshape = np.reshape(spec, [np.prod(sers_shape[:-1]), sers_shape[-1]])
    spec_std = calc_std_multiple_spectra(spec_reshape, k)
    if val is False:
        noise = np.concatenate([np.zeros([1] + list(sers_shape)), 
                                np.random.normal(norm_mean, norm_std, [repeats] + list(sers_shape))], axis=0)
    else:
        noise = np.random.normal(norm_mean, norm_std, [repeats] + list(sers_shape))
    spec = spec + noise * np.reshape(spec_std, sers_shape)
    return spec 
    

def aug_signal(spectra, label, concentration, wave_cut_g, repeat, k=10, norm_std=3, norm_mean=0, 
               detection=True, quantification=False, val=False):
    spec_update, label_update, concentration_update, wave_cut_update = [[] for _ in range(4)]
    sers_shape = np.shape(spectra[0])
    print("----------------------Before augmenting the data----------------------")
    print("SERS map shape", np.shape(spectra))
    print("Unique label and count", np.unique(label, return_counts=True))
    print("Unique concentration and count", np.unique(concentration, return_counts=True))
    print("                                                                       ")
    if detection is True and quantification is False:
        zero_repeat = (repeat+1) * np.sum(concentration != 0) // np.sum(concentration == 0) - 1
    if detection is False and quantification is True:
        zero_repeat = repeat
    real_or_fake = []
    for i, s_spec in enumerate(spectra):
        if label[i] == 0:
            repeat_use = zero_repeat
        else:
            repeat_use = repeat 
        s_spec_update = aug_spec(s_spec, k, norm_std, sers_shape, repeat_use, norm_mean, val=val)
        spec_update.append(s_spec_update)
        label_update.append([label[i] for j in range(len(s_spec_update))])
        concentration_update.append([concentration[i] for j in range(len(s_spec_update))])
        wave_cut_update.append([wave_cut_g[i] for j in range(len(s_spec_update))])
        real_or_fake.append([1] + [0 for _ in range(repeat_use)])
    group_stat = [spec_update, label_update, concentration_update, wave_cut_update, real_or_fake]
    for i, s_stat in enumerate(group_stat):
        group_stat[i] = np.concatenate(s_stat, axis=0)
    return group_stat


def split_tr_to_tr_val_data(tr_spectra, tr_label, tr_concentration, tr_peak, num_val=10, quantification=False, dataset="DNP", tt_conc=0.0, detection=True):
    """Split the training data into training and validation data
    Args:
    tr_spectra: [num_measurements, imh, imw, Nw]
    tr_label: [num_measurements]
    tr_concentration: [num_measurements]
    tr_peaks: [num_measurements, num_peaks]
    """
    unique_concentration = np.unique(tr_concentration)
    original_stat = [tr_spectra, tr_label, tr_concentration, tr_peak]
    tr_update, val_update = [[] for _ in range(4)], [[] for _ in range(4)]
    tr_index, val_index = [], []
    for i in unique_concentration:
        index = np.where(tr_concentration == i)[0]
        if dataset == "PA" and i == tt_conc and detection == False:
            tr_index.append(index)
            for j in range(4):
                tr_update[j].append(original_stat[j][index])
            continue 
        if i == 0:
            if quantification is False:
                num_val_act = num_val * (len(unique_concentration) - 1)
            else:
                num_val_act = num_val
        else:
            num_val_act = num_val
        tr_index.append(index[:-num_val_act])
        val_index.append(index[-num_val_act:])
        for j in range(4):
            tr_update[j].append(original_stat[j][index[:-num_val_act]])
        for j in range(4):
            val_update[j].append(original_stat[j][index[-num_val_act:]])
    tr_index = np.concatenate(tr_index, 0)
    val_index = np.concatenate(val_index, 0)
    print("There are %d training data and %d validation data" % (len(tr_index), len(val_index)))
    print("The replicated validation index in the training", [v for v in val_index if v in tr_index])
    for j in range(4):
        tr_update[j] = np.array([v for q in tr_update[j] for v in q])
        val_update[j] = np.array([v for q in val_update[j] for v in q])    
    return tr_update, val_update    
    
        
       
        
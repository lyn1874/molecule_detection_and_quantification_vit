#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   vis_utils.py
@Time    :   2022/04/05 08:20:53
@Author  :   Bo 
'''
from audioop import cross
import numpy as np 
import os
import data.prepare_sers_data as psd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import utils as utils 
import pickle 
from sklearn.metrics import f1_score
from scipy.special import softmax
import torch 


TEXTWIDTH = 6.75133
TEXTWIDTH = 5.39366
def set_size(width, fraction=1, enlarge=0):
    """
    Args:
        width: inches
        fraction: float
    """
    # Width of figure (in pts)
    fig_width_in = width * fraction
    golden_ratio = (5 ** .5 - 1) / 2
    if enlarge != 0:
        golden_ratio *= enlarge
    fig_height_in = fig_width_in * golden_ratio
    fig_dim = (fig_width_in, fig_height_in)
    return fig_dim


def give_figure_specify_size(fraction, enlarge=0):
    fig = plt.figure()
    fig.set_size_inches(set_size(TEXTWIDTH, fraction, enlarge))
    return fig



def get_tt_label(dataset, quantification=False, leave_index=0, leave_method="leave_one_chip_per_conc", normalization="none"):
    if "SIMU" in dataset:
        target_shape = [30, 30]
    elif dataset == "TOMAS":
        target_shape = [56, 56]    
    data_obj = psd.ReadSERSData(dataset, target_shape=target_shape,
                                bg_method="ar",
                                tr_limited_conc=[], 
                                percentage=0.0, top_selection_method="sers_maps",
                                concentration_float=0.0, leave_index=leave_index,
                                leave_method=leave_method,
                                path_mom="../rs_dataset/", 
                                quantification=quantification,
                                normalization=normalization)
    tr_g, val_g, tt_g, imshape, num_class = data_obj.forward_test()
    return tr_g, val_g, tt_g


def unvisible_ticks(ax, ax_side="xtick"):
    if ax_side == "xtick":
        for tick in ax.xaxis.get_major_ticks():
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)
            tick.label1.set_visible(False)
            tick.label2.set_visible(False)
    if ax_side == "ytick":
        for tick in ax.yaxis.get_major_ticks():
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)
            tick.label1.set_visible(False)
            tick.label2.set_visible(False)


def show_example_sers_maps(tr_maps, tr_conc, tr_peak, tr_wave, use_conc, use_index, fig, nrow, ncol, base_index, dataset="SIMU_TYPE_2"):
    color_use=['r','g','b']
    signal_stat = pickle.load(open("../rs_dataset/simulate_sers_maps/%s_signal.obj" % dataset, "rb"))
    print(signal)
    conc_unique = np.unique(tr_conc)
    imh, imw = np.shape(tr_maps)[1:-1]
    for i, s_conc in enumerate(use_conc):
        index = np.where(tr_conc == s_conc)[0][use_index[i]]
        index_signal = np.where(conc_unique == s_conc)[0][0]
        print(s_conc, index_signal)
        signal = signal_stat[index_signal][use_index[i]]
        map_use = tr_maps[index]
        peak_loc_g = []
        for q, s_peak in enumerate(tr_peak[index]):
            loc_index = int(np.where(tr_wave >= s_peak)[0][0])
            peak_loc_g.append(loc_index)
        if np.shape(tr_peak)[1] == 3:
            map_plot = [map_use[:, :, v] for v in peak_loc_g]
        elif np.shape(tr_peak)[1] == 2:
            map_plot = [map_use[:, :, v] for v in peak_loc_g] + [np.mean(map_use, axis=-1)]
        for j, s_map in enumerate(map_plot):
            ax = fig.add_subplot(nrow, ncol, i * ncol + j + 1 + base_index)
            ax.imshow(s_map)
            if j <= 1:
                ax.set_title("Peak %d" % (j + 1), color=color_use[j])
            if j == 0:
                ax.set_ylabel("%.4f" % s_conc)
            if j == 2:
                ax.set_title(["Average" if len(peak_loc_g) == 2 else "contaminants"][0], color=color_use[-1])
            if i * ncol + j + 1 + base_index < (nrow - 1) * ncol:
                # ax.xaxis.set_major_formatter(plt.NullFormatter())
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                if j == 1:
                    if dataset == "SIMU_TYPE_4":
                        ax.set_xlabel("SERS maps")
                if dataset == "SIMU_TYPE_4":
                    ax.set_yticks([])
                else:
                    ax.set_xticks([])
                    ax.set_yticks([])
        ax = fig.add_subplot(nrow, 2, i * 2 + 2 + base_index // (ncol // 2))
        spectra = np.reshape(map_use, [imh * imw, len(tr_wave)])
        for j, s_spec in enumerate(spectra):
            ax.plot(tr_wave, s_spec, color='gray', alpha=0.1, lw=0.3)
        for j, s_peak in enumerate(peak_loc_g):
            ax.plot([tr_wave[s_peak], tr_wave[s_peak]], [0, np.max(spectra)], 
                    color=color_use[j], ls=':')
        spectra_rank = np.argsort(np.sum(spectra[:, (tr_peak[0][:2]).astype(np.int32)], axis=-1))
        spectra_use = spectra[spectra_rank[-4:]]
        for j, s_spec in enumerate(spectra_use):
            ax.plot(tr_wave, s_spec, color='m', lw=0.4)
        spectra_use = spectra[spectra_rank[:4]]
        for j, s_spec in enumerate(spectra_use):
            ax.plot(tr_wave, s_spec, color='k', lw=0.4)
        if s_conc != 0:
            ax.plot(tr_wave, signal, color='orange', lw=0.4)
        else:
            ax.plot(tr_wave, np.zeros([len(tr_wave)]), color='orange', lw=0.4)
        ax.yaxis.set_major_formatter(plt.NullFormatter())
        ax.set_title("Raman spectra")
        if (i * 2 + 1 + base_index // (ncol // 2)) != nrow * 2 - 1:
            ax.xaxis.set_major_formatter(plt.NullFormatter())
        else:
            if dataset == "SIMU_TYPE_4":
                ax.set_xlabel("Wavenumber (cm-1)")
            else:
                ax.xaxis.set_major_formatter(plt.NullFormatter())


def show_example_maps_and_attention_maps(im_g, attn_g, peak_g, signal_g, conc_g, tr_wave, 
                                         quantile_percentage=0.9, save_name="sers_maps_and_attention_maps",
                                         tds_dir=None, save=False, dataset="SIMU_TYPE_4"):
    """Display the sers maps and the attention maps"""
    nrow = len(im_g)
    if dataset != "DNP" and dataset != "PA":
        ncol = 5
    else:
        ncol = 4
    if "SIMU" not in dataset:
        fig = give_figure_specify_size(1.0, 1.3)
    else:
        fig = give_figure_specify_size(1.0, 1.0)
    ax_global = ax_global_get(fig)
    ax_global.set_xticks([])
    ax_global.set_yticks([])
    spec5 = fig.add_gridspec(ncols=ncol, nrows=nrow, 
                             width_ratios=[1 for _ in range(len(peak_g[0]) + 1)] + [3.5], 
                             wspace=0.0)
    color = ['r', 'g', 'b', 'm']
    if "SIMU" in dataset:
        title_use = ["Peak 1", "Peak 2", "Contaminants", "Attention map (mask)"]
    elif dataset == "TOMAS":
        title_use = ["Peak 1", "Peak 2", "Peak 3", "Attention map (mask)"]
        conc_label = ["Blank", "0.1nM", "1nM", "10nM", "100nM", "1"+r'$\mu$'+"M"]
    elif dataset == "DNP":
        title_use = ["Peak 1", "Peak 2", "Attention map (mask)"]
        conc_label = ["Blank", "1nM", "10nM", "100nM", "1"+r'$\mu$'+"M", "10" + r'$\mu$'+ "M"]
    for i in range(nrow):
        _im, _attn, _s_peak, _s_signal = im_g[i], attn_g[i], peak_g[i], signal_g[i]
        if "SIMU" in dataset:   
            _im_show = [_im[:, :, int(v)] for v in _s_peak]
        else:
            _im_show = [np.mean(_im[:, :, v[0]:v[1]], axis=-1) for v in _s_peak]
        _im_show += [_attn]
        threshold = np.quantile(np.reshape(_attn, [-1]), quantile_percentage)
        bool_mask = (_attn >= threshold).astype(np.int8)
        for j in range(ncol - 1):
            ax = fig.add_subplot(spec5[i, j])
            if j != ncol - 2:
                ax.imshow(_im_show[j])
            else:
                ax.imshow(_im_show[j], cmap='Greys')
                ax.imshow(bool_mask, cmap='Blues', alpha=0.5)
            if i != nrow - 1:
                ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(title_use[j], color=color[j])
            if j == 0:
                if dataset == "TOMAS" or dataset == "DNP" or dataset == "PA":
                    ax.set_ylabel("%s" % conc_label[i])
                else:              
                    ax.set_ylabel("%.4f" % (conc_g[i]))
            if i == len(im_g) - 1 and j == 1:
                if "SIMU" in dataset:
                    ax.set_xlabel("SERS maps at different raman shift locations", loc='center')
                else:
                    ax.set_xlabel("SERS maps at different raman shift locations")
                    ax.xaxis.set_label_coords(1.0, -0.45)
        ax = fig.add_subplot(spec5[i, j + 1])
        spectra_all = np.reshape(_im, [-1, np.shape(_im)[-1]])
        if dataset == "TOMAS" or dataset == "DNP" or dataset == "PA":
            avg_spec, std_spec = np.mean(spectra_all, axis=0), np.std(spectra_all, axis=0)
            # ax.plot(tr_wave, avg_spec, color='gray', alpha=0.5, lw=0.4)
            # ax.fill_between(tr_wave, avg_spec - std_spec, avg_spec + std_spec, color='gray', alpha=0.5)
            for v in spectra_all:
                ax.plot(tr_wave, v, color='gray', alpha=0.2, lw=0.4)
        else:
            for v in spectra_all:
                ax.plot(tr_wave, v, color='gray', alpha=0.2, lw=0.4)
        if "SIMU_TYPE" in dataset:
            extract_spectra = np.sum(spectra_all[np.reshape(bool_mask, [-1]) == 1, :], axis=0)/np.sum(bool_mask)
            ax.plot(tr_wave, extract_spectra, color='m')
        else:
            index_use = np.logical_and((np.reshape(bool_mask, [-1]) == 1), np.sum(spectra_all, axis=-1) > 10)
            print(np.sum(index_use), np.sum(bool_mask), np.max(np.array(_im_show)), 
                  np.max(np.sum(spectra_all[index_use], axis=0)), np.min(np.array(_im_show)), 
                  np.min(_im))   
            # ax2 = ax.twinx()
            # ax2.plot(tr_wave, np.mean(spectra_all[index_use], axis=0), color='m', alpha=0.8, lw=1.0)
            # ax2.set_yticks([])
            enhance_factor = 8 if dataset == "TOMAS" else 2
            ax.plot(tr_wave, np.mean(spectra_all[index_use], axis=0) * enhance_factor , color='m')
        if len(_s_signal) > 0:
            ax.plot(tr_wave, _s_signal, color='orange')
        if "SIMU" in dataset:
            for q_iter, q in enumerate(_s_peak):
                ax.plot([q, q], [0, np.max(spectra_all)], color=color[q_iter], alpha=0.5, ls=':')
        else:
            for q_iter, q in enumerate(_s_peak):
                ax.fill_betweenx([0, np.max(spectra_all)], [tr_wave[q[0]], tr_wave[q[0]]], 
                                 [tr_wave[q[1]], tr_wave[q[1]]], color=color[q_iter])
        ax.set_yticks([])
        if i == 0:
            ax.set_title("Raman Spectra")
        if i != len(im_g) - 1:
            ax.xaxis.set_major_formatter(plt.NullFormatter())
        else:
            ax.set_xlabel("Wavenumber (cm-1)")
        # if dataset == "TOMAS":
        #     ax.set_ylim((0, np.sort(np.reshape(_im, [-1]))[-3]))
    ax_global.set_ylabel("Concentration\n")    
    plt.subplots_adjust(wspace=0.03)
    if save:
        plt.savefig(tds_dir + "/%s.pdf" % save_name, pad_inches=0, bbox_inches='tight')
        

def show_accuracy_spectra_heatmap(performance_table, stat, ax, use_xticks, tt_selection_method, title_use="none"):
    model_percentage, percentage = stat[0], stat[1]
    if use_xticks:
        xticklabel = ["%.1f" % (v*100) for v in model_percentage]
    else:
        xticklabel = ["" for v in model_percentage]
    num_selection_criteria = len(performance_table)
    if len(ax) == 0:
        if len(performance_table) <= 4:
            fig = plt.figure(figsize=(14 / 4 * num_selection_criteria, 4))
            ax = [fig.add_subplot(1, num_selection_criteria, i+1) for i in range(num_selection_criteria)]
        else:
            ncol = 4
            nrow = int(np.ceil(len(performance_table) / ncol)) 
            fig = plt.figure(figsize=(14 / 4 * ncol, nrow * 4)) 
            ax = [fig.add_subplot(nrow, ncol, i+1) for i in range(nrow * ncol)]      
        ax_global = utils.ax_global_get(fig)
        ax_global.set_xticks([])
        ax_global.set_yticks([])
        ax_global.set_title("%s" % tt_selection_method)
        
        plt.subplots_adjust(wspace=0.05)
    min_value, max_value = np.min(performance_table), np.max(performance_table)
    for j in range(num_selection_criteria):
        if j == 0:
            yticklabel = ["%.1f" % (v*100) for v in percentage]
        else:
            yticklabel = ["" for v in percentage]            
        sns.heatmap(performance_table[j], annot=True, ax=ax[j], cbar=False, cmap="YlGnBu",
                    fmt='0.0f', xticklabels=xticklabel, yticklabels=yticklabel, vmin=min_value,
                    vmax=max_value)
        ax[j].set_title(title_use[j])
        
        
def calc_agresti_coull(num_success, num_sample):
    success_tilde = num_success + 2 
    sample_tilde = num_sample + 4 
    p_hat = success_tilde / sample_tilde
    conf_interval = 1.96 * np.sqrt(p_hat * (1-p_hat) / sample_tilde)
    return conf_interval


def show_accuracy_over_concentration(pred, pred_label, tt_label, tt_conc, show=True, save=False, tds_dir=None):
    accu_per_conc, std_per_conc, xticklabels = [], [], []
    if len(pred_label) == 0:
        if len(np.shape(pred)) == 3:
            pred_label = np.argmax(pred, axis=-1)[:,0]
        else:
            pred_label = np.argmax(pred, axis=-1)
    correct_or_wrong = (pred_label == tt_label).astype(np.int8)
    unique_conc = np.unique(tt_conc)
    for i, s_conc in enumerate(unique_conc):
        index = np.where(tt_conc == s_conc)[0]
        accu = np.sum(correct_or_wrong[index]) / len(index)
        _std_value = calc_agresti_coull(np.sum(correct_or_wrong[index]), len(index))
        accu_per_conc.append(accu)
        std_per_conc.append(_std_value)
        xticklabels.append("%.4f" % s_conc)
    # print(np.mean(accu_per_conc), np.sum(correct_or_wrong) / len(tt_label))
    return accu_per_conc, std_per_conc, xticklabels

def calc_avg_prob_per_concentration(pred, tt_label, tt_conc):
    avg_prob = np.zeros([len(np.unique(tt_conc))])
    for i, v in enumerate(np.unique(tt_conc)):
        _index = np.where(tt_conc == v)[0]
        _avg_prob = np.mean(pred[_index, tt_label[_index].astype(np.int32)], axis=0)
        avg_prob[i] = _avg_prob 
    return avg_prob 

# def show_accuracy_over_concentration(pred, pred_label, tt_label, tt_conc, show=True, save=False, tds_dir=None):
#     if len(pred_label) == 0:
#         if len(np.shape(pred)) == 3:
#             pred_label = np.argmax(pred, axis=-1)[:,0]
#         else:
#             pred_label = np.argmax(pred, axis=-1)
#     accu_per_conc = []
#     std_per_conc = []
#     xticklabels = []
#     for i in sorted(np.unique(tt_label)):
#         index = np.where(tt_label == i)[0]
#         pred_subset = pred_label[index]
#         conc_subset = tt_conc[index]
#         correct_or_wrong = (pred_subset == tt_label[index]).astype(np.int32)
#         for j, s_conc in enumerate(np.unique(conc_subset)):
#             _subindex = np.where(conc_subset == s_conc)[0]
#             _accu = np.sum(correct_or_wrong[_subindex]) / len(_subindex)
#             _std_value = calc_agresti_coull(np.sum(correct_or_wrong[_subindex]), len(_subindex))
#             accu_per_conc.append(_accu)
#             std_per_conc.append(_std_value)
#             xticklabels.append("%.4f" % s_conc)
#     if show:
#         fig = plt.figure(figsize=(10, 3))
#         ax = fig.add_subplot(111)
#         xticks = np.arange(len(accu_per_conc) * 2)[::2]
#         ax.plot(xticks, accu_per_conc, 'r-x')
#         ax.set_xticks(xticks)
#         ax.set_xticklabels(xticklabels, fontsize=10)
#         ax.set_ylabel("Accuracy", fontsize=10)
#         ax.set_xlabel("Class (Concentration)", fontsize=10)
#         ax.grid(ls=':')
#         print("Overall accuracy: %.2f" % (np.mean(accu_per_conc) * 100))
#         if save:
#             plt.savefig(tds_dir + "/accuracy_under_each_concentration.pdf", pad_inches=0, bbox_inches='tight')
#     else:
#         return accu_per_conc, std_per_conc, xticklabels
                   
    
def get_accuracy_heatmap(dataset, model, model_method, tt_label, tt_conc, 
                         detection=True, quantification=False, avg_spectra=True, 
                         tt_selection_method=["top_mean", "top_std", "top_diff", "top_peak"],
                         perf_crit="rsquare",
                         leave_index=0, 
                         normalization="none", 
                         version=11, loc="home"):
    if loc == "home":
        path_start = "../"
    elif loc == "scratch":
        path_start = "/scratch/blia/"
    elif loc == "nobackup":
        path_start = "/nobackup/blia/"
    path_init = path_start + "exp_data/Spectra_%s/%s/" % (model, dataset)
    path_init += "detection_%s_quantification_%s_average_spectra_%s/tds/version_%d_selection_method_%s" % (detection, quantification, avg_spectra, version, model_method)
    if quantification:
        concentration_float = 1e-6 if dataset != "PA" else 1e-5 
    else:
        concentration_float = 0 
    if "SIMU_TYPE" not in dataset:
        path_init += "_leave_chip_%d_normalization_%s" % (leave_index, normalization)
    performance = pickle.load(open(path_init + ".obj", "rb"))
    path_avg_map_dim = path_init.split("selection_method_")[0] + "selection_method_avg_map_dim"
    if "SIMU_TYPE" not in dataset:
        path_avg_map_dim += "_leave_chip_%d_normalization_%s" % (leave_index, normalization)
    performance_with_all_selection = pickle.load(open(path_avg_map_dim + ".obj", "rb"))
    keys_group = list(performance.keys())
    model_percentage = np.sort(np.unique([float(v.split("percentage_")[1].split("_tt_method")[0]) for v in keys_group]))
    model_percentage = np.array(list(model_percentage) + [1.0])
    percentage = np.sort(np.unique([float(v.split("tt_percent_")[1].split("_tt_quantile")[0]) for v in keys_group]))
    performance_table = np.zeros([len(tt_selection_method), len(model_percentage),len(percentage)])
    pred_group = np.zeros([len(tt_selection_method), len(model_percentage), len(percentage), len(tt_label), len(np.unique(tt_label))])
    if quantification is True:
        pred_group = pred_group[:, :, :, :, 0]
    key_group, perf_group = [], []
    for j, s_method in enumerate(tt_selection_method):
        for q, s_model_percent in enumerate(model_percentage):
            for n, s_tt_percent in enumerate(percentage):
                if s_model_percent != 1.0:                        
                    key_use = "method_%s_percentage_%.3f_tt_method_%s_tt_percent_%.3f_tt_quantile_False" % (model_method,
                                                                                          s_model_percent,
                                                                                          s_method, 
                                                                                          s_tt_percent)
                    pred = performance[key_use]
                else:
                    key_use = "method_avg_map_dim_percentage_%.3f_tt_method_%s_tt_percent_%.3f_tt_quantile_False" % (s_model_percent, 
                                                                                                   s_method,
                                                                                                   s_tt_percent)
                    pred = performance_with_all_selection[key_use]
                if detection is True:
                    naive_accuracy = get_performance_baseon_crit_detection(tt_label, pred, perf_crit)
                if quantification is True:
                    naive_accuracy = get_performance_baseon_crit_quantification(tt_conc, pred, perf_crit, concentration_float)
                performance_table[j, q, n] = naive_accuracy
                key_group.append(key_use)
                perf_group.append(naive_accuracy)
                # print("The shape of the prediction", np.shape(pred))
                if detection is True:
                    pred_group[j, q, n] = np.array(pred)[:, 0] # need to check the shape
                else:                    
                    pred_group[j, q, n] = pred
            # print(s_method, s_model_percent)
    return performance_table, pred_group, [model_percentage, percentage, key_group, path_init.split("version_")[0]]


def get_performance_baseon_crit_quantification(tt_conc, pred_conc, crit, concentration_float=1e-6):
    if crit == "rae" or crit == "rmae" or crit == "rmse" or crit == "log_rsquare" or crit == "rsquare":
        tt_conc_update = tt_conc.copy()
        tt_conc_update[tt_conc_update == 0] = concentration_float
    else:
        tt_conc_update = tt_conc
    if crit == "rsquare":
        accu = utils.calc_rsquare(tt_conc_update, pred_conc)
    elif crit == "log_rsquare":
        
        accu = utils.calc_rsquare(np.log(tt_conc_update), np.log(pred_conc))
    elif crit == "rae":
        accu = utils.calc_rae(tt_conc_update, pred_conc)
    elif crit == "rmae":
        accu = utils.calc_rmae(tt_conc_update, pred_conc)
    elif crit == "mae":
        accu = np.sum(abs(tt_conc_update - pred_conc))
    return accu 


def cross_entropy(predictions, targets, epsilon=1e-8):
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions. 
    Input: predictions (N, k) ndarray
           targets (N, k) ndarray        
    Returns: scalar
    """
    target_onehot = np.zeros([len(targets), targets.max() + 1])
    target_onehot[np.arange(len(targets)), targets] = 1
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(target_onehot*np.log(predictions+epsilon))/N
    return ce


def get_performance_baseon_crit_detection(tt_label, pred, crit, label_tensor=None):
    if crit != "validation_loss":
        if len(np.shape(pred)) == 3:
            pred_label = np.argmax(np.mean(pred, axis=1), axis=-1)
        else:
            pred_label = np.argmax(pred, axis=-1)
    if crit == "global_accu":
        accu = np.sum(pred_label == tt_label) / len(tt_label) * 100
    elif crit == "f1_score":
        accu = f1_score(tt_label, pred_label)
    elif crit == "false_positive":
        if np.sum(tt_label == 0) != 0:
            accu = np.sum(np.array(pred_label)[tt_label == 0] == 0) / np.sum(tt_label == 0)
        else:
            accu = 0
    elif crit == "validation_loss":
        if pred.dtype == torch.float32:
            accu = torch.nn.CrossEntropyLoss(reduction='sum')(pred, label_tensor).cpu().detach().numpy() / len(tt_label)
        else:
            accu = cross_entropy(pred, tt_label)
    return accu 

    
    
def ax_global_get(fig):
    ax_global = fig.add_subplot(111, frameon=False)
    ax_global.spines['top'].set_color('none')
    ax_global.spines['bottom'].set_color('none')
    ax_global.spines['left'].set_color('none')
    ax_global.spines['right'].set_color('none')
    ax_global.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')

    return ax_global
        
        
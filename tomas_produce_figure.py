"""
Created on 18:01 at 19/01/2022
@author: bo
"""
from ensurepip import version
from enum import EnumMeta
from operator import mod
from jinja2 import select_autoescape
import numpy as np
import os
import utils
import data.prepare_sers_data as psd
import pickle
import seaborn as sns
import matplotlib
from sklearn.metrics import roc_curve, auc, f1_score
import pickle
import data.read_tomas as read_tomas
from scipy.special import softmax
from matplotlib.ticker import FormatStrFormatter
from configs.common import str2bool
import pandas as pd 
import torch 
import data.read_tomas as read_tomas
import data.read_dnp as read_dnp
import data.read_pa as read_pa


FONTSIZE = 7
matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.size": FONTSIZE,
        "legend.fontsize": FONTSIZE,
        "xtick.labelsize": FONTSIZE,
        "ytick.labelsize": FONTSIZE,
        "legend.title_fontsize": FONTSIZE,
        "axes.titlesize": FONTSIZE,
        
    }
)
import matplotlib.pyplot as plt
import vis_utils as vu
import ml_method as mm 
import argparse


def give_args():
    """This function is used to give the argument"""
    parser = argparse.ArgumentParser(description='Reproduce figures in the paper')
    parser.add_argument('--dir2read_exp', type=str, default="../exp_data/exp_group/")
    parser.add_argument('--dir2read_data', type=str, default="../data_group/")
    parser.add_argument('--dir2save', type=str, default="../rs_dataset/paper_figure/")
    parser.add_argument('--index', type=str, default="figure_1", help="which figure or table do you want to produce?")
    parser.add_argument("--save", type=str2bool, default=False, help="whether to save the image or not")
    parser.add_argument("--pdf_pgf", type=str, default="pgf", help="in what kind of format will I save the image?")
    return parser.parse_args()


def prepare_label_and_concentration(detection=False, quantification=True, dataset="TOMAS"):
    conc_group, label_group = [[] for _ in range(2)], [[] for _ in range(2)]
    leave_method = "leave_one_chip"
    path_mom="../rs_dataset/"
    normalization="none"
    if dataset == "TOMAS":
        targ_shape = [56, 56]
    elif dataset == "DNP":
        targ_shape = [44, 44]
    elif dataset == "PA":
        targ_shape = [40, 40]
    max_leave_index = 30 if dataset != "PA" else 25 
    for leave_index in np.arange(max_leave_index).astype(np.int32):    
        data_obj = psd.ReadSERSData(dataset, target_shape=targ_shape, 
                                    bg_method="ar",
                                    tr_limited_conc=[0],
                                    percentage=0, top_selection_method="sers_maps", 
                                    path_mom=path_mom, use_map=False, quantification=quantification,
                                    detection=detection,
                                    cast_quantification_to_classification=False,
                                    normalization=normalization, leave_index=leave_index, 
                                    skip_value=1, leave_method=leave_method)
        _, [_, val_label, val_conc, _], [_, tt_label, tt_conc, _, _], imshape, num_class = data_obj.forward_test()
        for i, s_label in enumerate([val_label, tt_label]):
            label_group[i].append(s_label)
        for i, s_conc in enumerate([val_conc, tt_conc]):
            conc_group[i].append(s_conc)
    for i, s_stat in enumerate(conc_group):
        print(np.shape(s_stat))
    for i, s_stat in enumerate(label_group):
        print(np.shape(s_stat))    
    with open("../rs_dataset/%s_label_conc_detection_%s_quantification_%s.obj" % (dataset, detection, 
                                                                                  quantification), 
            "wb") as f:
        pickle.dump([label_group, conc_group], f)
        
        
def show_spectra(select_index=0, percentage=0.01, tds_dir=None, save=False):
    dataset_g = ["TOMAS", "DNP", "PA"]
    label_conc_g = [["Blank", "0.1 nM", "1 nM", "10 nM", "100 nM", "1 "+r'$\mu$'+"M"], 
                   ["Blank", "1 nM", "10 nM", "100 nM", "1 "+r'$\mu$'+"M", "10 "+r'$\mu$'+"M"],
                   ["Blank", "10 nM", "100 nM", "1 "+r'$\mu$'+"M", "10 "+r'$\mu$'+"M"]]
    peak_loc_g = [[1081, 1571, 1334],
                  [830, 1320],
                  [820, 1330]]
    targ_shape = [[56, 56], 
                  [44, 44],
                  [40, 40]]
    color_use = ['r','g','b','orange','m','tab:brown']
    fig = vu.give_figure_specify_size(0.5, 3.5)   
    ax_global = vu.ax_global_get(fig)
    ax_global.set_xticks([])
    ax_global.set_yticks([])
    title_group = ["4-NBT", "DNP", "Picric Acid"]
    enlarge = [4, 1, 0.5]
    text_loc = [4.5, 1, 0.54]
    text_en = [2, 1, 0.56]
    for i, s_dataset in enumerate(dataset_g):
        if s_dataset == "TOMAS":
            sers_maps, label, concentration, wavenumber, mapsize = read_tomas.prepare_tomas_data(targ_shape[i], 
                                                                                                testing=True)
        elif s_dataset == "DNP":
            sers_maps, label, concentration, wavenumber, mapsize = read_dnp.prepare_dnp_data(targ_shape[i],
                                                                                             testing=True)
        elif s_dataset == "PA":
            sers_maps, label, concentration, wavenumber, mapsize = read_pa.prepare_pa_data(targ_shape[i],
                                                                                          testing=True)
        unique_conc = np.unique(concentration)
        map_subset = [sers_maps[np.where(concentration == i)[0][select_index]] for i in unique_conc]
        map_subset = np.concatenate([map_subset], axis=0)
        extracted_spectra, info = psd.simple_selection(map_subset, wavenumber, percentage, "top_peak", s_dataset)
        ax = fig.add_subplot(3, 1, i+1)
        add_value = 0
        for j, s_spec in enumerate(extracted_spectra):
            ax.plot(wavenumber, s_spec+add_value * enlarge[i], color=color_use[j], label=label_conc_g[i][j])
            ax.text(wavenumber[-70], s_spec[-70] + add_value * text_loc[i]+10, label_conc_g[i][j], 
                   color=color_use[j])
            add_value += (np.max(s_spec)) 
        for j, s_peak in enumerate(peak_loc_g[i]):
            ax.plot([s_peak, s_peak], 
                    [np.min(extracted_spectra), add_value * text_en[i] + np.max(s_spec) / 3 ], ls=':',
                    color='gray')
            ax.text(s_peak-80, add_value * text_en[i] + np.max(s_spec) / 2 , "%d cm" % s_peak + r"$^{-1}$", 
                    )
        ax.set_ylim((np.min(extracted_spectra), add_value * text_en[i] + np.max(s_spec) ))
        ax.set_yticks([])
        ax.set_title(title_group[i])
    ax_global.set_xlabel("\nWavenumber (cm" + r"$^{-1}$" + ")")
    ax_global.set_ylabel("A.U.")
    plt.subplots_adjust(hspace=0.26)
    if save:
        plt.savefig(tds_dir + "/example_spectra.pdf", pad_inches=0, bbox_inches='tight')

                
def aggregate_performance(dataset, detection=True, quantification=False, target_size=56, loss="mse", 
                          learning_rate=0.0004, normalization="none",
                          model_init="xavier", lr_schedule="cosine", 
                          version_group=[], loc="nobackup"):
    path = "../exp_data/VIT/%s/detection_%s_quantification_%s" % (dataset, detection, quantification)
    if quantification == True:
        path += "_nozero"
    path += "/"
    file2look = [v for v in os.listdir(path) if ".obj" in v \
                and "leave_one_chip_" in v and "._stat" not in v \
                and "target_h_%d" % target_size in v and "chip.obj" not in v \
                and "%s" % loss in v and "lr_%.4f_" % learning_rate in v and "normalization_%s" % normalization in v]
    if len(version_group) > 0:
        file2look = [v for v in file2look if int(v.split("chip_v")[1].split(".obj")[0]) in version_group]
        # file2look = [v for v in file2look if int(v.split("version_")[1].split("_")[0]) in version_group]
    if dataset == "PA" or dataset == "DNP" or dataset == "TOMAS":
        file2look = [v for v in file2look if model_init in v and lr_schedule in v]
    print(file2look)
    stat_g = []
    for i, s in enumerate(file2look):
        s_stat = pickle.load(open(path + s, 'rb'))
        stat_g.append(s_stat)
    for i in range(len(file2look))[1:]:
        for q in range(len(stat_g[0])):
            stat_g[0][q].update(stat_g[i][q])
    update_filename = file2look[0].split("leave_one_chip")[0]
    if loc == "scratch" and quantification == True:
        update_filename += "without_zero_concentration_"
    name2save = path + update_filename + "leave_one_chip.obj"
    print(name2save)
    with open(name2save, "wb") as f:
        pickle.dump(stat_g[0], f)
        
        
def aggregate_figures_diff_input_shape(tds_dir="../rs_dataset/paper_figure/", save=False):
    x_axis, performance_detect = compare_performance_with_diff_input_shape(True, False, False)
    x_axis, performance_quantify = compare_performance_with_diff_input_shape(False, True, False)
    fig = vu.give_figure_specify_size(0.5, 2.6/6 * 5)
    ax_global = vu.ax_global_get(fig)
    ax_global.set_xticks([])
    ax_global.set_yticks([])
    ax = fig.add_subplot(311)
    ax.plot(x_axis, performance_detect[:, 0], 'r-^')
    # ax.plot(x_axis, performance_detect[:, 1], "g-^", label="F1 score")
    ax.set_xscale("log")
    ax.set_ylabel("Detection accuracy")
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    # ax.set_title("Detection")    
    ax = fig.add_subplot(312)
    ax.plot(x_axis, performance_quantify[:, 0], 'r-^')
    ax.set_xscale("log")
    ax.set_ylabel("Rsquare")
    # ax.set_title("Quantification")
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.set_ylim((0.975, 1.0))
    ax = fig.add_subplot(313)
    ax.plot(x_axis, performance_quantify[:, 2], 'r-^')
    ax.set_xscale("log")
    ax.set_ylabel("MAE")
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax_global.set_xlabel("\nNumber of spectra per map")
    plt.subplots_adjust(hspace=0.08)
    if save:
        plt.savefig(tds_dir + "/Tomas_influence_of_different_number_of_spectra.pdf", pad_inches=0, bbox_inches='tight')
        
        
def get_attention_map_TOMAS(conc_use, select_sample_index, dataset,
                                         detection=True, quantification=False,
                                         show=True):
    attn_path = "../exp_data/VIT/%s/detection_%s_quantification_%s/%s.obj" % (dataset, detection, 
                                                                              quantification, dataset)
    attn_tot = pickle.load(open(attn_path, "rb"))
    conc = np.array(attn_tot["concentration"])
    sers_map = attn_tot["sers_maps"]
    if dataset == "TOMAS":
        map_region = [[1071, 1083], [1324, 1336], [1566, 1575]]
    elif dataset == "DNP":
        map_region = [[802.34, 840.91], [1280.46, 1350.03]]
    tr_wave = attn_tot["wavenumber"][0]
    map_index = [[np.where(tr_wave >= v[0])[0][0], np.where(tr_wave >= v[1])[0][0]] for v in map_region]
    attention_map = attn_tot["attention_map"]
    unique_conc = np.unique(conc)
    print("The shape of the concentration, sers maps and peaks", np.shape(conc), 
          np.shape(sers_map), np.shape(attention_map))
    im_g, attn_g, peak_g, signal_g = [], [], [], []
    for i, s_conc in enumerate(conc_use):
        s_sample_index = select_sample_index[i]    
        index = np.where(conc == s_conc)[0][s_sample_index]
        print(s_sample_index)
        _attn_map = attention_map[index]
        _sers_map = sers_map[index]
        im_g.append(_sers_map)
        attn_g.append(_attn_map)
        peak_g.append(map_index)
    signal_g = [[] for _ in im_g]
    print("The results image and attention maps shape", np.shape(im_g), np.shape(attn_g), np.shape(peak_g), 
           np.shape(signal_g))    
    return im_g, attn_g, peak_g, signal_g, tr_wave


def show_attention_maps_dnp(tds_dir="../rs_dataset/paper_figure/", save=False):
    conc_use = [0, 0.0001, 0.001, 0.01, 0.1, 1]
    select_sample_index = [2, 3, 4, 3, 3, 1]
    im_g, attn_g, peak_g, signal_g, tr_wave = get_attention_map_TOMAS(conc_use, select_sample_index, 
                                                                     "DNP", False, True)
    vu.show_example_maps_and_attention_maps(im_g, attn_g, peak_g, signal_g, conc_use, tr_wave, 
                                        quantile_percentage=0.995, save_name="sers_maps_and_attention_maps_dnp",
                                        tds_dir=tds_dir, save=save, dataset="DNP")


def show_attention_maps_tomas(tds_dir="../rs_dataset/paper_figure/", save=False):
    conc_use = [0, 0.0001, 0.001, 0.01, 0.1, 1]
    select_sample_index = [4, 0, 2, 4, 1, 4]
    im_g, attn_g, peak_g, signal_g, tr_wave = get_attention_map_TOMAS(conc_use, select_sample_index, 
                                                                     "TOMAS", False, True)
    vu.show_example_maps_and_attention_maps(im_g, attn_g, peak_g, signal_g, conc_use, tr_wave, 
                                        quantile_percentage=0.995, save_name="sers_maps_and_attention_maps_tomas",
                                        tds_dir=tds_dir, save=save, dataset="TOMAS")


def compare_performance_with_diff_input_shape(detection, quantification, show=False):
    if detection:
        lr, normalization, concentration_float, quantification_loss = 0.008, "max", 0, "none"
    if quantification:
        lr, normalization, concentration_float, quantification_loss = 0.08, "none", 1e-6, "mae"
    target_shape = [8, 10, 12, 14, 20, 30, 56]
    performance_group = []
    pred_group = []
    for i, s_targ in enumerate(target_shape):
        obj = GetTomasPerformanceVIT(detection, quantification, lr, 2, [], normalization=normalization, 
                                     concentration_float=concentration_float, quantification_loss=quantification_loss, 
                                     target_shape=[s_targ, s_targ], leave_method="leave_one_chip", dataset="TOMAS")
        if quantification:
            pred, conc, con_update, perf_group = obj.get_quantification_performance(show=show, tds_dir=None, save=False)
            performance_group.append(perf_group)
            pred_group.append(pred)
        if detection:
            pred, pred_label, tt_label, conc, perf = obj.get_detection_performance(show=show, tds_dir=None, save=False)
            performance_group.append(perf)
    performance_group = np.array(performance_group)
    x_axis = np.array(target_shape) ** 2
    x_label = "Number of spectra per map"
    if detection:
        if show:
            fig = vu.give_figure_specify_size(0.5)
            ax = fig.add_subplot(111)
            ax.plot(x_axis, performance_group[:, 0], 'r-^', label="Global accuracy")
            ax.plot(x_axis, performance_group[:, 1], "g-^", label="F1 score")
            ax.set_xscale("log")
            ax.legend(loc='best')
            ax.set_xlabel(x_label)
            ax.set_ylabel("Detection accuracy")
    
    if quantification:
        if show:
            fig = vu.give_figure_specify_size(0.5, 1.7)
            ax_global = vu.ax_global_get(fig)
            ax_global.set_xticks([])
            ax_global.set_yticks([])
            ax = fig.add_subplot(211)
            ax.plot(x_axis, performance_group[:, 0], 'r-^', label="Rsquare")
            ax.set_xscale("log")
            ax.legend(loc='best')
            ax.xaxis.set_major_formatter(plt.NullFormatter())
            ax = fig.add_subplot(212)
            # ax.plot(x_axis, performance_group[:, 1], 'g-^', label="RMAE")
            ax.plot(x_axis, performance_group[:, 2], 'b-^', label="MAE")
            ax.legend(loc='best')
            ax.set_xscale("log")
            ax_global.set_xlabel("\n%s" % x_label)
            ax_global.set_ylabel("Quantification accuracy\n\n\n")
            plt.subplots_adjust(hspace=0.05)
    return x_axis, performance_group 


def combine_detection_and_quantification_heatmap(model, 
                                                 perf_crit=["global_accu", "rsquare"],
                                                 dataset="TOMAS",
                                                 real_val=False, show=False, tds_dir=None,
                                                 save=False):
    """This function combines the detection and quantification heatmap together
    Args:
        model: a list of string that specify the models
    """
    tt_pred, \
        [best_key_group, perf_val_detection_stat, tr_criteria, percentage] = get_spectra_performance_detection_heatmap(model, 
                                                                                                             perf_crit=perf_crit[0], 
                                                                                                             real_val=real_val, 
                                                                                                             dataset=dataset,
                                                                                                             show=False, 
                                                                                                             tds_dir=tds_dir, save=False)
    _, \
        [best_key_group, perf_val_quantification_stat, tr_criteria, percentage] = get_spectra_performance_quantification_heatmap(model=model, 
                                                                                                                 perf_crit=perf_crit[1],
                                                                                                                 real_val=real_val, 
                                                                                                                 dataset=dataset,
                                                                                                                 show=False, 
                                                                                                                 tds_dir=tds_dir, save=False)
    # perf_val_detection_stat = [np.random.random([4, 10, 10]) for i in model]
    # perf_val_quantification_stat = [np.random.random([4, 10, 10]) for i in model]
    if dataset == "TOMAS":
        percentage = [0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
    else:
        percentage = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    # tr_criteria = ["top_peak", "top_std", "top_mean", "top_diff"]
    tr_criteria = ["top_peak"]
    model_title = ["Xception", "U-CNN", "ResNet"]
    selection_method = ["Peak", "Std", "Mean", "Diff"]
    heatmap_title = ["Validation detection performance (logarithm of the cross entropy loss)", 
                     "Validation quantification performance (Rsquare of the log-transformed concentration)"]
    nrow = len(model) * 2 
    ncol = len(tr_criteria)
    fig = vu.give_figure_specify_size(1.0, 2.6 / 3 * 2)
    ax_global = vu.ax_global_get(fig)
    ax_global.set_xticks([])
    ax_global.set_yticks([])
    perf_group = [perf_val_detection_stat, perf_val_quantification_stat]
    ax_group = []
    for i, s_perf in enumerate(perf_group):
        base = len(model) * ncol * i 
        ax_group = []
        if perf_crit[i] == "rsquare" or perf_crit[i] == "log_rsquare":
            s_perf = np.maximum(s_perf, 0)
            vmin, vmax = 0.0, 1.0
        if perf_crit[i] == "validation_loss" or perf_crit[i] == "mae":
            s_perf = np.log(s_perf + 1e-1)
            vmin, vmax = -4.0, 4.0
            # s_perf = s_perf
        cmap = ["Blues_r" if perf_crit[i] == "validation_loss" else "Blues"][0]
        
        for j, s_model in enumerate(model):
            ax_fake = fig.add_subplot(nrow, 1, j + 1 + len(model) * i, frameon=False)
            ax_fake.set_xticks([])
            ax_fake.set_yticks([])
            ax_fake.set_title(heatmap_title[i] + " (%s)" % model_title[j], pad=2)
            for q, s_crit in enumerate(tr_criteria):
                ax = fig.add_subplot(nrow, ncol, base + j * len(tr_criteria) + q + 1)
                if q == 0:
                    yticks = ["%.1f" % (v * 100) for v in percentage]
                else:
                    yticks = ["" for v in percentage]
                if base == len(model) * ncol and j == len(model) - 1:
                    xticks = ["%.1f" % (v * 100) for v in percentage]
                else:
                    xticks = ["" for v in percentage]
                im=sns.heatmap(s_perf[j][q], annot=False, cmap=cmap, 
                            xticklabels=xticks, yticklabels=yticks,
                            cbar=False, ax=ax, vmin=vmin, vmax=vmax)
                if q != 0:
                    ax.set_yticks([])
                if i * len(model) + j != nrow - 1:
                    ax.set_xticks([])
                if base == 0 and j == 0:
                    ax.set_title(selection_method[q] + "\n")
                ax_group.append(ax)
        cax = plt.axes([0.92, 0.13 + 0.75 / 1.9 * (1 - i), 0.025, 0.75 / 2.2])
        mappable = im.get_children()[0]
        plt.colorbar(mappable, cax=cax, ax=ax_group)
    ax_global.set_xlabel("\n\n\nPercentage of spectra used at testing (%)")
    ax_global.set_ylabel("Percentage of spectra used at training (%)\n\n\n")
    plt.subplots_adjust(wspace=0.05, hspace=0.2)
    if save:
        plt.savefig(tds_dir + "/combined_%s_detection_and_quantification_heatmap.pdf" % dataset, pad_inches=0, bbox_inches='tight')
        
                
def get_spectra_performance_detection_heatmap(model=["xception"], perf_crit="global_accu", 
                                              real_val=False,
                                              dataset="TOMAS",
                                              show=True,
                                              tds_dir=None, save=False):
    version_detection={}
    if dataset == "TOMAS":
        version_detection["xception"] = [0, 1, 2, 3, 4]
        version_detection["unified_cnn"] = [0 , 3, 10, 11, 12]
        version_detection["resnet"] = [0, 1, 2, 3, 4]
    elif dataset == "PA":
        version_detection["xception"] = [0, 1, 2, 3, 4]
        version_detection["unified_cnn"] = [0, 1, 2, 3, 4] #[0, 1, 2, 3, 4]
        version_detection["resnet"] = [0, 2, 3, 4] #[0, 1, 2, 3, 4]
    elif dataset == "DNP":
        version_detection["xception"] = [5, 6, 7, 8, 9]
        version_detection["unified_cnn"] = [5, 6, 7, 8, 9]
        version_detection["resnet"] = [5, 6, 7, 8, 9]
        
    if dataset == "TOMAS":
        percentage = [0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
    else:
        percentage = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    # normalization = "none"
    normalization = "max" if dataset == "TOMAS" else "none" 
    selection_method = ["Peak", "Std", "Mean", "Diff"]
    # tr_criteria = ["top_peak", "top_std", "top_mean", "top_diff"]
    tr_criteria = ["top_peak"]
    nrow, ncol = len(model), len(tr_criteria)
    perf_val_stat, perf_tt_stat = [], []
    tt_pred, best_key_group = [], []
    for i, s_model in enumerate(model):
        perf_val_stat_perf_model, perf_tt_stat_perf_model, val_accu_per_model = [], [], []
        tt_pred_per_model, tt_accu_per_model, best_key_per_model, tt_pred_all_crit = [], [], [], []
        obj_detect = GetSpectraPerformanceTOMAS(s_model, True, False, version_detection[s_model], 
                                                True, perf_crit, normalization=normalization, real_val=real_val, dataset=dataset)
        for j, s_crit in enumerate(tr_criteria):
            val_collect, tt_collect, avg_val_perf, avg_tt_perf = obj_detect._find_best_index_baseon_aggregate_perf(s_crit, 
                                                                                                      perf_crit=perf_crit,
                                                                                                      show=False)            
            perf_val_stat_perf_model.append(avg_val_perf[0])
            perf_tt_stat_perf_model.append(avg_tt_perf)
            tt_pred_per_model.append(np.array(tt_collect[2])[:, 0])
            tt_label = np.concatenate(obj_detect.label[1], axis=0)
            tt_conc = np.concatenate(obj_detect.concentration[1], axis=0)
            best_key_per_model.append(val_collect[1])
            tt_accu_per_model.append(tt_collect[1])
            val_accu_per_model.append(val_collect[2])
        if perf_crit == "global_accu" or perf_crit == "f1_score" or perf_crit == "false_positive":
            _bb_index = np.argmax(val_accu_per_model)
        elif perf_crit == "validation_loss":
            _bb_index = np.argmin(val_accu_per_model)
        tt_pred.append(tt_pred_per_model[_bb_index])
        best_key_group.append(best_key_per_model[_bb_index])
        perf_val_stat.append(perf_val_stat_perf_model)
        perf_tt_stat.append(perf_tt_stat_perf_model)
    color_use = ['r', 'g', 'b']
    if show:
        fig = vu.give_figure_specify_size(0.5)
        ax = fig.add_subplot(111)
        x_ticks = np.arange(len(np.unique(tt_conc)) + 1)[1:]
        for i, s_model in enumerate(model):
            correct_or_wrong = (np.argmax(tt_pred[i], axis=-1) == tt_label).astype(np.int32) * 100
            accu_per_conc, _, _ = vu.show_accuracy_over_concentration(tt_pred[i], np.argmax(tt_pred[i], axis=-1), 
                                                                    tt_label, tt_conc, show=False, save=False)
            ax.plot(x_ticks, np.array(accu_per_conc) * 100, color_use[i], label=s_model)
            for j, s_conc in enumerate(np.unique(tt_conc)):
                index = np.where(tt_conc == s_conc)[0]
                for q in index:
                    ax.plot(x_ticks[j], correct_or_wrong[q], ls='', marker='.')
        ax.legend(loc='best')
        ax.set_xscale("log")
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(["%.3f" % v for v in np.unique(tt_conc)], rotation=90)
        ax.set_xlabel("Concentration (uM)")
        ax.set_ylabel("Detection accuracy (%)")
        if save:
            plt.savefig(tds_dir + "/detection_accuracy_tomas.pdf", pad_inches=0, bbox_inches='tight')
        perf_title = ["validation_heatmap", "testing_heatmap"]
        print("validation heatmap", np.shape(perf_val_stat), np.shape(perf_tt_stat))
        for q, perf_stat in enumerate([perf_val_stat, perf_tt_stat]):
            print("the heatmap shape", np.shape(perf_stat))
            if perf_crit == "validation_loss" and q == 0:
                perf_stat = np.log(np.array(perf_stat) + 1e-3)
            vmin, vmax = np.min(np.reshape(perf_stat, [-1])), np.max(np.reshape(perf_stat, [-1]))
            print(vmin, vmax)
            fig = vu.give_figure_specify_size(1, 2.6 / 6 * len(model))
            ax_global = vu.ax_global_get(fig)
            ax_global.set_xticks([])
            ax_global.set_yticks([])
            ax_group = []
            for i, s_model in enumerate(model):
                axt = fig.add_subplot(nrow, 1, i+1, frameon=False)
                axt.set_xticks([])
                axt.set_yticks([])
                axt.set_title(s_model)
                for j, s_crit in enumerate(tr_criteria):        
                    ax = fig.add_subplot(nrow, ncol, i * ncol + j + 1)
                    if j == 0:
                        yticklabel = ["%.1f" % (v * 100) for v in percentage]
                    else:
                        yticklabel = ["" for v in percentage]
                    if i == nrow - 1:
                        xticklabel = ["%.1f" % (v * 100) for v in percentage]
                    else:
                        xticklabel = ["" for v in percentage]
                    im = sns.heatmap(perf_stat[i][j],
                        vmin=vmin,
                        vmax=vmax,
                        ax=ax,
                        yticklabels=yticklabel,
                        xticklabels=xticklabel,
                        cbar=False,
                        cmap="Blues",
                    )
                    if i == 0:
                        ax.set_title(selection_method[j] + "\n")
                    ax_group.append(ax)
            cax = plt.axes([0.92, 0.13, 0.025, 0.75])
            mappable = im.get_children()[0]
            plt.colorbar(mappable, cax=cax, ax=ax_group)
            ax_global.set_xlabel("\n\n\nPercentage of spectra used at testing (%)")
            ax_global.set_ylabel("Percentage of spectra used at training (%)\n\n\n")
            plt.subplots_adjust(wspace=0.05, hspace=0.2)
            if save:
                plt.savefig(tds_dir + "/detection_heatmap_%s_tomas.pdf" % (perf_title[q]), pad_inches=0, bbox_inches='tight')
    return tt_pred, [best_key_group, np.array(perf_val_stat) + 1e-8, tr_criteria, percentage]


def compare_concentration_prediction_curve(model_spectra=["xception"], perf_crit="rsquare",
                                           dataset="TOMAS",
                                           tds_dir="../rs_dataset/paper_figure/", save=False,
                                           return_value=False):
    """Compare predicted concentration over different models"""
    detection = False 
    quantification = True 
    if dataset == "TOMAS":
        lr = 0.08 
        targ_shape = [56, 56]
        version_u = [6338,  5353, 12885,  1746,  8543] #[]
        x_label_conc = ["Blank", "0.1nM", "1nM", "10nM", "100nM", "1"+r'$\mu$'+"M"]

    elif dataset == "DNP":
        lr = 0.006 
        targ_shape = [44, 44]
        version_u = [13579, 9860, 19800, 12266, 29055] #[13579, 9860, 19800, 12266, 29055]
        x_label_conc = ["Blank", "1nM", "10nM", "100nM", "1"+r'$\mu$'+"M", "10"+r'$\mu$'+"M"]

    elif dataset == "PA":
        lr = 0.0006 #4 
        targ_shape = [40, 40]
        version_u = [30541,   989, 31058, 18197, 30988] #[20402,  4964,  1064, 20042, 26410]#[25147, 13307, 15835 , 31058, 20402]
        x_label_conc = ["Blank", "10nM", "100nM", "1"+r'$\mu$'+"M", "10"+r'$\mu$'+"M"]
    x_label_conc = x_label_conc[1:]
    concentration_float = 1e-6 if dataset != "PA" else 1e-5 
    normalization = "none"
    quantification_loss = "mse"    
    model_title = ["Xception", "U-CNN", "ResNet", "ViT"]
    obj = GetTomasPerformanceVIT(detection, quantification, lr=lr, patch_size=2, 
                                 version_use=version_u, 
                                 normalization=normalization, concentration_float=concentration_float, 
                                 quantification_loss=quantification_loss,
                                 target_shape=targ_shape, 
                                 leave_method="leave_one_chip", dataset=dataset, 
                                 model_init="xavier", lr_schedule="cosine")
    vit_pred, conc, conc_update, _ = obj.get_quantification_performance(show=False, tds_dir=None, save=False)
    spectra_pred, _ = get_spectra_performance_quantification_heatmap(model=model_spectra, perf_crit=perf_crit, 
                                                                     real_val=False, 
                                                                     dataset=dataset,
                                                                     show=False, 
                                                                  tds_dir=None, save=False)
    color_use = ['r', 'g', 'b']
    conc_pred_stat = list(spectra_pred) + [vit_pred]
    if return_value:
        return conc_pred_stat, conc_update, x_label_conc
    fig = vu.give_figure_specify_size(1.0, 2.6/6)
    ax_global = vu.ax_global_get(fig)
    ax_global.set_xticks([])
    ax_global.set_yticks([])
    for i, s_model in enumerate(model_spectra + ["ViT"]):
        ax = fig.add_subplot(1, len(model_spectra) + 1, i+1)
        x_ticks, quan_stat = utils.get_rsquare_curve_stat(conc_update, np.array(conc_pred_stat[i]))
        _rsquare = vu.get_performance_baseon_crit_quantification(conc_update, np.array(conc_pred_stat[i]), "log_rsquare")
        ax.plot(x_ticks, quan_stat[:, 0], color='r')
        for j, s_conc in enumerate(np.unique(conc_update)):
            index = np.where(conc_update == s_conc)[0]                        
            rand_add = np.zeros([len(index)]) + np.random.normal(s_conc/5, 0.00001, size=[len(index)]) * np.random.randint(-1, 2, [len(index)])
            for i_index, q in enumerate(index):
                _pred = conc_pred_stat[i][q]
                ax.plot(x_ticks[j]+rand_add[i_index], _pred, 
                        marker='+', ls='', markersize=5, alpha=0.9)
        ax.plot(x_ticks, np.unique(conc_update), color='gray')
        ax.set_xscale("symlog", linthresh=np.sort(np.unique(conc_update))[1])
        ax.set_yscale("symlog", linthresh=np.sort(np.unique(conc_update))[1])
        if i != 0:
            ax.yaxis.set_major_formatter(plt.NullFormatter())
        # ax.set_title(model_title[i] + "\nRsquare (log(concentration)): %.2f" % ((_rsquare)))
        ax.set_title(model_title[i] + ": %.2f" % ((_rsquare)))
        ax.set_xticks(x_ticks)    
        ax.set_xticklabels(["%s" % v for v in x_label_conc], rotation=90)
        if i == 0:
            ax.set_yticks(x_ticks)
            ax.set_yticklabels(["%s" % v for v in x_label_conc])
        # ax.set_xticklabels(["%.4f" % v for v in np.unique(conc)], rotation=90)
        ax.set_ylim((-0.0001, np.max(np.reshape(conc_pred_stat, [-1])) + 5))
        # ax.set_yticks(quan_stat[:, -1])
        # ax.set_yticklabels(["%.4f" % v for v in quan_stat[:, -1]])
    plt.subplots_adjust(wspace=0.1)
    # ax_global.set_xlabel("\n\n\n\nConcentration (uM)")
    ax_global.set_xlabel("\n\n\n\nConcentration")
    ax_global.set_ylabel("Predicted concentration\n\n\n")    
    if save:
        plt.savefig(tds_dir + "/%s_concentration_prediction_comparison_%s.pdf" % (dataset, perf_crit), pad_inches=0, bbox_inches='tight')
    
    
def rand_jitter(arr):
    stdev = .01 * (max(arr) - min(arr))
    return arr + np.random.randn(len(arr)) * stdev


def combined_concentration_prediction_curve_with_ml(tds_dir="../rs_dataset/paper_figure/",
                                            save=False):
    dataset_group = ["TOMAS", "DNP", "PA"]
    dataset_title = ["4-NBT", "DNP", "PA"]
    model_spectra = ["xception", "unified_cnn", "resnet"]
    perf_crit = "log_rsquare"
    model_title = ["Random Forest", "Decision Tree", "SVM", "GradientBoost", 
                   "Xception", "U-CNN", "ResNet", "ViT"]
    perf_group, conc_group, x_label_group = [], [], []
    for data_index, s_data in enumerate(dataset_group):
        pred_machine, conc_machine, method_machine, perf_machine = mm.get_concentration_perf(s_data, show=False, 
                                                                                    tds_dir=None, save=False,                            
                                                                                    return_value=True)
        conc_pred_stat, conc_update, x_label_conc = compare_concentration_prediction_curve(model_spectra, perf_crit, 
                                                                                           dataset=s_data, tds_dir=tds_dir,
                                                                                           save=False, return_value=True)
        perf_group.append(np.concatenate([np.exp(pred_machine), conc_pred_stat], axis=0))
        conc_group.append(conc_update)
        x_label_group.append(x_label_conc)    
    fig = vu.give_figure_specify_size(1.0, 2.0)
    ax_global = vu.ax_global_get(fig)
    ax_global.set_xticks([])
    ax_global.set_yticks([])
    nrow, ncol = len(model_spectra) + 1 + len(method_machine), len(dataset_group)
    for i in range(nrow):
        for j in range(ncol):
            if i == 0:
                ax_g = fig.add_subplot(1, ncol, j+1, frameon=False)
                ax_g.set_xticks([])
                ax_g.set_yticks([])
                ax_g.set_title(dataset_title[j]+"\n")
            s_pred = perf_group[j][i]
            x_ticks = np.unique(conc_group[j])            
            ax = fig.add_subplot(nrow, ncol, i * ncol + j + 1)            
            _, quan_stat = utils.get_rsquare_curve_stat(conc_group[j], np.array(s_pred))
            _rsquare = utils.calc_rsquare_log(conc_group[j], np.array(s_pred))            
            ax.plot(x_ticks, quan_stat[:, 0], color='r')
            for q, s_conc in enumerate(x_ticks):
                index = np.where(conc_group[j] == s_conc)[0]                        
                rand_add = np.zeros([len(index)]) + np.random.normal(s_conc/5, 0.00001, size=[len(index)]) * np.random.randint(-1, 2, [len(index)])
                for i_index, s_i in enumerate(index):
                    _pred = s_pred[s_i]
                    ax.plot(s_conc+rand_add[i_index], _pred, 
                            marker='+', ls='', markersize=5, alpha=0.9)
            ax.plot(x_ticks, x_ticks, color='gray')
            ax.set_xscale("symlog", linthresh=np.sort(x_ticks)[1])
            ax.set_yscale("symlog", linthresh=np.sort(x_ticks)[1])
            ax.set_title(model_title[i] + ":%.2f" % ((_rsquare)))
            ax.set_yticks(x_ticks)
            ax.set_yticklabels(["%s" % v for v in x_label_group[j]])
            if i == nrow-1:
                ax.set_xticks(x_ticks)    
                ax.set_xticklabels(["%s" % v for v in x_label_group[j]], rotation=90)
            else:
                ax.xaxis.set_major_formatter(plt.NullFormatter())
            ax.set_ylim((-0.0001, np.max(np.reshape(conc_pred_stat, [-1])) + 5))
            
    plt.subplots_adjust(wspace=0.35, hspace=0.48)
    ax_global.set_xlabel("\n\n\n\nConcentration")
    ax_global.set_ylabel("Predicted concentration\n\n\n\n")    
    if save:
        plt.savefig(tds_dir+"/combined_quantification_results.pdf", pad_inches=0, bbox_inches='tight')
        
        
def combined_detection_prediction_curve_with_ml(tds_dir="../rs_dataset/paper_figure/",
                                        save=False):
    dataset_group = ["TOMAS", "DNP", "PA"]
    dataset_title = ["4-NBT", "DNP", "PA"]
    model_spectra = ["xception", "unified_cnn", "resnet"]
    perf_crit = "validation_loss"
    model_title = ["KNN", "Random Forest", "SVM", "GradientBoost", 
                  "Xception", "U-CNN", "ResNet", "ViT"]
    pred_group, conc_update_group, x_label_conc_g, tt_label_g = [], [], [], []
    for data_index, s_data in enumerate(dataset_group):
        print(model_spectra, s_data, perf_crit)
        machine_pred_prob, machine_pred_label, _, _, machine_method = mm.get_detection_perf(s_data)        
        conc_pred_stat,  conc_update, x_label_conc, \
            tt_label, conc = compare_detection_prediction_curve(model_spectra, perf_crit, 
                                                                dataset=s_data, tds_dir=tds_dir, 
                                                                save=False,
                                                                return_value=True) 
        pred_g = np.concatenate([machine_pred_prob, conc_pred_stat], axis=0)
        pred_group.append(pred_g)
        conc_update_group.append(conc_update)
        x_label_conc_g.append(x_label_conc)
        tt_label_g.append(tt_label)
    fig = vu.give_figure_specify_size(1.0, 2.0)
    ax_global = vu.ax_global_get(fig)
    ax_global.set_xticks([])
    ax_global.set_yticks([])
    nrow, ncol = len(model_spectra) + 1 + len(machine_method), len(dataset_group)
    for i in range(nrow):
        for j in range(ncol):
            if i == 0:
                ax_g = fig.add_subplot(1, ncol, j+1, frameon=False)
                ax_g.set_xticks([])
                ax_g.set_yticks([])
                ax_g.set_title(dataset_title[j]+"\n")
            x_axis = np.unique(conc_update_group[j])
            
            ax = fig.add_subplot(nrow, ncol, i * ncol + j + 1)
            if i <= 3:
                pred = np.array(pred_group[j][i])
            else:
                pred = softmax(np.array(pred_group[j][i]), axis=-1)
            pred_label = np.argmax(pred, axis=-1)
            s_label=tt_label_g[j]
            correct_or_wrong = (pred_label == s_label).astype(np.int8)
            global_accu = np.sum(correct_or_wrong) / len(s_label)
            avg_prob = vu.calc_avg_prob_per_concentration(pred, s_label, conc_update_group[j])
            
            ax.plot(x_axis, avg_prob, color='r', marker='.')
            for q, s_conc in enumerate(x_axis):
                index = np.where(conc_update_group[j] == s_conc)[0]
                cor_index = index[np.where(correct_or_wrong[index] == 1)[0]]
                wor_index = index[np.where(correct_or_wrong[index] == 0)[0]]
                if x_axis[j] >= 0.0001:
                    rand_add = np.random.random([len(index)]) * x_axis[q] / 2 * np.random.randint(-1, 2, [len(index)])
                else:
                    rand_add = np.zeros([len(index)])
                if len(cor_index) > 0:
                    for m, s_cor_index in enumerate(cor_index):                
                        ax.plot(x_axis[q] + rand_add[m], np.max(pred[s_cor_index]), 
                                marker='+', ls='', color='g', markersize=5)
                if len(wor_index) > 0:
                    for m, s_wor_index in enumerate(wor_index):                
                        ax.plot(x_axis[q] + rand_add[-m-1], 1.0 - np.max(pred[s_wor_index]), 
                                marker='x', ls='', color='b', markersize=5)
            ax.plot(x_axis, [0.5 for _ in x_axis], ls=':', color='gray')
            ax.text(x_axis[-2] -0.075, 0.7, 'correct', color='g')
            ax.text(x_axis[-2] -0.075, 0.2, 'wrong', color='b')
            ax.set_xscale("log")
            ax.set_ylim((-0.05, 1.08))
            if j != 0:
                ax.yaxis.set_major_formatter(plt.NullFormatter())
#                 ax.yaxis.set_minor_formatter(plt.NullFormatter())
            if i != nrow - 1:
                ax.set_xticks(x_axis)                
                ax.xaxis.set_major_formatter(plt.NullFormatter())
            else:
                ax.set_xticks(x_axis)    
                ax.set_xticklabels(["%s" % v for v in x_label_conc_g[j]], rotation=90)

            ax.set_title(model_title[i] + ": " + r'$\frac{%d}{%d}$' % (np.sum(correct_or_wrong), len(tt_label)) + "=%.2f" % global_accu)
    plt.subplots_adjust(wspace=0.08, hspace=0.41)
    ax_global.set_xlabel("\n\n\n\nConcentration")
    ax_global.set_ylabel("Predicted probability\n\n")   
    if save:
        plt.savefig(tds_dir + "/combined_detection_map.pdf", pad_inches=0, bbox_inches='tight') 


def combined_concentration_prediction_curve(tds_dir="../rs_dataset/paper_figure/",
                                            save=False):
    dataset_group = ["DNP", "PA", "TOMAS"]
    dataset_title = ["DNP", "PA", "4-NBT"]
    model_spectra = ["xception", "unified_cnn", "resnet"]
    perf_crit = "log_rsquare"
    model_title = ["Xception", "U-CNN", "ResNet", "ViT"]
    fig = vu.give_figure_specify_size(1.0, 1.8)
    ax_global = vu.ax_global_get(fig)
    ax_global.set_xticks([])
    ax_global.set_yticks([])
    nrow, ncol = len(dataset_group), len(model_spectra) + 1
    nrow, ncol = len(model_spectra) + 1 + 4, len(dataset_group)
    
    
    for data_index, s_data in enumerate(dataset_group):
        pred_machine, conc_machine, method_machine, perf_machine = mm.get_concentration_perf(dataset, show=False, 
                                                                                    tds_dir=None, save=False,                            
                                                                                    return_value=False)
        conc_pred_stat, conc_update, x_label_conc = compare_concentration_prediction_curve(model_spectra, perf_crit, 
                                                                                           dataset=s_data, tds_dir=tds_dir,
                                                                                           save=False, return_value=True)
        ax_g = fig.add_subplot(nrow, 1, data_index+1, frameon=False)        
        ax_g.set_xticks([])
        ax_g.set_yticks([])
        ax_g.set_title(dataset_title[data_index] + "\n")
        x_ticks = np.unique(conc_update)
        for i, s_model in enumerate(model_spectra + ["ViT"]):
            ax = fig.add_subplot(nrow, ncol, data_index * ncol + i+1)
            _, quan_stat = utils.get_rsquare_curve_stat(conc_update, np.array(conc_pred_stat[i]))
            _rsquare = vu.get_performance_baseon_crit_quantification(conc_update, np.array(conc_pred_stat[i]), "log_rsquare")
            ax.plot(x_ticks, quan_stat[:, 0], color='r')
            for j, s_conc in enumerate(np.unique(conc_update)):
                index = np.where(conc_update == s_conc)[0]                        
                rand_add = np.zeros([len(index)]) + np.random.normal(s_conc/5, 0.00001, size=[len(index)]) * np.random.randint(-1, 2, [len(index)])
                for i_index, q in enumerate(index):
                    _pred = conc_pred_stat[i][q]
                    ax.plot(x_ticks[j]+rand_add[i_index], _pred, 
                            marker='+', ls='', markersize=5, alpha=0.9)
            ax.plot(x_ticks, np.unique(conc_update), color='gray')
            ax.set_xscale("symlog", linthresh=np.sort(np.unique(conc_update))[1])
            ax.set_yscale("symlog", linthresh=np.sort(np.unique(conc_update))[1])
            if i != 0:
                ax.yaxis.set_major_formatter(plt.NullFormatter())
            ax.set_title(model_title[i] + ": %.2f" % ((_rsquare)))
            ax.set_xticks(x_ticks)    
            ax.set_xticklabels(["%s" % v for v in x_label_conc], rotation=90)
            if i == 0:
                ax.set_yticks(x_ticks)
                ax.set_yticklabels(["%s" % v for v in x_label_conc])
            ax.set_ylim((-0.0001, np.max(np.reshape(conc_pred_stat, [-1])) + 5))
            
    plt.subplots_adjust(wspace=0.1, hspace=0.53)
    ax_global.set_xlabel("\n\n\n\nConcentration")
    ax_global.set_ylabel("Predicted concentration\n\n\n")    
    if save:
        plt.savefig(tds_dir+"/combined_quantification_results.pdf", pad_inches=0, bbox_inches='tight')


def combined_detection_prediction_curve(tds_dir="../rs_dataset/paper_figure/",
                                        save=False):
    dataset_group = ["DNP", "PA", "TOMAS"]
    dataset_title = ["DNP", "PA", "4-NBT"]
    model_spectra = ["xception", "unified_cnn", "resnet"]
    perf_crit = "validation_loss"
    model_title = ["Xception", "U-CNN", "ResNet", "ViT"]
    fig = vu.give_figure_specify_size(1.0, 1.8)
    ax_global = vu.ax_global_get(fig)
    ax_global.set_xticks([])
    ax_global.set_yticks([])
    nrow, ncol = len(dataset_group), len(model_spectra) + 1
    for data_index, s_data in enumerate(dataset_group):
        print(model_spectra, s_data, perf_crit)
        conc_pred_stat,  conc_update, x_label_conc, \
            tt_label, conc = compare_detection_prediction_curve(model_spectra, perf_crit, 
                                                                dataset=s_data, tds_dir=tds_dir, 
                                                                save=False,
                                                                return_value=True) 
        ax_g = fig.add_subplot(nrow, 1, data_index+1, frameon=False)
        ax_g.set_xticks([])
        ax_g.set_yticks([])
        ax_g.set_title(dataset_title[data_index] + "\n")
        x_axis = np.unique(conc_update)
        for i, s_model in enumerate(list(model_spectra) + ["ViT"]):
            ax = fig.add_subplot(nrow, ncol, data_index * ncol + i+1)
            pred = softmax(np.array(conc_pred_stat[i]), axis=-1)
            # print(np.max(pred, axis=-1))
            pred_label = np.argmax(pred, axis=-1)
            correct_or_wrong = (pred_label == tt_label).astype(np.int8)
            global_accu = np.sum(correct_or_wrong) / len(tt_label)
            avg_prob = vu.calc_avg_prob_per_concentration(pred, tt_label, conc)
            ax.plot(x_axis, avg_prob, color='r', marker='.')
            for j, s_conc in enumerate(np.unique(conc_update)):
                index = np.where(conc_update == s_conc)[0]
                cor_index = index[np.where(correct_or_wrong[index] == 1)[0]]
                wor_index = index[np.where(correct_or_wrong[index] == 0)[0]]
                if x_axis[j] >= 0.0001:
                    rand_add = np.random.random([len(index)]) * x_axis[j] / 2 * np.random.randint(-1, 2, [len(index)])
                else:
                    rand_add = np.zeros([len(index)])
                if len(cor_index) > 0:
                    for m, q in enumerate(cor_index):                
                        ax.plot(x_axis[j] + rand_add[m], np.max(pred[q]), 
                                marker='+', ls='', color='g', markersize=5)
                if len(wor_index) > 0:
                    for m, q in enumerate(wor_index):                
                        ax.plot(x_axis[j] + rand_add[-m-1], 1.0 - np.max(pred[q]), 
                                marker='x', ls='', color='b', markersize=5)
            ax.plot(x_axis, [0.5 for _ in x_axis], ls=':', color='gray')
            ax.text(np.unique(conc_update)[-2] -0.075, 0.7, 'correct', color='g')
            ax.text(np.unique(conc_update)[-2] -0.075, 0.2, 'wrong', color='b')
            ax.set_xscale("log")
            ax.set_ylim((-0.05, 1.05))
            if i != 0:
                ax.yaxis.set_major_formatter(plt.NullFormatter())
                ax.yaxis.set_minor_formatter(plt.NullFormatter())
            ax.set_title(model_title[i] + ": " + r'$\frac{%d}{%d}$' % (np.sum(correct_or_wrong), len(tt_label)) + "=%.2f" % global_accu)
            ax.set_xticks(x_axis)    
            ax.set_xticklabels(["%s" % v for v in x_label_conc], rotation=90)
    plt.subplots_adjust(wspace=0.1, hspace=0.53)
    ax_global.set_xlabel("\n\n\n\nConcentration")
    ax_global.set_ylabel("Predicted probability\n\n")   
    if save:
        plt.savefig(tds_dir + "/combined_detection_map.pdf", pad_inches=0, bbox_inches='tight') 


def compare_detection_prediction_curve(model_spectra=["xception"], perf_crit="global_accu",
                                       dataset="TOMAS", 
                                      tds_dir="../rs_dataset/paper_figure/", save=False,
                                      return_value=False):
    """Compare predicted concentration over different models"""
    detection = True 
    quantification = False 
    if dataset == "TOMAS":    
        lr = 0.008
        v_use = [30582, 1066, 30186, 4731, 2814]
        targ_shape = [56, 56]
        x_label_conc = ["Blank", "0.1nM", "1nM", "10nM", "100nM", "1"+r'$\mu$'+"M"]
    elif dataset == "DNP":
        lr = 0.005 
        v_use = [32619, 9093,31050,10637, 31797] # [19570, 25646, 6714, 13117, 10573] when normalization="max"
        targ_shape = [44, 44]
        x_label_conc = ["Blank", "1nM", "10nM", "100nM", "1"+r'$\mu$'+"M", "10"+r'$\mu$'+"M"]
    elif dataset == "PA":
        lr = 0.0006
        v_use = [12759, 5260, 20202, 20887, 4209]
        targ_shape = [40, 40]
        x_label_conc = ["Blank", "10nM", "100nM", "1"+r'$\mu$'+"M", "10"+r'$\mu$'+"M"]
        
    concentration_float = 1e-6 if dataset != "PA" else 1e-5
    normalization="none"
    # normalization = "max" if dataset != "PA" else "none"
    quantification_loss = "none"    
    model_title = ["Xception", "U-CNN", "ResNet", "ViT"]
    obj = GetTomasPerformanceVIT(detection, quantification, lr=lr, patch_size=2, 
                                 version_use=v_use, 
                                 normalization=normalization, concentration_float=concentration_float, 
                                 quantification_loss=quantification_loss,
                                 target_shape=targ_shape, 
                                 leave_method="leave_one_chip", dataset=dataset,
                                 model_init="xavier", lr_schedule="cosine")
    vit_pred, pred_label, tt_label, conc, perf = obj.get_detection_performance(show=False, tds_dir=None, save=False)
    spectra_pred, _ = get_spectra_performance_detection_heatmap(model=model_spectra, perf_crit=perf_crit,
                                                                dataset=dataset,
                                                                show=False, tds_dir=None, save=False)
    conc_update = conc.copy()
    conc_update[conc_update == 0.0] = concentration_float
    color_use = ['r', 'g', 'b']
    conc_pred_stat = list(spectra_pred) + [vit_pred]
    if return_value:
        return conc_pred_stat,  conc_update, x_label_conc, tt_label, conc
    fig = vu.give_figure_specify_size(1.0, 2.6/6)
    ax_global = vu.ax_global_get(fig)
    ax_global.set_xticks([])
    ax_global.set_yticks([])
    x_axis = np.unique(conc_update)
    for i, s_model in enumerate(list(model_spectra) + ["ViT"]):
        ax = fig.add_subplot(1, len(model_spectra) + 1, i+1)
        pred = softmax(np.array(conc_pred_stat[i]), axis=-1)
        print(np.max(pred, axis=-1))
        pred_label = np.argmax(pred, axis=-1)
        correct_or_wrong = (pred_label == tt_label).astype(np.int8)
        global_accu = np.sum(correct_or_wrong) / len(tt_label)
        _f1_score = f1_score(tt_label, pred_label)
        avg_prob = vu.calc_avg_prob_per_concentration(pred, tt_label, conc)
        ax.plot(x_axis, avg_prob, color='r', marker='.')
        for j, s_conc in enumerate(np.unique(conc_update)):
            index = np.where(conc_update == s_conc)[0]
            cor_index = index[np.where(correct_or_wrong[index] == 1)[0]]
            wor_index = index[np.where(correct_or_wrong[index] == 0)[0]]
            if x_axis[j] >= 0.0001:
                rand_add = np.random.random([len(index)]) * x_axis[j] / 2 * np.random.randint(-1, 2, [len(index)])
            else:
                rand_add = np.zeros([len(index)])
            if len(cor_index) > 0:
                for m, q in enumerate(cor_index):                
                    ax.plot(x_axis[j] + rand_add[m], np.max(pred[q]), 
                            marker='+', ls='', color='g', markersize=5)
            if len(wor_index) > 0:
                for m, q in enumerate(wor_index):                
                    ax.plot(x_axis[j] + rand_add[-m-1], 1.0 - np.max(pred[q]), 
                            marker='x', ls='', color='b', markersize=5)
        ax.plot(x_axis, [0.5 for _ in x_axis], ls=':', color='gray')
        ax.text(np.unique(conc_update)[-2] -0.075, 0.7, 'correct', color='g')
        ax.text(np.unique(conc_update)[-2] -0.075, 0.2, 'wrong', color='b')
        ax.set_xscale("log")
        # ax.set_yscale("symlog", linthresh=0.001)
        ax.set_ylim((-0.05, 1.05))
        if i != 0:
            ax.yaxis.set_major_formatter(plt.NullFormatter())
            ax.yaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_title(model_title[i] + ": " + r'$\frac{%d}{%d}$' % (np.sum(correct_or_wrong), len(tt_label)) + "=%.2f" % global_accu)
        # ax.set_title(model_title[i] + ": %.2f" % (global_accu * 100))
        ax.set_xticks(x_axis)    
        ax.set_xticklabels(["%s" % v for v in x_label_conc], rotation=90)
        # ax.set_xticklabels(["%.4f" % v for v in np.unique(conc)], rotation=90)
    
    plt.subplots_adjust(wspace=0.06)
    ax_global.set_xlabel("\n\n\n\nConcentration")
    ax_global.set_ylabel("Predicted probability\n\n")    

    if save:
        print(tds_dir)
        plt.savefig(tds_dir + "/%s_detection_prediction_comparison.pdf" % dataset, pad_inches=0, bbox_inches='tight')
        

def produce_vit_model_architecture_figure(tds_dir=None, save=False):
    from einops.layers.torch import Rearrange
    target_shape = [54, 54]
    sers_maps, label, concentration, wavenumber, mapsize = read_tomas.prepare_tomas_data(target_shape, testing=True)
    s_map = sers_maps[-6:-5]
    imh, imw, ch = np.shape(s_map)[1:]
    patch_height, patch_width = 18, 18
    num_p_h, num_p_w = imh // patch_height, imw // patch_width
    split_layer = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width)
    s_out = split_layer(torch.from_numpy(s_map).permute(0, 3, 1, 2))
    s_out_npy = s_out.detach().cpu().numpy()
    s_out_npy = np.reshape(s_out_npy, [1, num_p_h * num_p_w, patch_height, patch_width, ch])[0]
    for i in range(imh // patch_height):
        for j in range(imw // patch_width):
            fig = plt.figure(figsize=(3, 3), frameon=False)
            ax = fig.add_subplot(111)
            num = i * 3 + j
            ax.imshow(np.mean(s_out_npy[num], axis=-1))
            ax.set_xticks([])
            ax.set_yticks([])
            print(i, j, np.shape(np.mean(s_out_npy[num], axis=-1)))
            if save:
                plt.savefig(tds_dir + "/patch_h_%d_w_%d_for_vit.jpg" % (i, j), pad_inches=0, bbox_inches='tight', 
                            dpi=500)
        # plt.close('all')
    # return s_map, s_out_npy 


def produce_spectra_model_architecture_figure(tds_dir=None, save=False):
    target_shape = [56, 56]       
    sers_maps, label, concentration, wavenumber, mapsize = read_tomas.prepare_tomas_data(target_shape, 
                                                                                         testing=True)
    s_map = sers_maps[-6:-5]
    extracted_spectra, info = psd.simple_selection(s_map, wavenumber, 0.05, "top_peak", "TOMAS")
    map_region = [[1071, 1083], [1324, 1336], [1566, 1575]]
    map_select = read_tomas.get_map_at_specified_regions(s_map, map_region, wavenumber)
    nrow, ncol = 1, 3
    fig = plt.figure(figsize=(8, 1.4))
    spec5 = fig.add_gridspec(ncols=ncol, nrows=nrow, 
                             width_ratios=[1] + [2.4, 2.4], 
                             wspace=0.15)
    ax = fig.add_subplot(spec5[0,0])
    ax.imshow(map_select[0])
    ax.set_xlabel("SERS map")

    ax = fig.add_subplot(spec5[0,1])
    s_map_reshape = np.reshape(s_map, [-1, len(wavenumber)])
    for s in s_map_reshape:
        ax.plot(wavenumber, s, lw=0.4, alpha=0.2, color='gray')

    for v in s_map_reshape[info[0][0]]:
        ax.plot(wavenumber, v, lw=0.4, color='red', alpha=0.5)

    for s_map in map_region:
        ax.fill_betweenx([np.min(s_map_reshape), np.max(s_map_reshape)], 
                         [s_map[0], s_map[0]], [s_map[1], s_map[1]], color='g', 
                         alpha=0.5)
    ax.set_xlabel("Wavenumber (cm" + r"$^{-1}$" + ")")
    ax.yaxis.set_major_formatter(plt.NullFormatter())

    ax = fig.add_subplot(spec5[0, 2])
    ax.plot(wavenumber, extracted_spectra[0], color='r')

    ax.set_xlabel("Wavenumber (cm" + r"$^{-1}$" + ")")
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    if save:
        plt.savefig(tds_dir+"/spectra_architecture_figure.jpg", pad_inches=0, bbox_inches='tight', 
                    dpi=600)


class RelocateSelectionSersMap(object):
    def __init__(self, dataset, quantile=0.995):
        self.dataset = dataset 
        summary = pickle.load(open("../exp_data/VIT/%s/detection_False_quantification_True/%s.obj" % (dataset, dataset), "rb"))
        sers_maps = np.array(summary["sers_maps"])
        attention = summary["attention_map"]
        wavenumber = summary["wavenumber"][0]
        concentration = np.array(summary["concentration"])
        self.num_measurement, self.imh, self.imw, self.ch = np.shape(sers_maps)
        if self.dataset == "TOMAS":
            self.peak_loc = [1081, 1571, 1334]
        elif self.dataset == "DNP":
            self.peak_loc = [830, 1320]
        elif self.dataset == "PA":
            self.peak_loc = [820, 1330]
        if self.dataset == "DNP":
            x_label_conc = ["Blank", "1nM", "10nM", "100nM", "1"+r'$\mu$'+"M", "10"+r'$\mu$'+"M"]
        elif self.dataset == "TOMAS":
            x_label_conc = ["Blank", "0.1nM", "1nM", "10nM", "100nM", "1"+r'$\mu$'+"M"]
        elif self.dataset == "PA":
            x_label_conc = ["Blank", "10nM", "100nM", "1"+r'$\mu$'+"M", "10"+r'$\mu$'+"M"]
        self.x_label_conc = x_label_conc
        perc_use = [0.05, 0.20, 1.0]
        spectra_g, map_peak_locations = [], []
        fake_map = np.zeros([len(perc_use), self.num_measurement, self.imh*self.imw])
        for i, i_perc in enumerate(perc_use):
            spectra, [sort_index, map_peak_loc] = psd.simple_selection(sers_maps, wavenumber, i_perc, 
                                                       "top_peak", dataset)
            spectra_g.append(spectra)
            for j in range(self.num_measurement):
                fake_map[i, j, sort_index[j].astype("int32")] = 1
            if i == 0:
                map_peak_locations.append(map_peak_loc)
        
        self.perc_use = perc_use 
        self.sers_maps = sers_maps 
        self.attention = attention 
        self.map_peak_locations = np.squeeze(map_peak_locations, 0) 
        self.concentration = concentration 
        self.spectra_attention = self.get_spectra_from_attention(quantile)
        self.fake_map = fake_map 
        self.wavenumber = wavenumber 
        self.spectra_g = spectra_g
        
    def get_spectra_from_attention(self, quantile):
        spectra_attention = []
        for i, s_map in enumerate(self.attention):
            s_sers_map = np.reshape(self.sers_maps[i], [self.imh * self.imw, self.ch])
            threshold=np.quantile(np.reshape(s_map, [-1]), quantile)
            bool_mask = (s_map >= threshold).astype(np.int8)
            index_use = np.logical_and((np.reshape(bool_mask, [-1]) == 1), 
                                      np.sum(s_sers_map, axis=-1) > 10)
            spec = np.mean(s_sers_map[index_use], axis=0) * 2 
            spectra_attention.append(spec)
        return spectra_attention
    
    def show_maps_single_concentration(self, tds_dir=None, save=False):
        ncol = 5
        nrow = 6
        color_group = ['g','b', 'orange', 'm']
        title_use = ["Top %s" % (s_perc * 100) + "%" for s_perc in self.perc_use[:-1]] + ["ViT"]
        fig = vu.give_figure_specify_size(1.0, 1.3)
        spec5 = fig.add_gridspec(ncols=ncol, nrows=nrow, 
                                 width_ratios=[1 for _ in range(len(self.perc_use) + 1)] + [3.5], 
                                 wspace=0.0)
        if self.dataset == "DNP":
            index_conc = [2, 3, 4, 3, 3, 1]
        elif self.dataset == "TOMAS":
            index_conc = [3, 0, 2, 4, 1, 4]
        elif self.dataset == "PA":
            index_conc = [3, 4, 0, 0, 2]
        index = []
        for i, s_conc in enumerate(np.unique(self.concentration)):
            index.append(np.where(self.concentration == s_conc)[0][index_conc[i]])
        for i, s in enumerate(index):
            ax = fig.add_subplot(spec5[i, 0])
            ax.imshow(self.map_peak_locations[s])
            ax.set_ylabel(self.x_label_conc[i])
            ax.set_yticks([])
            if i != len(index) - 1:
                ax.set_xticks([])
            if i == 0:
                ax.set_title("Map at peaks")
            attn_map = self.attention[s]
            attn_map = (attn_map - np.min(attn_map))
            attn_map = attn_map / np.max(attn_map)
            for j in range(len(self.perc_use)):
                ax = fig.add_subplot(spec5[i, j+1])
                if j != len(self.perc_use) - 1:
                    ax.imshow(np.reshape(self.fake_map[j, s], [self.imh, self.imw]))
                    if i == 0:
                        ax.set_title(title_use[j], color=color_group[j])
                else:
                    ax.imshow(attn_map)
                    if i == 0:
                        ax.set_title(title_use[j], color=color_group[j+1])
                if i != len(index) - 1:
                    ax.set_xticks([])
                    ax.set_yticks([])
                else:
                    ax.set_yticks([])
                    # ax.xaxis.set_major_formatter(plt.NullFormatter())
                if i == len(index) - 1 and j == 1:
                    ax.set_xlabel("SERS maps and importance maps", loc='right')
            ax = fig.add_subplot(spec5[i, len(self.perc_use) + 1])    
            max_value = 0.0
            for j, s_perc in enumerate(self.perc_use):
                ax.plot(self.wavenumber, self.spectra_g[j][s], 
                        color=color_group[j], label="%s" % (s_perc * 100))
                max_value = np.maximum(max_value, np.max(self.spectra_g[j][s]))
            ax.plot(self.wavenumber, self.spectra_attention[s], 'r', label="VIT", color=color_group[j+1])
            for j, s_peak in enumerate(self.peak_loc):
                ax.plot([s_peak, s_peak], [np.min(self.spectra_attention[s]), max_value + max_value / 10], ls=':',color="gray")
            if i == 0:
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.4), frameon=False,
                  ncol=5, fancybox=False, shadow=False, labelspacing=0.3, handlelength=1.5, handletextpad=0.5)
            if i != len(index) - 1:
                ax.xaxis.set_major_formatter(plt.NullFormatter())
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(plt.NullFormatter())
            ax.yaxis.set_minor_formatter(plt.NullFormatter())
            ax.set_xlabel("wavenumber (cm" + r"$^{-1}$" + ")")
        if save:
            plt.savefig(tds_dir + "/attention_maps_%s.pdf" % self.dataset, pad_inches=0, bbox_inches='tight')
    

def get_spectra_performance_quantification_heatmap(model=["xception"], perf_crit="global_accu", 
                                                   real_val=False,
                                                   old_or_new="new",
                                                   show=True,
                                                   dataset="TOMAS",
                                                   tds_dir=None, save=False):
    version_quantification = {}
    if dataset == "TOMAS":
        if old_or_new == "new":
            version_quantification["xception"] = [20,21,22,23,24] #[10, 11, 12, 13, 14]
            version_quantification["unified_cnn"] = [20, 21, 22, 23, 24] #[0, 1, 2, 4, 5]
            version_quantification["resnet"] = [20, 21, 22, 24] #[0, 1, 2, 3, 4]
        elif old_or_new == "old":
            version_quantification["xception"] = [10, 11, 12, 13, 14]
            version_quantification["unified_cnn"] = [0, 1, 2, 4, 5]
            version_quantification["resnet"] = [0, 1, 2, 3, 4]
    else:
        if dataset != "PA":
            version_quantification["xception"] = [0, 1, 2, 3, 4]
            version_quantification["unified_cnn"] = [0, 1, 2, 3, 4]
            version_quantification["resnet"] = [0, 1, 2, 3, 4]
        else:
            version_quantification["xception"] = [0, 1, 2, 3, 4]
            version_quantification["unified_cnn"] = [5, 6, 7, 8, 9]
            version_quantification["resnet"] = [5, 6, 9, 10, 13]
    concentration_float = 1e-6 if dataset != "PA" else 1e-5 
    if dataset == "TOMAS":         
        percentage = [0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
    else:
        percentage = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    selection_method = ["Peak", "Std", "Mean", "Diff"]
    # tr_criteria = ["top_peak", "top_std", "top_mean", "top_diff"]
    tr_criteria = ["top_peak"]
    nrow, ncol = len(model), len(tr_criteria)
    perf_val_stat, perf_tt_stat = [], []
    tt_pred, best_key_group, tt_pred_group = [], [], []
    for i, s_model in enumerate(model):
        perf_val_stat_perf_model, perf_tt_stat_perf_model = [], []
        tt_pred_per_model, tt_accu_per_model, best_key_per_model, tt_pred_all_crit = [], [], [], []
        val_accu_per_model = []
        obj_detect = GetSpectraPerformanceTOMAS(s_model, False, True, version_quantification[s_model], 
                                                True, perf_crit, real_val=real_val, dataset=dataset)
        for j, s_crit in enumerate(tr_criteria):
            val_collect, tt_collect, avg_val_perf, avg_tt_perf = obj_detect._find_best_index_baseon_aggregate_perf(s_crit, 
                                                                                                      perf_crit=perf_crit,
                                                                                                      show=False)
            perf_val_stat_perf_model.append(avg_val_perf[0])
            perf_tt_stat_perf_model.append(avg_tt_perf)
            tt_pred_per_model.append(np.array(tt_collect[2])[:, 0])
            tt_label = np.concatenate(obj_detect.label[1], axis=0)
            tt_conc = np.concatenate(obj_detect.concentration[1], axis=0)
            best_key_per_model.append(val_collect[1])
            tt_accu_per_model.append(tt_collect[1])
            val_accu_per_model.append(val_collect[2])
        if perf_crit == "rmae" or perf_crit == "mae" or perf_crit == "rae":
            key_index = np.argmin(val_accu_per_model)
        elif perf_crit == "rsquare" or perf_crit == "log_rsquare":
            key_index = np.argmax(val_accu_per_model)
        tt_pred.append(tt_pred_per_model[key_index])
        best_key_group.append(best_key_per_model[key_index])
        perf_val_stat.append(perf_val_stat_perf_model)
        perf_tt_stat.append(perf_tt_stat_perf_model)
        tt_pred_group.append(tt_pred_all_crit)  # [4, 30, 10, 10]
    print("The shape of the prediction group", np.shape(tt_pred_group))
    tt_conc_update = tt_conc.copy()
    tt_conc_update[tt_conc_update == 0] = concentration_float
    if show:
        color_use = ['r', 'g', 'b']
        fig = vu.give_figure_specify_size(0.5)
        ax = fig.add_subplot(111)
        # x_ticks = np.arange(len(np.unique(tt_conc)) + 1)[1:]
        x_ticks = np.unique(tt_conc)
        for i, s_model in enumerate(model):
            print(best_key_group[i])
            x_tick, quan_stat = utils.get_rsquare_curve_stat(tt_conc_update, np.array(tt_pred[i]))
            _rsquare = vu.get_performance_baseon_crit_quantification(tt_conc_update, np.array(tt_pred[i]), perf_crit)
            # _rsquare = utils.calc_rsquare(tt_conc_update, np.array(tt_pred[i]))
            ax.errorbar(x_ticks, quan_stat[:, 0], quan_stat[:, 1], capsize=2, marker='.', 
                        label="%s: %.2f" % (s_model, (_rsquare)))
            for j, s_conc in enumerate(np.unique(tt_conc_update)):
                index = np.where(tt_conc_update == s_conc)[0]
                for q in index:
                    ax.plot(x_ticks[j], tt_pred[i][q], ls='', marker='.')
        ax.plot(x_ticks, np.unique(tt_conc), color='gray')
        ax.legend(loc='best')
        ax.set_xscale("symlog", linthresh=np.unique(tt_conc)[1])
        ax.set_yscale("symlog", linthresh=np.unique(tt_conc)[1])
        ax.set_ylim(-1e-3, 1.1)
        ax.set_xticks(x_ticks)    
        ax.set_xticklabels(["%.3f" % v for v in np.unique(tt_conc)], rotation=90)
        ax.set_xlabel("Concentration (uM)")
        ax.set_ylabel("Predicted concentration (uM)")
        
        perf_title = ["validation_heatmap", "testing_heatmap"]
        for q, perf_stat in enumerate([perf_val_stat, perf_tt_stat]):
            if perf_crit == "validation_loss" and q == 0:
                perf_stat = np.log(perf_stat)
            if perf_crit == "rsquare" or perf_crit == "log_rsquare":
                vmin, vmax = 0, 1
            else:
                vmin, vmax = np.min(np.reshape(perf_stat, [-1])), np.max(np.reshape(perf_stat, [-1]))
            fig = vu.give_figure_specify_size(1, 2.6 / 6 * len(model))
            ax_global = vu.ax_global_get(fig)
            ax_global.set_xticks([])
            ax_global.set_yticks([])
            ax_group = []
            for i, s_model in enumerate(model):
                axt = fig.add_subplot(nrow, 1, i+1, frameon=False)
                axt.set_xticks([])
                axt.set_yticks([])
                axt.set_title(s_model)
                for j, s_crit in enumerate(tr_criteria):        
                    ax = fig.add_subplot(nrow, ncol, i * ncol + j + 1)
                    if j == 0:
                        yticklabel = ["%.1f" % (v * 100) for v in percentage]
                    else:
                        yticklabel = ["" for v in percentage]
                    if i == nrow - 1:
                        xticklabel = ["%.1f" % (v * 100) for v in percentage]
                    else:
                        xticklabel = ["" for v in percentage]
                    im = sns.heatmap(perf_stat[i][j],
                        vmin=vmin,
                        vmax=vmax,
                        ax=ax,
                        yticklabels=yticklabel,
                        xticklabels=xticklabel,
                        cbar=False,
                        cmap="Blues",
                    )
                    if i == 0:
                        ax.set_title(selection_method[j] + "\n")
                    ax_group.append(ax)
            cax = plt.axes([0.92, 0.13, 0.025, 0.75])
            mappable = im.get_children()[0]
            plt.colorbar(mappable, cax=cax, ax=ax_group)
            ax_global.set_xlabel("\n\n\nPercentage of spectra used at testing (%)")
            ax_global.set_ylabel("Percentage of spectra used at training (%)\n\n\n")
            plt.subplots_adjust(wspace=0.05, hspace=0.2)
            if save:
                plt.savefig(tds_dir + "/quantification_heatmap_%s_tomas.pdf" % (perf_title[q]), pad_inches=0, bbox_inches='tight')
    return tt_pred, [best_key_group, perf_val_stat, tr_criteria, percentage]


class GetSpectraPerformanceTOMAS(object):
    def __init__(
        self,
        model,
        detection,
        quantification,
        version_group,
        avg_spectra=True,
        perf_crit="rsquare",
        normalization="none",
        dataset="TOMAS",
        loc="home",
        real_val=False,
        without_zero=False,
        
    ):
        super(GetSpectraPerformanceTOMAS).__init__()
        self.dataset = dataset
        self.num_leave_index = 30 if self.dataset != "PA" else 25 
        if without_zero == True and quantification == True:
            self.num_leave_index = self.num_leave_index - 5
        self.model = model
        self.detection = detection
        self.quantification = quantification
        self.normalization = normalization #["max" if self.detection else "none"][0]
        self.version_group = version_group
        self.avg_spectra = avg_spectra
        self.perf_crit = perf_crit
        self.real_val = real_val
        self.loc = loc
        self.concentration_float = 1e-6 if self.dataset != "PA" else 1e-5 
        # if model == "xception":
        # label, concentration = pickle.load(
        #     open("../rs_dataset/Tomas_visualization/label_conc_detection_%s_quantification_%s.obj" % (detection, 
        #                                                                                              quantification), "rb"))
        # elif model == "unified_cnn":
        if self.dataset == "TOMAS":
            label, concentration = pickle.load(
                open("../rs_dataset/Tomas_visualization/label_conc_detection_%s_quantification_%s_unified_cnn.obj" % (detection,           
                                                                                                                    quantification), "rb"))
        elif self.dataset == "DNP" or self.dataset == "PA":
            label, concentration = pickle.load(open("../rs_dataset/%s_label_conc_detection_%s_quantification_%s.obj" % (self.dataset, 
                                                                                                                        self.detection, 
                                                                                                                        self.quantification), "rb"))        
        if without_zero == True and quantification == True:
            if self.dataset != "TOMAS":
                label = [v[5:] for v in label]
                label[0] = np.array(label[0])[:, 2:]
                concentration = [v[5:] for v in concentration]
                concentration[0] = np.array(concentration[0])[:, 2:]
            else:
                non_zero_index = np.where(label[0][0] != 0)[0]
                label = [v[5:] for v in label]
                label[0] = np.array(label[0])[:, non_zero_index]
                concentration = [v[5:] for v in concentration]
                concentration[0] = np.array(concentration[0])[:, non_zero_index]
        true_index = np.zeros([6]).astype(np.int8)
        if self.quantification:
            for i in range(6):
                true_index[i] = 1 + i * 11
            true_index -= 1
        else:
            for i in range(5):
                true_index[i+1] = i * 11 + np.where(label[0][0] == 1)[0][0]         
        self.true_index = true_index
        self.label = label 
        self.concentration = concentration 
        if self.dataset == "TOMAS":
            self.percentage = [0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
        else:
            self.percentage = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
        self.percentage = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
        self.tt_method = ["top_peak", "top_std", "top_mean", "top_diff"]
        
    def organize_concentration_label(self, leave_index):
        val_label, tt_label = self.label[0][leave_index], self.label[1][leave_index]
        val_conc, tt_conc = self.concentration[0][leave_index], self.concentration[1][leave_index]
        label_group = np.concatenate([val_label, tt_label], axis=0)
        conc_group = np.concatenate([val_conc, tt_conc], axis=0)
        conc_update = conc_group.copy()
        conc_update[conc_update == 0] = self.concentration_float
        val_tt_group = np.concatenate([np.ones([len(val_label)]), np.zeros([len(tt_label)])], axis=0)
        return label_group, conc_group, conc_update, val_tt_group, \
                [val_label, conc_update[val_tt_group == 1]], \
                [tt_label, conc_update[val_tt_group == 0]]

    def _get_perf_stat_per_leave_index(self, tr_selection_criteria,
                                       tt_selection_method=["top_peak", "top_std", "top_mean", "top_diff"],
                                       perf_crit="f1_score", 
                                       s_leave_index=0, show=False):
        label_group, conc_group, conc_update, val_tt_group, [val_label, val_conc], \
            [tt_label, tt_conc] = self.organize_concentration_label(s_leave_index)
        val_index = np.where(val_tt_group == 1)[0]
        tt_index = np.where(val_tt_group == 0)[0]  
        if self.real_val:
            val_conc_update = conc_update[self.true_index]
            val_label = val_label[self.true_index]
            val_conc = val_conc[self.true_index]
            # label_group = np.concatenate([val_label, tt_label], axis=0)
            # conc_group = np.concatenate([val_conc, tt_conc], axis=0)
        else:
            val_conc_update = conc_update[val_index]
        # print("The shape of the label and conc", np.shape(val_label), np.shape(val_conc))
        tt_conc_update = conc_update[tt_index]
        prediction_group_val, prediction_group_tt = [], []
        perf_crit_temp = ["global_accu" if self.detection else "rsquare"][0]
        for s_version in self.version_group:
            _performance_table, raw_prediction, stat = vu.get_accuracy_heatmap(
                self.dataset,
                self.model,
                tr_selection_criteria,
                label_group,
                conc_group,
                self.detection,
                self.quantification,
                self.avg_spectra,
                tt_selection_method=tt_selection_method,
                leave_index=s_leave_index,
                normalization=self.normalization,
                perf_crit=perf_crit_temp,
                version=s_version,
                loc=self.loc)
            raw_prediction = np.maximum(raw_prediction, 0)
            if self.real_val:
                prediction_group_val.append(np.array(raw_prediction)[:, :, :, self.true_index])
            else:
                prediction_group_val.append(np.array(raw_prediction)[:, :, :, val_index])
            prediction_group_tt.append(np.array(raw_prediction)[:, :, :, tt_index])
        ensemble_perf_val, ensemble_perf_val_conc, \
            ensemble_prediction_val = get_performance_spectra_experiment(prediction_group_val,
                                                                        stat,
                                                                        tt_selection_method,
                                                                        perf_crit,
                                                                        self.percentage, 
                                                                        detection=self.detection, 
                                                                        quantification=self.quantification,
                                                                        label_use=val_label,
                                                                        conc_use=val_conc_update,
                                                                        show=False)
        
        ensemble_perf_tt, ensemble_perf_tt_conc, \
            ensemble_prediction_tt = get_performance_spectra_experiment(prediction_group_tt,
                                                                        stat,
                                                                        tt_selection_method,
                                                                        perf_crit,
                                                                        self.percentage, 
                                                                        detection=self.detection, 
                                                                        quantification=self.quantification,
                                                                        label_use=tt_label,
                                                                        conc_use=tt_conc_update,
                                                                        show=False)
        return [ensemble_perf_val, ensemble_perf_tt],\
                [ensemble_prediction_val, ensemble_prediction_tt], \
                stat[:-2], [val_label, tt_label], [val_conc, tt_conc]    

    def _get_best_perf_index(
        self,
        performance_matrix,
        pred_matrix,
        label_use,
        best_index,
        perf_crit,
        tt_select_method,
        show=False,
    ):
        key_use = np.zeros_like(performance_matrix).astype(str)
        for i, s_method in enumerate(tt_select_method):
            for j, s_per in enumerate(self.percentage):
                for q, s_tr_per in enumerate(self.percentage):
                    key_use[i, j, q] = "_".join(
                        ["%s" % v for v in [s_method, s_per, s_tr_per]]
                    )
        key_reshape = np.reshape(key_use, [-1])
        performance_reshape = np.reshape(performance_matrix[0], [-1])
        if self.detection:
            pred_reshape = np.reshape(pred_matrix, [len(pred_matrix), len(self.percentage) ** 2, -1, 2])
        elif self.quantification:
            pred_reshape = np.reshape(pred_matrix, [len(pred_matrix), len(self.percentage) ** 2, -1])
        if len(best_index) == 0:
            if perf_crit == "global_accu" or perf_crit == "f1_score" or perf_crit == "rsquare" or perf_crit == "false_positive" or perf_crit == "log_rsquare":
                best_index = np.argmax(performance_reshape)
            elif perf_crit == "rmae" or perf_crit == "mae" or perf_crit == "validation_loss" or perf_crit == "rae":
                best_index = np.argmin(performance_reshape)
        else:
            best_index = best_index[0]
        return (
            best_index,
            key_reshape[best_index],
            performance_reshape[best_index],
            pred_reshape[:, best_index],
        )
        
    def _get_performance_baseon_validation_perf(self, tr_selection_criteria="top_peak",
                                                leave_index=0, perf_crit="global_accuracy",
                                                show=False):
        [ensemble_perf_val, ensemble_perf_tt],\
            [ensemble_prediction_val, ensemble_prediction_tt], \
            stat, \
            [val_label, tt_label], \
            [val_conc, tt_conc] = self._get_perf_stat_per_leave_index(tr_selection_criteria, [tr_selection_criteria], 
                                                       perf_crit=perf_crit, s_leave_index=leave_index, show=show)
        _best_val_index, _best_val_key, \
            _best_val_perf, _best_val_pred = self._get_best_perf_index(ensemble_perf_val,
                                                                       ensemble_prediction_val,
                                                                       val_label,
                                                                       [], perf_crit, [tr_selection_criteria], show=show)
        _, _best_tt_key, \
            _best_tt_perf, _best_tt_pred = self._get_best_perf_index(ensemble_perf_tt, ensemble_prediction_tt,
                                                                     tt_label, [_best_val_index],
                                                                     perf_crit, [tr_selection_criteria], show=show)
        val_collect = [_best_val_index, _best_val_key, _best_val_perf, _best_val_pred]
        tt_collect = [_best_tt_key, _best_tt_perf, _best_tt_pred]
        return val_collect, tt_collect, [ensemble_perf_val, ensemble_perf_tt] , [ensemble_prediction_val, ensemble_prediction_tt], \
            [val_label, tt_label], [val_conc, tt_conc]
            
            
    def _aggregate_over_multiple_leave_index(self, tr_selection_criteria="top_peak", perf_crit="global_accu", show=False):
        ensemble_perf_multi = []
        ensemble_pred_multi = [[], []]
        conc_tt_update = np.concatenate(self.concentration[1], axis=0)
        conc_tt_update[conc_tt_update == 0] = self.concentration_float 
        tt_label = np.concatenate(self.label[1], axis=0)
        for s_leave_index in np.arange(self.num_leave_index).astype(np.int32):
            _ensemble_perf, _ensemble_pred, stat, _label, _conc = self._get_perf_stat_per_leave_index(tr_selection_criteria, 
                                                                                                      [tr_selection_criteria], 
                                                                                                      perf_crit=perf_crit, 
                                                                                                      s_leave_index=s_leave_index, 
                                                                                                      show=False)
            ensemble_perf_multi.append(_ensemble_perf[0])
            for i, v in enumerate(ensemble_pred_multi):
                if self.quantification:
                    v.append(np.maximum(_ensemble_pred[i], self.concentration_float))
                else:
                    v.append(_ensemble_pred[i])
        avg_val_ensemble_perf = np.mean(ensemble_perf_multi, axis=0) # [1, 10, 10]
        if self.quantification:
            tt_pred_tot = np.reshape(ensemble_pred_multi[1], [self.num_leave_index, len(self.percentage), len(self.percentage)])
        if self.detection:
            tt_pred_tot = np.reshape(ensemble_pred_multi[1], [self.num_leave_index, len(self.percentage), len(self.percentage), 2])
        ensemble_perf_tt = np.zeros([len(self.percentage), len(self.percentage)])
        for i in range(len(self.percentage)):
            for j in range(len(self.percentage)):
                _pred = tt_pred_tot[:, i, j]
                if self.quantification:
                    if perf_crit == "rsquare":
                        _perf = utils.calc_rsquare(conc_tt_update, _pred)
                    elif perf_crit == "log_rsquare":
                        _perf = utils.calc_rsquare_log(conc_tt_update, _pred)
                if self.detection:
                    _perf = vu.get_performance_baseon_crit_detection(tt_label, _pred, crit="global_accu")
                ensemble_perf_tt[i, j] = _perf
        if show:
            fig = plt.figure(figsize=(10, 5))
            ax = fig.add_subplot(121)
            if perf_crit == "rsquare" or perf_crit == "log_rsquare":
                sns.heatmap(np.maximum(avg_val_ensemble_perf[0], 0), annot=True, ax=ax, xticklabels=["%.3f" % v for v in self.percentage], 
                            yticklabels=["%.3f" % v for v in self.percentage], cmap="Blues", 
                            annot_kws={'fontsize': 8, 'fontfamily': 'serif'})
            else:
                sns.heatmap(avg_val_ensemble_perf[0], annot=True, ax=ax, xticklabels=["%.3f" % v for v in self.percentage], 
                            yticklabels=["%.3f" % v for v in self.percentage], cmap="Blues")
            ax = fig.add_subplot(122)
            sns.heatmap(np.maximum(ensemble_perf_tt, 0), annot=True, ax=ax, xticklabels=["%.3f" % v for v in self.percentage], 
                            yticklabels=["%.3f" % v for v in self.percentage], cmap="Blues", 
                            annot_kws={'fontsize': 8, 'fontfamily': 'serif'})
            plt.subplots_adjust(wspace=0.05)
        return [ensemble_perf_multi, ensemble_perf_tt], ensemble_pred_multi, avg_val_ensemble_perf
    
    def _find_best_index_baseon_aggregate_perf(self, tr_selection_criteria="top_peak", perf_crit="global_accu", show=False):
        ensemble_perf_multi, ensemble_pred_multi, avg_val_ensemble_perf = self._aggregate_over_multiple_leave_index(tr_selection_criteria, 
                                                                                                                    perf_crit, show=show)
        _best_val_index, _best_val_key, \
            _best_val_perf, _best_val_pred = self._get_best_perf_index(avg_val_ensemble_perf, ensemble_pred_multi[0], self.label[0], [], 
                                                                       perf_crit, [tr_selection_criteria], show=False)
        _best_tt_index, _best_tt_key, \
            _best_tt_perf, _best_tt_pred = self._get_best_perf_index(np.expand_dims(ensemble_perf_multi[1], axis=0), 
                                                                     ensemble_pred_multi[1], self.label[1], 
                                                                     [_best_val_index], perf_crit, [tr_selection_criteria])
        val_collect = [_best_val_index, _best_val_key, _best_val_perf, _best_val_pred]
        tt_collect = [_best_tt_key, _best_tt_perf, _best_tt_pred]
        
        return val_collect, tt_collect, avg_val_ensemble_perf, ensemble_perf_multi[1]
    
    
        
def get_performance_spectra_experiment(
        prediction_group,
        stat,
        tt_selection_method,
        perf_crit,
        percentage, 
        detection=True, quantification=False,
        label_use=[],
        conc_use=[],
        show=False):
    num_percentage = len(percentage)
    ensemble_prediction = np.mean(np.array(prediction_group), axis=0)  # [len(tt_selection_method), 10, 10, ]
    ensemble_perf = np.zeros([len(tt_selection_method), num_percentage, num_percentage])
    ensemble_perf_per_concentration = np.zeros([len(tt_selection_method), num_percentage,
                                                num_percentage, len(np.unique(conc_use))])
    label_use_tensor = torch.from_numpy(label_use).long()
    ensemble_prediction_tensor = torch.from_numpy(ensemble_prediction).to(torch.float32)
    
    for i in range(len(tt_selection_method)):
        for j in range(num_percentage):
            for m in range(num_percentage):
                if perf_crit == "validation_loss":
                    pred_t = ensemble_prediction_tensor[i, j, m]
                else:
                    pred_t = ensemble_prediction[i, j, m]
                pred = ensemble_prediction[i, j, m]
                if detection:
                    accu = vu.get_performance_baseon_crit_detection(
                        label_use, pred_t, perf_crit, label_use_tensor, 
                    )
                    # accu_per_conc, _, _ = vu.show_accuracy_over_concentration(
                    #     pred,
                    #     np.argmax(pred, axis=-1),
                    #     label_use,
                    #     conc_use,
                    #     show=False,
                    #     save=False,
                    #     tds_dir=None,
                    # )
                    # ensemble_perf_per_concentration[i, j, m] = np.array(accu_per_conc) * 100
                if quantification:
                    accu = vu.get_performance_baseon_crit_quantification(
                        conc_use, pred, perf_crit
                    )
                ensemble_perf[i, j, m] = accu
    if show:
        vu.show_accuracy_spectra_heatmap(
            ensemble_perf, stat[:-2], [], True, tt_selection_method
        )
    return ensemble_perf, ensemble_perf_per_concentration, ensemble_prediction
            
                
class GetTomasPerformanceVIT(object):
    def __init__(self, detection, quantification, lr=0.08, patch_size=2, version_use=[], 
                 normalization="max", concentration_float=1e-6, quantification_loss="mae",
                 target_shape=[56, 56], leave_method="leave_one_chip_per_conc", dataset="TOMAS", 
                 model_init="none", lr_schedule="cosine", loc="nobackup"):
        super(GetTomasPerformanceVIT).__init__()
        exp_dir = "../exp_data/VIT/%s/detection_%s_quantification_%s" % (dataset, detection, quantification)
        if quantification == True:
            exp_dir += "_nozero"
        exp_dir += "/"
        file2load = [v for v in os.listdir(exp_dir) if "stat_patch_%d_lr_%.4f" % (patch_size, lr) in v \
            and "normalization_%s_%s_target_h_%d" % (normalization, quantification_loss, 
                                                        target_shape[0]) in v and '.obj' in v and "_v" not in v]
        # print(file2load)        
        # if model_init != "none":
        #     file2load = [v for v in file2load if model_init in v and lr_schedule in v]  
        # else:
        #     file2load = [v for v in file2load if "initialisation" not in v]   
        print(file2load)
        file2load = exp_dir + file2load[0]
        # print(file2load)
        stat_g = pickle.load(open(file2load, "rb"))
        self.key_use = [v for v in list(stat_g[0].keys()) if "version" in v]
        if len(version_use) > 0:
            self.version_use = ["version_%d" % v for v in version_use]
        else:
            self.version_use = self.key_use
        self.act_key = [v for v in self.key_use if v in self.version_use]
        # print(self.version_use)
        self.color = ['r', 'g', 'b', 'c', 'm', 'orange']
        self.dataset = dataset
        self.normalization = normalization
        self.detection=detection 
        self.quantification=quantification 
        self.concentration_float=concentration_float
        ensemble_pred, all_pred, ensemble_label, ensemble_conc = self.get_ensemble(stat_g)
        self.ensemble_pred = ensemble_pred 
        self.ensemble_label = ensemble_label 
        self.ensemble_conc = ensemble_conc
        self.all_pred = all_pred
        if self.dataset == "DNP":
            x_label_conc = ["Blank", "1nM", "10nM", "100nM", "1"+r'$\mu$'+"M", "10"+r'$\mu$'+"M"]
        elif self.dataset == "TOMAS":
            x_label_conc = ["Blank", "0.1nM", "1nM", "10nM", "100nM", "1"+r'$\mu$'+"M"]
        elif self.dataset == "PA":
            x_label_conc = ["Blank", "10nM", "100nM", "1"+r'$\mu$'+"M", "10"+r'$\mu$'+"M"]
        
        if self.quantification and loc == "scratch":
            self.x_label_conc = x_label_conc[1:]
        else:
            self.x_label_conc = x_label_conc
        
        self.quan_perf = []

    def get_ensemble(self, stat_g):
        ensemble_pred, all_pred = [], []
        ensemble_label, ensemble_conc = [], []
        keys_act = []
        key_use = list(stat_g[0].keys())
        # print("ensemble function:", key_use, "s_stat keys", stat_g[0].keys())
        for i, s_stat in enumerate(stat_g):
            s_stat_use = [s_stat[k] for k in key_use if k in self.version_use]
            all_pred.append(s_stat_use)
            _avg_perf = np.mean(s_stat_use, axis=0)                    
            ensemble_pred.append(_avg_perf)
            ensemble_label.append(s_stat["label"])
            ensemble_conc.append(s_stat["concentration"])
        return ensemble_pred, np.array(all_pred), np.array(ensemble_label), np.array(ensemble_conc)
    
    def _get_detection_performance(self):
        pred_g = np.transpose(self.all_pred[:, :, :, 0], (1, 0, 2, 3))  # [num_exp, num_points, 1]
        tt_label = np.reshape(self.ensemble_label, [-1])
        for i, s_pred in enumerate(pred_g):
            s_pred = np.reshape(s_pred, [-1, len(np.unique(tt_label))])
            pred_label = np.argmax(s_pred, axis=-1)
            accu_per_conc, std_per_conc, xtick_label = vu.show_accuracy_over_concentration(s_pred, pred_label, tt_label,
                                                                                           np.reshape(self.ensemble_conc, [-1]), show=False)
            # print(self.act_key[i], ["%.3f" % v for v in accu_per_conc], np.sum((pred_label == tt_label)) / len(tt_label))
            # print(self.act_key[i], s_pred[:5], np.argmax(s_pred[:5], axis=-1))
    
    def get_detection_performance(self, show=True, tds_dir=None, save=False):
        np.random.seed(1002)
        self._get_detection_performance()
        tt_label = np.reshape(self.ensemble_label, [-1])
        pred = np.reshape(np.array(self.ensemble_pred)[:, :, 0], [-1, len(np.unique(tt_label))])
        pred_prob = softmax(pred, axis=-1)
        pred_label = np.argmax(pred, axis=-1)
        conc = np.reshape(self.ensemble_conc, [-1])
        conc_update = conc.copy()
        conc_update[conc_update == 0.0] = self.concentration_float
        perf = np.sum(pred_label == tt_label) / len(tt_label)
        pred_accu = (pred_label == tt_label).astype(np.int32)
        accu_per_conc, std_per_conc, xtick_label = vu.show_accuracy_over_concentration(pred, pred_label, 
                                                                                       tt_label, conc, 
                                                                                        show=False)
        if show:
            min_value, max_value = np.min(conc_update), np.max(conc_update)
            x_axis = np.unique(conc) #np.unique(conc_update) #np.arange(len(xtick_label) + 1)[1:]
            fig = plt.figure(figsize=(5, 3))
            ax = fig.add_subplot(111)
            correct_or_wrong = (pred_label == tt_label).astype(np.int32)
            print("The shape of the conc", np.shape(conc), np.unique(conc), np.shape(correct_or_wrong))
            for j, s_conc in enumerate(np.unique(conc)):
                index = np.where(conc == s_conc)[0]
                cor_index = index[np.where(correct_or_wrong[index] == 1)[0]]
                wor_index = index[np.where(correct_or_wrong[index] == 0)[0]]
                rand_add = np.zeros([len(index)]) + np.random.normal(s_conc/10, 0.00001, size=[len(index)]) * np.random.randint(-1, 2, [len(index)])
                if np.min(rand_add) + x_axis[j] < min_value:
                    min_value = x_axis[j] + np.min(rand_add)
                if np.max(rand_add) + x_axis[j] > max_value:
                    max_value = x_axis[j] + np.max(rand_add)
                # else:
                #     rand_add = np.zeros([len(index)])
                if len(cor_index) > 0:
                    for m, q in enumerate(cor_index):                
                        ax.plot(x_axis[j] + rand_add[m], np.max(pred_prob[q]), 
                                marker='+', ls='', color='g', markersize=5)
                if len(wor_index) > 0:
                    for m, q in enumerate(wor_index):                
                        ax.plot(x_axis[j] + rand_add[-m-1], 1.0 - np.max(pred_prob[q]), 
                                marker='x', ls='', color='r', markersize=5)
            print(min_value)
            ax.plot([min_value, max_value], [0.5, 0.5], ls=':', color='gray')
            ax.text(np.unique(conc)[-2], 0.8, "correct", color="g")
            ax.text(np.unique(conc)[-2], 0.3, "wrong", color="r")
            ax.set_ylim((0, 1.05))
            ax.set_xlabel("Concentration")
            ax.set_ylabel("Predicted probability")
            ax.set_xscale("symlog", linthresh=np.sort(np.unique(conc))[1])
            ax.set_xticks(np.unique(x_axis))
            ax.set_xticklabels(self.x_label_conc)
            if save:
                plt.savefig(tds_dir + "/detection_leave_one_out_%s_normalization_%s.pdf" % (self.dataset, self.normalization),
                            pad_inches=0, bbox_inches='tight')
        return pred, pred_label, tt_label, conc, [perf, f1_score(tt_label, pred_label)]
                
    
    def _get_quantification_performance_per_version(self, show=True):
        pred_g = np.squeeze(np.transpose(self.all_pred, (1, 0, 2)), axis=-1)  # [num_exp, num_points, 1]
        
        for i in range(len(pred_g)):
            conc = np.reshape(self.ensemble_conc, [-1])
            conc_update = conc.copy()
            conc_update[conc_update == 0] = self.concentration_float
            print(self.act_key[i], utils.calc_rsquare_log(conc_update[5:], pred_g[i][5:]))
            self.quan_perf.append([self.act_key[i], utils.calc_rsquare_log(conc_update[5:], pred_g[i][5:])])
            _ = get_quantification_performance_plot(pred_g[i], conc, conc_update, self.color, name_use=self.act_key[i], show=show)

    
    def get_quantification_performance(self, show=True, tds_dir=None, save=False):
        conc = np.reshape(self.ensemble_conc, [-1])
        pred = np.reshape(self.ensemble_pred, [-1])
        conc_update = conc.copy()
        conc_update[conc_update == 0] = self.concentration_float
        # print("The shape of the concentration prediction", np.shape(conc_update))
        # print("The shape of the gt conc", np.shape(conc_update))
        perf_group = get_quantification_performance_plot(pred, conc, conc_update, self.color, show=show, xlabel=self.x_label_conc)
        print("Rsquare subset:", utils.calc_rsquare_log(conc_update[conc_update != self.concentration_float], pred[conc_update != self.concentration_float]))
        print(perf_group)
        if save:
            plt.savefig(tds_dir + "/quantification_leave_one_out_%s.pdf" % self.dataset, pad_inches=0, bbox_inches='tight')
        return pred, conc, conc_update, perf_group
        
    
def get_quantification_performance_plot(prediction, concentration, concentration_update, color, name_use="Quantification", show=True, xlabel=[None]):
    [rsquare, rsquare_subset], [rae, rae_subset], \
        [rmae, rmae_subset], [mae, mse] = utils.get_quantification_performance(concentration_update, prediction, print_info=False)
    # print("==============%s performance=================" % name_use)
    # print("Rsquare: %.4f" % (rsquare))
    # print("Rsquare(log): %.4f" % rsquare_subset)
    # print("rmae: %.4f, rae: %.4f" % (rmae, rae))
    
    x_tick, quan_stat = utils.get_rsquare_curve_stat(concentration,prediction)
    
    if show:
        fig = plt.figure(figsize=(5, 5))
        # fig = vu.give_figure_specify_size(0.5)
        ax = fig.add_subplot(111)
        x_axis = np.log(np.unique(concentration))
        ax.errorbar(x_axis, quan_stat[:, 0], quan_stat[:, 1], color='m', lw=1, capsize=3)
        # ax.plot(x_axis, quan_stat[:, 0], color='m', lw=1)
        ax.plot(x_axis, x_axis, color='gray')
        pred_log = np.log(prediction)
        for i, s_conc in enumerate(x_tick):
            index = np.where(concentration == s_conc)[0]
            for j in range(len(index)):
                ax.plot(x_axis[i], pred_log[index[j]], color=color[j], marker='x', ls='', markersize=3)
                ax.text(x_axis[i]*1.01, pred_log[index[j]], "%d" % j, color=color[j])
        # ax.set_xscale("symlog", linthresh=np.sort(x_axis)[1])
        # ax.set_yscale("symlog", linthresh=np.sort(x_axis)[1])
        if np.min(concentration) <= 1e-5:
            ax.set_title("R^2:%.2f" % (utils.calc_rsquare_log(concentration_update, prediction)) + " R^2 (conc !=0 ):%.2f" % (utils.calc_rsquare_log(concentration[concentration != 0], 
                                                                                                                                             prediction[concentration != 0])),
                         fontsize=10)
        else:
            ax.set_title("R^2:%.2f" % (utils.calc_rsquare_log(concentration, prediction)), 
                         fontsize=10)
        # ax.grid(ls=':')
        ax.set_xticks(x_axis)
        if len(xlabel) > 1:
            ax.set_xticklabels(xlabel)
        ax.set_xlabel("True concentration (uM)")
        ax.set_ylabel("Predicted concentration (uM)")
    return rsquare, rsquare_subset, rmae, mae

        
def get_intensity_plot_over_different_percentage(map_group, label_group, conc_group, chip_group, wavenumber, selection_crit, 
                                                 save_name=None,
                                                 tds_dir=None, save=False):
    """Args:
    map_group: [num_measurements, imh, imw, num_wave]
    label_group: [num_measurements]
    conc_group: [num_measurements]
    chip_group: [num_measurements]
    """
    map_region = [[1071, 1083], [1324, 1336], [1566, 1575]]
    _, imh, imw, num_wave = np.shape(map_group)
    map_reshape = np.reshape(map_group, [-1, imh * imw, num_wave])
    if selection_crit == "top_peak":
        map_select = read_tomas.get_map_at_specified_regions(map_group, map_region, wavenumber)
        crit = np.argsort(np.reshape(map_select, [-1, imh * imw]), axis=-1)
    elif selection_crit == "top_std":
        crit = np.std(map_reshape, axis=-1)
    elif selection_crit == "top_mean":
        crit = np.mean(map_reshape, axis=-1)
    unique_conc = np.unique(conc_group)
    unique_conc[unique_conc == 0] = 1e-6
    conc_group_update = conc_group.copy()
    conc_group_update[conc_group_update == 0] = 1e-6
    percentage = [0.002, 0.01, 0.1, 0.5, 1]
    color_use = ['r', 'g', 'b', 'c', 'orange']
    fig = plt.figure(figsize=(8, 10))
    ax_global = utils.ax_global_get(fig)
    ax_global.set_xticks([])
    ax_global.set_yticks([])
    for i, s_percentage in enumerate(percentage):
        ax = fig.add_subplot(len(percentage), 1, i + 1)
        select_spectra, _, _, _ = psd.select_baseon_percentage(map_reshape, label_group, conc_group, crit, s_percentage, avg=True)
        select_intensity = read_tomas.get_intensity_at_specified_regions(select_spectra, map_region, wavenumber)
        peak_stat = utils.calc_avg_baseon_concentration(conc_group, select_intensity)
        
        ax.errorbar(unique_conc, peak_stat[:, 0], peak_stat[:, 1], color='m', capsize=4)
        for j, s_chip in enumerate(np.unique(chip_group)):
            sub_index = np.where(chip_group == s_chip)[0]
            ax.plot(conc_group_update[sub_index], select_intensity[sub_index], color=color_use[j], marker='.', ls='', label="chip%d" % s_chip)
        ax.grid(ls=':')
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(unique_conc)
        ax.set_xticklabels(["%.4f" % v for v in np.unique(conc_group)])
        ax.set_title("Top: %d" % (s_percentage * imh * imw))
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.subplots_adjust(hspace=0.38)
    ax_global.set_xlabel("\n\nConcentration (uM)")
    ax_global.set_ylabel("Peak intensity \n\n\n")
    ax_global.set_title("Selection criterior (%s) " % selection_crit + "\n\n")
    if save:
        # print(selection_crit, save_name)
        plt.savefig(tds_dir + "/pure_peak_intensity_and_concentration_relation_selection_%s_%s.pdf" % (selection_crit, save_name), 
                    pad_inches=0, bbox_inches='tight')
        

def get_pure_relation_between_concentration_and_peak_intensity_tomas(target_shape=[56, 56],
                                                                     skip_value=1,
                                                                     selection_crit="top_std",
                                                                     normalization="none",
                                                                     save=False, tds_dir=None):
    if not os.path.exists(tds_dir):
        os.makedirs(tds_dir)
    map_region = [[1071, 1083], [1324, 1336], [1566, 1575]]
    tr_out, tt_out, wavenumber, tr_chip_index, tt_chip_index = read_tomas.prepare_tomas_data(target_shape,
                                                                                             skip_value=skip_value, 
                                                           padding_approach="zero", 
                                                           leave_index=0, 
                                                           path="../rs_dataset/Tomas_obj/", 
                                                           check_filename=True)
    print("The sers maps shape", np.shape(tr_out[0]), np.shape(tt_out[0]) )
    map_group = np.concatenate([tr_out[0], tt_out[0]], axis=0)
    label_group = np.concatenate([tr_out[1], tt_out[1]], axis=0)
    conc_group = np.concatenate([tr_out[2], tt_out[2]], axis=0)
    chip_group = np.concatenate([tr_chip_index, tt_chip_index], axis=0)
    _, imh, imw, num_wave = np.shape(map_group)
    if normalization == "max":
        map_group = map_group / np.max(map_group, axis=(-1, -2, -3), keepdims=True)
    elif normalization == "single_max":
        map_group = map_group / np.max(tr_out[0])
    get_intensity_plot_over_different_percentage(map_group, label_group, conc_group, chip_group, wavenumber, selection_crit, 
                                                 save_name="_target_shape_%d_skip_%d" % (target_shape[0], skip_value),
                                                 tds_dir=tds_dir, save=save)
    
   

if __name__ == '__main__':
    args = give_args()
    save = args.save
    pdf_pgf = args.pdf_pgf
    tds_dir = args.dir2save
    if not os.path.exists(tds_dir):
        os.makedirs(tds_dir)
    if args.index == "heatmap_combine":
        combine_detection_and_quantification_heatmap(["xception", "unified_cnn", "resnet"], 
                                                     ["validation_loss", "log_rsquare"], 
                                                     real_val=False, show=False, tds_dir=tds_dir, 
                                                     save=save)
    elif args.index == "compare_detection":
        compare_detection_prediction_curve(model_spectra=["xception", "unified_cnn", "resnet"], 
                                           perf_crit="validation_loss", 
                                           tds_dir=tds_dir, save=save)
    elif args.index == "compare_quantification":
        compare_concentration_prediction_curve(model_spectra=["xception", "unified_cnn", "resnet"], 
                                               perf_crit="log_rsquare", tds_dir=tds_dir, 
                                               save=save)
    elif args.index == "tomas_attention_maps":
        show_attention_maps_tomas(tds_dir=tds_dir, save=save)
    
    elif args.index == "dnp_attention_maps":
        show_attention_maps_dnp(tds_dir=tds_dir, save=save)
    elif args.index == "show_example_spectra":
        show_spectra(select_index=0, percentage=0.01, tds_dir=tds_dir, save=save)
    elif "attention_maps_peak_loc" in args.index:
        dataset = args.index.split("_")[-1]
        mm_obj = RelocateSelectionSersMap(dataset)
        mm_obj.show_maps_single_concentration(tds_dir=tds_dir, save=save)
    elif args.index == "combined_detection_map":
        combined_detection_prediction_curve_with_ml(tds_dir=tds_dir,
                                                    save=save)
        # combined_detection_prediction_curve(tds_dir=tds_dir,
        #                                     save=save)
    elif args.index == "combined_quantification_map":
        combined_concentration_prediction_curve_with_ml(tds_dir=tds_dir, save=save)
        # combined_concentration_prediction_curve(tds_dir=tds_dir,
        #                                     save=save)
    elif args.index == "prepare_spectra_model_arch":
        produce_spectra_model_architecture_figure(tds_dir=tds_dir, save=save)
    elif args.index == "produce_vit_model_arch":
        produce_vit_model_architecture_figure(tds_dir=tds_dir, save=save)
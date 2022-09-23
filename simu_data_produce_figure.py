#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   produce_figure_simu_data.py
@Time    :   2022/04/02 13:53:31
@Author  :   Bo 
"""
from re import M
import numpy as np
import os
import utils
import data.prepare_sers_data as psd
import pickle
import seaborn as sns
import matplotlib
from sklearn.metrics import roc_curve, auc, f1_score
import pickle
from scipy.special import softmax
from PyPDF2 import PdfFileMerger, PdfFileReader, PdfFileWriter
from PyPDF2.pdf import PageObject
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from configs.common import str2bool
import torch 


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


def get_concentration_and_peak_intensity_multiple_datasets(
    dataset_group=["SIMU_TYPE_12", "SIMU_TYPE_13", "SIMU_TYPE_14"],
    percentage_group=[0.1, 0.25, 1.0],
    tds_dir="../rs_dataset/paper_figure/",
    save=False,
):
    fig = vu.give_figure_specify_size(0.5, 3.0)
    ax_global = vu.ax_global_get(fig)
    ax_global.set_xticks([])
    ax_global.set_yticks([])
    color_use = ["r", "g", "b"]
    title_group = ["Type 1", "Type 2", "Type 3"]
    for i, s_data in enumerate(dataset_group):
        ax = fig.add_subplot(len(dataset_group), 1, i + 1)
        tr_g, val_g, tt_g = vu.get_tt_label(s_data, True)
        map_g = np.concatenate([tr_g[0], tt_g[0]], axis=0)
        conc_g = np.concatenate([tr_g[2], tt_g[2]], axis=0)
        for j, s_percent in enumerate(percentage_group):
            stat, unique_conc = get_concentration_and_peak_intensity_plot(
                map_g, conc_g, tr_g[-2][0][:2], s_percent, 
                show=False, save=False, dataset=s_data,
            )
            ax.errorbar(
                np.unique(conc_g),
                stat[:, 0],
                stat[:, 1],
                color=color_use[j],
                capsize=4,
                label="Top %d%%" % (s_percent * 100),
            )
        ax.set_xscale("symlog", linthresh=np.sort(np.unique(conc_g))[2])
        if i == len(dataset_group) - 1:
            ax.set_xticks(np.unique(conc_g))
            ax.set_xticklabels(["%.4f" % v for v in np.unique(tr_g[2])], rotation=90)
        else:
            ax.xaxis.set_major_formatter(plt.NullFormatter())
        ax.grid(ls=":")
        ax.legend(loc="best")
        ax.set_title("%s" % title_group[i])
        ax.yaxis.set_major_formatter(plt.NullFormatter())

    ax_global.set_xlabel("\n\n\n\nConcentration")
    ax_global.set_ylabel("Peak intensity\n")
    if save:
        plt.savefig(
            tds_dir + "/relation_vs_concentration_simulated_dataset.pdf",
            pad_inches=0,
            bbox_inches="tight",
        )
        
        
def get_example_sers_maps_and_spectra_figure(tds_dir="../rs_dataset/paper_figure/", save=False):
    data_set = ["SIMU_TYPE_2", "SIMU_TYPE_3", "SIMU_TYPE_4"]
    data_title = ["Type 1 (no contaminants)", 
                  "Type 2 (random contaminants at a fixed Raman shift location)",
                  "Type 3 (random contaminants at random Raman shift locations)"]
    select_concentration = [[0, 0.005, 0.5], [0, 0.025, 0.4], [0.01, 0.05, 0.3]]
    select_index = [[0, 13, 16], [18, 5, 4], [6, 25, 26]]
    for i, s_dataset in enumerate(data_set):
        fig = vu.give_figure_specify_size(1.0, 2.7 / 3.5)
        ax_global = vu.ax_global_get(fig)
        ax_global.set_xticks([])
        ax_global.set_yticks([])
        ncol = 6 
        nrow = 3 
        tr_g, val_g, tt_g = vu.get_tt_label(s_dataset, 
                                                    quantification=False, 
                                                    leave_index=0,
                                                    leave_method="leave_one_chip_per_conc",
                                                    normalization="none")
        base_index = 0 #3 * ncol * i 
        tr_maps, tr_label, tr_conc, tr_peak, tr_wave = tr_g
        vu.show_example_sers_maps(tr_maps, tr_conc, tr_peak, tr_wave, select_concentration[i], 
                                select_index[i], fig, nrow, ncol, base_index, dataset=data_set[i])
        plt.subplots_adjust(wspace=0.005, hspace=0.3)
        ax_global.set_title("\n" + data_title[i] + "\n")
        if s_dataset == "SIMU_TYPE_3":
            ax_global.set_ylabel("Concentration\n\n")
        else:
            ax_global.set_ylabel(" " + "\n\n")
        if save:
            plt.savefig(tds_dir + "/example_simulate_data_%s.pdf" % data_set[i], pad_inches=0, bbox_inches='tight')
            
            

def prepare_attention_map_example(quantile_percent=0.8, tds_dir="../rs_dataset/paper_figure/", save=False):
    conc_use = [0, 0, 0.01, 0.4, 0.5]
    select_sample_index = [42, 56, 73, 22, 3]
    im_g, attn_g, peak_g, signal_g = get_attention_map_for_single_dataset(conc_use, select_sample_index, 
                                                                          "SIMU_TYPE_14", False, True, False)
    vu.show_example_maps_and_attention_maps(im_g, attn_g, peak_g, signal_g, conc_use, np.arange(110), 
                                            quantile_percentage=quantile_percent, 
                                            save_name="type_14_sers_maps_and_attention_maps_quantile_%.2f" % quantile_percent,
                                            tds_dir=tds_dir, save=save)

def get_attention_map_for_single_dataset(conc_use, select_sample_index, dataset,
                                         detection=True, quantification=False,
                                         show=True):
    attn_path = "../exp_data/VIT/%s/detection_%s_quantification_%s/%s.obj" % (dataset, detection, 
                                                                                   quantification, dataset)
    attn_tot = pickle.load(open(attn_path, "rb"))
    signal_path = "../rs_dataset/simulate_sers_maps/%s_signal.obj" % dataset
    signal_tot = pickle.load(open(signal_path, "rb"))
    conc = np.array(attn_tot["concentration"])
    sers_map = attn_tot["sers_maps"]
    peaks = attn_tot["peak"]
    attention_map = attn_tot["attention_map"]
    unique_conc = np.unique(conc)
    print("The shape of the concentration, sers maps and peaks", np.shape(conc), 
          np.shape(sers_map), np.shape(peaks), np.shape(attention_map))
    print("The shape of the signal", len(signal_tot), len(unique_conc))
    im_g, attn_g, peak_g, signal_g = [], [], [], []
    for i, s_conc in enumerate(conc_use):
        s_sample_index = select_sample_index[i]    
        index = np.where(conc == s_conc)[0][s_sample_index]
        _attn_map = attention_map[index]
        _sers_map = sers_map[index]
        _s_index = np.where(unique_conc == s_conc)[0][0]
        _signal_group = signal_tot[_s_index]
        _num_map_group = len(_signal_group) // 3
        _tr_tt_index = np.concatenate([np.arange(len(_signal_group))[:_num_map_group], 
                                       np.arange(len(_signal_group))[-_num_map_group:]], axis=0)
        im_g.append(_sers_map)
        attn_g.append(_attn_map)
        peak_g.append(peaks[index])
        signal_g.append(_signal_group[_tr_tt_index][s_sample_index])
    print("The results image and attention maps shape", np.shape(im_g), np.shape(attn_g), np.shape(peak_g), 
           np.shape(signal_g))
    return im_g, attn_g, peak_g, signal_g

            
            
def combine_sers_maps_example_map(tds_dir="../rs_dataset/paper_figure/", save=False):
    data_path = ["SIMU_TYPE_2", "SIMU_TYPE_3", "SIMU_TYPE_4"]
    data_path = [tds_dir + "/example_simulate_data_%s.pdf" % v for v in data_path]
    pdfs = []    
    for pdf in data_path:
        pdfs.append(PdfFileReader(pdf, 'rb').getPage(0))
    
    total_width = pdfs[0].mediaBox.upperRight[0]
    height_group = [v.mediaBox.upperRight[1] for v in pdfs]
    print(height_group)
    height_cum = np.cumsum(np.array(height_group[::-1]).astype(np.float32))
    print(height_cum)
    total_height = np.sum(height_group)
    new_page = PageObject.createBlankPage(None, total_width, total_height)
    pdf_update = []
    pdf_update = [pdfs[1], pdfs[0], pdfs[2]]
    for i in range(len(pdfs)):
        if i == 0:
            new_page.mergePage(pdf_update[len(pdfs) - i - 1])
        else:
            new_page.mergeTranslatedPage(pdf_update[len(pdfs) - i - 1], 0, height_cum[len(pdfs) - i - 1])
    output = PdfFileWriter()
    output.addPage(new_page)
    output.write(open(tds_dir + "/combined_example_spectra.pdf", "wb"))
    
    
def produce_map_for_architecture(tds_dir="../rs_dataset/paper_figure/", save=False):
    data_obj = psd.ReadSERSData("SIMU_TYPE_2", target_shape=[30, 30], 
                            bg_method="ar",
                            tr_limited_conc=[0], 
                            percentage=0, top_selection_method="sers_maps", 
                            path_mom="../rs_dataset/", use_map=False, quantification=False,
                            detection=True,
                           cast_quantification_to_classification=False,
                           normalization="none", leave_index=0, 
                           skip_value=1, leave_method="none")
    [tr_maps, tr_label, tr_conc, _, _], _, _, imshape, num_class = data_obj.forward_test()
    map_use = tr_maps[-1, :, :, 25]    
    patch_size = 10 
    nrow, ncol = np.array(imshape)[:2] // patch_size 
    for i in range(nrow):
        for j in range(ncol):
            im = map_use[i * patch_size:(i+1)*patch_size, j * patch_size:(j+1)*patch_size]
            fig = plt.figure(figsize=(3, 3))
            ax = fig.add_subplot(111)
            ax.imshow(im, vmin=np.min(map_use), vmax=np.max(map_use))
            ax.set_xticks([])
            ax.set_yticks([])
            if save:
                plt.savefig(tds_dir + "/sers_patch_%d.jpg" % (i * ncol + j), pad_inches=0, bbox_inches='tight',
                           dpi=500)
            plt.close('all')
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.imshow(map_use)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(tds_dir + "/example_sers_map.jpg", pad_inches=0, bbox_inches='tight', 
                dpi=500)
                
    fig = plt.figure(figsize=(5, 2))
    ax = fig.add_subplot(111)
    map_reshape = np.reshape(tr_maps[-1], [-1, 110])
    print(np.shape(map_reshape))
    for v in map_reshape:
        ax.plot(np.arange(110), v, color='gray', alpha=0.4)
    peak_intensity = np.sum(map_reshape[:, [24, 75]], axis=-1)
    num_select = len(map_reshape) * 0.01
    top_intensity = np.argsort(peak_intensity)[-int(num_select):]
    for v in top_intensity:
        ax.plot(np.arange(110), map_reshape[v], alpha=0.4, color='r')
    ax.set_yticks([])
    if save:
        plt.savefig(tds_dir + "/example_sers_map_spectra.jpg", pad_inches=0, bbox_inches='tight', dpi=500)

    fig = plt.figure(figsize=(5, 2))
    ax = fig.add_subplot(111)
    ax.plot(np.arange(110), np.mean(map_reshape[top_intensity], axis=0), color='r')
    ax.set_yticks([])
    if save:
        plt.savefig(tds_dir + "/example_sers_map_avg_spectra.jpg", pad_inches=0, bbox_inches='tight', dpi=500)



def get_concentration_and_peak_intensity_plot(
    maps, concentration, peak_location, percentage, 
    show=False, save=False, tds_dir="../rs_dataset/paper_figure/", dataset="SIMU_TYPE_12"
):
    """Get the concentration and peak intensity plot
    Args:
        maps: [num_maps, imh, imw, num_wave]
        concentration: [num_maps]
        peak_location: [num_peaks]
        percentage: float
    """
    unique_conc = np.unique(concentration)
    peak_location_signal = peak_location[:2]
    imh, imw, wavenumber = np.shape(maps)[1:]
    num_select = int(percentage * imh * imw)
    print("Selecting %d spectra from each map" % num_select)
    avg_std_intensity = np.zeros([len(unique_conc), 2])
    avg_std_intensity_ind_peak = np.zeros([len(peak_location), len(peak_location), len(unique_conc), 2])
    for i, s_conc in enumerate(unique_conc):
        index = np.where(concentration == s_conc)[0]
        sub_maps = np.reshape(maps[index], [len(index), imh * imw, wavenumber])
        sub_peak = np.sum(sub_maps[:, :, peak_location.astype(np.int32)], axis=-1)
        sort_index = np.argsort(sub_peak, axis=-1)[:, -num_select:]
        select_value = [np.mean(sub_peak[j, v]) for j, v in enumerate(sort_index)]
        avg_std_intensity[i] = [
            np.mean(select_value),
            1.95 * np.std(select_value) / np.sqrt(len(select_value)),
        ]
        for q, q_peak in enumerate(peak_location_signal.astype(np.int32)):
            indi_spec = sub_maps[:, :, q_peak]
            sort_index = np.argsort(indi_spec, axis=-1)[:, -num_select:]
            select_value = [np.mean(indi_spec[j, v]) for j, v in enumerate(sort_index)]
            other_index = np.delete(np.arange(len(peak_location_signal)), q)
            select_other_peak_value = [np.mean(sub_maps[j, v, int(peak_location_signal[other_index])]) for j, v in enumerate(sort_index)]
            avg_std_intensity_ind_peak[q, q, i] = [np.mean(select_value), 1.95 * np.std(select_value) / np.sqrt(len(select_value))]
            avg_std_intensity_ind_peak[q, other_index, i] = [np.mean(select_other_peak_value), 1.95 * np.std(select_other_peak_value) / np.sqrt(len(select_other_peak_value))]
    if show:
        x_axis = np.arange(len(unique_conc) + 1)[1:]
        color_use = ['r', 'g', 'b']
        fig = plt.figure(figsize=(10, 3))
        ax_global = vu.ax_global_get(fig)
        ax_global.set_xticks([])
        ax_global.set_yticks([])
        for q, s_peak in enumerate(peak_location_signal):
            ax = fig.add_subplot(1, 3, q+1)
            other_index = np.delete(np.arange(len(peak_location_signal)), q)[0]
            print(other_index, np.shape(avg_std_intensity_ind_peak))
            ax.errorbar(x_axis, avg_std_intensity_ind_peak[q, q, :, 0], 
                        avg_std_intensity_ind_peak[q, q, :, 1], color=color_use[0], capsize=3, label="Peak:%d" % (s_peak), marker='o')
            ax.errorbar(x_axis, avg_std_intensity_ind_peak[q, other_index, :, 0], 
                        avg_std_intensity_ind_peak[q, other_index, :, 1], color=color_use[1], capsize=3, label="Peak:%d" % (peak_location[other_index]),
                        marker='^')
            ax.legend(loc='best')
            ax.grid(ls=":")
            ax.set_xticks(x_axis)
            ax.set_xticklabels(["%.3f" % v for v in unique_conc], rotation=90)
            ax.set_title("Sort the spectra based on peak: %d" % (s_peak))
        ax = fig.add_subplot(133)
        ax.errorbar(
            x_axis,
            avg_std_intensity[:, 0],
            avg_std_intensity[:, 1],
            color="m",
            marker="o",
            capsize=3,
        )
        ax.grid(ls=":")
        ax.set_xticks(x_axis)
        ax.set_xticklabels(["%.3f" % v for v in unique_conc], rotation=90)
        ax.set_title("Sort the spectra based on sum of the peak intensities")
        ax_global.set_xlabel("\n\n\nConcentration")
        ax_global.set_ylabel("Peak intensities\n\n\n")
        if save:
            plt.savefig(tds_dir + "/peak_intensity_and_concentration_%s_percentage_%.3f.pdf" % (dataset, percentage), pad_inches=0, bbox_inches='tight')
    return avg_std_intensity, unique_conc



def get_spectra_performance_different_percentage(
    detection,
    quantification,
    version_group,
    perf_crit,
    dataset_group=[],
    model=[],
    save=False,
    tds_dir="../rs_dataset/paper_figure/",
):
    if detection:
        if len(dataset_group) == 0:
            dataset_group = ["SIMU_TYPE_2", "SIMU_TYPE_3", "SIMU_TYPE_4"]
        if len(model) == 0:
            model = ["xception", "unified_cnn", "resnet"]
    else:
        if len(dataset_group) == 0:
            dataset_group = ["SIMU_TYPE_12", "SIMU_TYPE_13", "SIMU_TYPE_14"]
        if len(model) == 0:
            model=["xception", "unified_cnn", "resnet"]
    val_perf_group = []
    for i, s_data in enumerate(dataset_group):
        for j, s_method in enumerate(model):
            obj = GetSpectraPerformanceSIMU(
                s_data,
                s_method,
                detection,
                quantification,
                version_group,
                avg_spectra=True,
                perf_crit=perf_crit,
                loc="home",
            )
            val_perf, val_pred = obj.get_performance_diagonal(perf_crit, show=False)
            if quantification:
                if perf_crit == "rsquare" or perf_crit == "log_rsquare":
                    val_perf = np.maximum(val_perf, np.zeros_like(val_perf))
            # if detection:
                # if perf_crit == "validation_loss":
                #     val_perf = np.log(val_perf)
            val_perf_group.append(val_perf)
    # val_perf_group = [np.random.randint(0, 100, [4, 10, 10]) for _ in range(len(dataset_group) * len(model))]
    print("The shape of the validation prediction", np.shape(val_perf_group), np.min(val_perf_group), np.max(val_perf_group))
    if quantification:
        if perf_crit == "rsquare":
            min_value, max_value = 0, 1.0 
        elif perf_crit == "log_rsquare":
            min_value, max_value = 0, 0.25
    else:
        if perf_crit == "global_accu":
            min_value, max_value = 50, 72.5
        else:
            min_value, max_value = np.min(val_perf_group), np.max(val_perf_group)
    percentage = [0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
    selection_method = ["Peak", "Std", "Mean", "Diff"]
    data_set = [
        "Type 1 (no contaminants)",
        "Type 2 (random contaminants at a fixed Raman shift location)",
        "Type 3 (random contaminants at random Raman shift locations)",
    ]
    data_set = np.repeat(data_set, len(model))
    model_title = ["Xception", "U-CNN", "ResNet"]
    model_use_repeat = np.reshape([model_title for _ in range(len(data_set))], [-1])
    fig = vu.give_figure_specify_size(1.0, 2.6)
    ax_global = vu.ax_global_get(fig)
    ax_global.set_xticks([])
    ax_global.set_yticks([])
    nrow = int(len(dataset_group) * len(model))
    ncol = 4
    ax_group = []
    cmap = ["Blues_r" if perf_crit == "validation_loss" else "Blues"][0]
    for i in range(nrow):
        if perf_crit == "validation_loss":
            ax_group = []
            min_value, max_value = np.min(val_perf_group[i]), np.max(val_perf_group[i])
        ax = fig.add_subplot(nrow, 1, i + 1, frameon=False)
        ax.set_title("%s + %s" % (data_set[i], model_use_repeat[i]), pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        for j in range(ncol):
            ax = fig.add_subplot(nrow, ncol, i * ncol + j + 1)
            if j == 0:
                yticklabel = ["%.1f" % (v * 100) for v in percentage]
            else:
                yticklabel = ["" for v in percentage]
                ax.set_yticks([])
            if i == nrow - 1:
                xticklabel = ["%.1f" % (v * 100) for v in percentage]
            else:
                xticklabel = ["" for v in percentage]
            im = sns.heatmap(
                val_perf_group[i][j],
                vmin=min_value,
                vmax=max_value,
                ax=ax,
                yticklabels=yticklabel,
                xticklabels=xticklabel,
                cbar=False,
                cmap=cmap,
            )
            if i == 0:
                ax.set_title(selection_method[j] + "\n")
            if j != 0:
                ax.set_yticks([])
            if i != nrow - 1:
                ax.set_xticks([])
            ax_group.append(ax)
        if perf_crit == "validation_loss":
            cax = plt.axes([0.92, 0.113 + 0.78 / nrow * (nrow - i - 1), 0.015, 0.78 / (nrow + 2)])
            mappable = im.get_children()[0]
            plt.colorbar(mappable, cax=cax, ax=ax_group)
    if perf_crit != "validation_loss":
        cax = plt.axes([0.92, 0.1, 0.025, 0.78])
        mappable = im.get_children()[0]
        plt.colorbar(mappable, cax=cax, ax=ax_group)
    ax_global.set_xlabel("\n\n\nPercentage of spectra used at testing (%)")
    ax_global.set_ylabel("Percentage of spectra used at training (%)\n\n\n")
    plt.subplots_adjust(wspace=0.05, hspace=0.2)
    if save:
        plt.savefig(
            tds_dir
            + "/heatmap_detection_%s_quantification_%s_for_the_simulated_data_%s.pdf"
            % (detection, quantification, perf_crit),
            pad_inches=0,
            bbox_inches="tight",
        )
        
        
def get_detection_performance_different_selection_methods(perf_crit="global_accu", 
                                                          tds_dir="../rs_dataset/paper_figure/",
                                                          save=False, 
                                                          fix_percent=False):
    data_type = ["SIMU_TYPE_2", "SIMU_TYPE_3", "SIMU_TYPE_4"]
    model_group = ["xception", "unified_cnn", "resnet"]
    model_label = ["Xception", "U-CNN", "ResNet"]
    data_set = [
        "Type 1 (no contaminants)",
        "Type 2 (random contaminants at a fixed Raman shift location)",
        "Type 3 (random contaminants at random Raman shift locations)",
    ]
    tt_collect_group = []
    label_group, conc_group = [], []
    vit_accu_group = []
    version_spectra = [20, 21, 22, 23, 24]
    for i, s_data in enumerate(data_type):
        tt_collect_per_data = []
        for j, s_model in enumerate(model_group):
            obj = GetSpectraPerformanceSIMU(s_data, s_model, True, False, version_spectra,
                                            True, "global_accu", loc="home")
            _, _, _, tt_collect, label_use = obj.get_performance_baseon_validation_perf(perf_crit, show=False, fix_percent=fix_percent)
            tt_collect_per_data.append([v[-1] for v in tt_collect])
        label_group.append(obj.tt_label)
        conc_group.append(obj.tt_conc)
        tt_collect_group.append(tt_collect_per_data) # [4, 880, 2]
        obj = GetPerformanceSIMUDataset(s_data, True, False, 0.2, 2, 0, [], save=False)
        vit_accu, _, vit_gp, vit_f1 = obj.get_detection_results(show=False)
        vit_accu_group.append(np.concatenate([vit_accu, [vit_gp]], axis=0))
        
    x_axis = np.arange(len(np.unique(conc_group[0])) + 1)[1:]
    fig = vu.give_figure_specify_size(1.0, 2.7 / 1.9)
    nr, nc = len(data_type), len(model_group) + 1
    spec5 = fig.add_gridspec(ncols=nc, nrows=nr, width_ratios=[4 for _ in model_group] + [1])
    ax_global = vu.ax_global_get(fig)
    ax_global.set_xticks([])
    ax_global.set_yticks([])
    ax_group = []
    color_use = ['r', 'g', 'b', 'orange']
    for i, s_data in enumerate(data_type):
        ax_title = fig.add_subplot(nr, 1, i+1, frameon=False)
        ax_title.set_xticks([])
        ax_title.set_yticks([])
        ax_title.set_title(data_set[i])
        for j, s_model in enumerate(model_group):
            pred_data = tt_collect_group[i][j]
            ax = fig.add_subplot(spec5[i, j])
            perf_conc = np.zeros([len(pred_data), len(np.unique(conc_group[0])) + 1])
            for q in range(len(pred_data)):
                pred_label = np.argmax(pred_data[q], axis=-1)
                perf = np.sum(pred_label == label_group[i]) / len(label_group[i])
                accu_per_conc, \
                    std_per_conc, \
                    xtick_label = vu.show_accuracy_over_concentration(pred_data[q], pred_label, 
                                                                      label_group[i], 
                                                                      conc_group[i], show=False)
                perf_conc[q, :-1] = np.array(accu_per_conc) * 100 
                perf_conc[q, -1] = (perf * 100)
            if j == 0:
                yticklabel = ["%.4f" % v for v in np.unique(conc_group[0])] + ["Average"]
            else:   
                yticklabel = ["" for v in xtick_label] + [""]
            if i == len(data_set) - 1:
                xticklabel = label_use
            else:
                xticklabel = ["" for v in label_use]           
            im=sns.heatmap(np.round(perf_conc, 0).astype(np.int8).T, annot=True, ax=ax, cbar=False, cmap="Blues", 
                        xticklabels=xticklabel, yticklabels=yticklabel, vmin=25, vmax=100, fmt="d", annot_kws={"fontsize":7})
            ax.tick_params(axis='both', which='major', pad=0.5)
            ax.axhline(12, color='white', lw=1.5)
            ax.tick_params(tick1On=False)
            if i == 0:
                ax.set_title(model_label[j] + "\n")
            ax_group.append(im)          
        ax = fig.add_subplot(spec5[i, j+1])
        if i == len(data_set) - 1:
            xticklabel = ["Map"]
        else:
            xticklabel = [""]
        yticklabel = ["" for v in label_use]
        im = sns.heatmap(np.expand_dims(np.round(vit_accu_group[i] * 100), axis=0).astype(np.int8).T, annot=True, ax=ax, cbar=False, 
                         cmap="Blues", xticklabels=xticklabel, yticklabels=yticklabel, fmt="d", 
                         vmin=20, vmax=100, annot_kws={"fontsize":7})
        if i == 0:
            ax.set_title("ViT" + "\n")
        ax.axhline(12, color='white', lw=1.5)
        ax.tick_params(tick1On=False)
        ax_group.append(im)
        ax.tick_params(axis='both', which='major', pad=0.5)
    cax = plt.axes([0.92, 0.13, 0.025, 0.75])
    mappable = im.get_children()[0]
    plt.colorbar(mappable, cax=cax, ax=ax_group)
    ax_global.set_xlabel("\nSelection criteria")
    ax_global.set_ylabel("Concentration\n\n\n\n")    
    plt.subplots_adjust(wspace=0.05, hspace=0.2)
    if save:
        if fix_percent:
            plt.savefig(tds_dir + "/detection_performance_using_each_of_the_selection_criteria_100_100.pdf", pad_inches=0, bbox_inches='tight')
        else:
            plt.savefig(tds_dir + "/detection_performance_using_each_of_the_selection_criteria.pdf", pad_inches=0, bbox_inches='tight')
        
import shutil
def move_dir():
    path_org = "/scratch/blia/exp_data/Spectra_xception/SIMU_TYPE_14/detection_False_quantification_True_average_spectra_True/"
    path2move = sorted([v for v in os.listdir(path_org) if "version_10_" in v and "top_peak" in v])[1:]
    print(len(path2move))
    for v in path2move:
        print(v)
    path_des = "/nobackup/blia/exp_data/Spectra_xception/SIMU_TYPE_14/detection_False_quantification_True_average_spectra_True/"
    for v in path2move:
        shutil.move(path_org + v, path_des + v)
        
        
def get_quantification_performance_different_selection_methods(perf_crit="rsquare", 
                                                              tds_dir="../rs_dataset/paper_figure/",
                                                              data_type=[], 
                                                              model_group=[],
                                                              save=False):
    if len(data_type) == 0:
        data_type = ["SIMU_TYPE_12", "SIMU_TYPE_13", "SIMU_TYPE_14"]
    if len(model_group) == 0:
        model_group = ["xception", "unified_cnn"]
    model_label = ["Xception", "U-CNN"]
    data_title = [
        "Type 1 (no contaminants)",
        "Type 2 (random contaminants at fixed Raman shift)",
        "Type 3 (random contaminants at random Raman shift)",
    ]
    tt_collect_group = []
    label_group, conc_group = [], []
    version_group = {}
    version_group["SIMU_TYPE_12"] = [10, 11, 12, 13, 14]
    version_group["SIMU_TYPE_13"] = [10, 11, 12, 13, 14]
    version_group["SIMU_TYPE_14"] = [10, 11, 12, 13, 14]
    for i, s_data in enumerate(data_type):
        tt_collect_per_data = []
        for j, s_model in enumerate(model_group):
            obj = GetSpectraPerformanceSIMU(s_data, s_model, False, True, version_group[s_data],
                                            True, perf_crit, loc="home")
            best_tt_pred, _, tt_collect, label_use = obj.get_performance_baseon_validation_perf(perf_crit, show=False)
            tt_collect_per_data.append([v[-1] for v in tt_collect])
        
        label_group.append(obj.tt_label)
        conc_group.append(obj.tt_conc)
        tt_collect_group.append(tt_collect_per_data) # [4, 880, 2]
    x_axis = np.arange(len(np.unique(conc_group[0])) + 1)[1:]
    fig = vu.give_figure_specify_size(1.0, 2.7 / 3)
    ax_global = vu.ax_global_get(fig)
    ax_global.set_xticks([])
    ax_global.set_yticks([])
    nr, nc = len(data_type), len(model_group)
    color_use = ['r', 'g', 'b', 'orange']
    for i, s_data in enumerate(data_type):
        ax_title = fig.add_subplot(nr, 1, i+1, frameon=False)
        ax_title.set_xticks([])
        ax_title.set_yticks([])
        ax_title.set_title(data_title[i])
        for j, s_model in enumerate(model_group):
            pred_data = tt_collect_group[i][j]
            ax = fig.add_subplot(nr, nc, i * nc + j + 1)
            for q in range(len(pred_data)):
                _rsquare = vu.get_performance_baseon_crit_quantification(conc_group[i], s_pred, "log_rsquare")
                s_pred = pred_data[q]
                _, _pred_conc = utils.get_rsquare_curve_stat(conc_group[i], s_pred)
                ax.plot(x_axis, _pred_conc[:, 0], color=color_use[q], marker='.', label=label_use[q] + ": %.2f" % _rsquare)
                # ax.errorbar(x_axis, _pred_conc[:, 0], _pred_conc[:, 1], color=color_use[q], marker='.', capsize=4, label=label_use[q])
            ax.legend(loc='best', ncol=2)            
            ax.plot(x_axis[1:], np.unique(conc_group[i])[1:], color='gray', ls='-')
            ax.set_xscale("log")
            ax.set_yscale("log")
            if i == len(data_type) - 1:
                ax.set_xticks(x_axis)
                ax.set_xticklabels(["%.4f" % v for v in np.unique(conc_group[i])], rotation=90)
            else:
                ax.xaxis.set_major_formatter(plt.NullFormatter())
            if j == len(model_group) - 1:
                ax.yaxis.set_major_formatter(plt.NullFormatter())
    ax_global.set_xlabel("\n\n\n\nConcentration")
    ax_global.set_ylabel("Predicted concentration (%)\n\n\n")    
    plt.subplots_adjust(wspace=0.05, hspace=0.3)
    if save:
        plt.savefig(tds_dir + "/detection_performance_using_each_of_the_selection_criteria.pdf", pad_inches=0, bbox_inches='tight')        


def get_boxplot_detection_performance_different_approaches(perf_crit="global_accu", 
                                                           tds_dir="../rs_dataset/paper_figure/", 
                                                           save=False, show=True):
    data_type = ["SIMU_TYPE_2", "SIMU_TYPE_3", "SIMU_TYPE_4"]
    model_group = ["xception", "unified_cnn", "resnet"]
    data_title = [
        "Type 1 (no contaminants)",
        "Type 2 (random contaminants at a fixed Raman shift location)",
        "Type 3 (random contaminants at random Raman shift locations)",
    ]
    perf_tot = []
    version_spectra = [20, 21, 22, 23, 24]
    for i, s_data in enumerate(data_type):
        accuracy = []
        for j, s_model in enumerate(model_group):
            obj = GetSpectraPerformanceSIMU(
                s_data,
                s_model,
                True,
                False,
                version_spectra,
                True,
                perf_crit=perf_crit,
                loc="home",
            )
            (
                accu_per_conc,
                std_per_conc,
                x_ticklabel, _, _,
            ) = obj.get_performance_baseon_validation_perf(perf_crit, show=False)
            accuracy.append(np.concatenate([[accu_per_conc], [std_per_conc]], axis=0))

        obj = GetPerformanceSIMUDataset(s_data, True, False, 0.2, 2, 0, [], save=False)
        accu_per_conc, std_per_conc, vit_gp, vit_f1 = obj.get_detection_results(show=False)
        accuracy.append(np.concatenate([[accu_per_conc], [std_per_conc]], axis=0))
        perf_tot.append(accuracy)
    
    if show:
        x_ticklabel = ["%.4f" % v for v in [0, 0.0025, 0.0050, 0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5]]
        model_title = ["Xception", "U-CNN", "ResNet", "ViT"]
        # perf_tot = []
        # for i in range(len(data_title)):
        #     _d, _r = [], []
        #     for j in range(len(model_title)):
        #         _d.append([np.random.random(len(x_ticklabel)), np.random.random(len(x_ticklabel))])
        #     perf_tot.append(_d)
        color_use = ["r", "g", "b", "m"]
        x_axis = [float(v) for v in x_ticklabel]
        fig = vu.give_figure_specify_size(0.5, 3.0)
        ax_global = vu.ax_global_get(fig)
        ax_global.set_xticks([])
        ax_global.set_yticks([])
        for i in range(3):
            ax = fig.add_subplot(3, 1, i + 1)
            if i == 0:
                for j in range(len(model_group) + 1):
                    ax.plot([], [], color=color_use[j], label=model_title[j])
                # ax.legend(
                #     loc="lower right", handlelength=0.8, handletextpad=0.3, borderaxespad=0.3
                # )
                ax.legend(ncol=len(model_group)+1, bbox_to_anchor=(0, 1),handlelength=1.0, columnspacing=1.6,
                          loc='lower left')
            stat = perf_tot[i]
            for j in range(len(model_group) + 1):
                s_stat = stat[j]
                ax.errorbar(x_axis, s_stat[0], s_stat[1], color=color_use[j], capsize=3)
            ax.set_xscale("symlog", linthresh=x_axis[2])
            ax.grid(ls=":")
            ax.set_xticks(x_axis)
            ax.set_xticklabels(x_ticklabel, rotation=90)
            if i != len(data_type) - 1:
                ax.xaxis.set_major_formatter(plt.NullFormatter())
            if i == 0:
                ax.set_title(data_title[i]+"\n\n")
            else:
                ax.set_title(data_title[i])
        ax_global.set_xlabel("\n\n\n\nConcentration")
        ax_global.set_ylabel("Detection accuracy\n\n\n")
        plt.subplots_adjust(hspace=0.2)
        if save:
            plt.savefig(
                tds_dir + "/detection_global_accuracy_comparision_simulated_data.pdf",
                pad_inches=0,
                bbox_inches="tight",
            )
    else:
        return model_group, data_title, x_axis, perf_tot


def get_boxplot_quantification_performance_different_approaches(
    perf_crit="global_accu", 
    data_type=[],
    model_group=[],
    tds_dir="../rs_dataset/paper_figure/", save=False, show=True,
):
    if len(data_type) == 0:
        data_type = ["SIMU_TYPE_12", "SIMU_TYPE_13", "SIMU_TYPE_14"]
    if len(model_group) == 0:
        model_group = ["xception", "unified_cnn", "resnet"]
    version_spectra = [200, 201, 202, 203, 204]
    data_title = [
        "Type 1 (no contaminants)",
        "Type 2 (random contaminants at a fixed Raman shift location)",
        "Type 3 (random contaminants at random Raman shift locations)",
    ]
    pred_tot = []
    for i, s_data in enumerate(data_type):
        pred_s_model = []
        for j, s_model in enumerate(model_group):
            obj = GetSpectraPerformanceSIMU(
                s_data,
                s_model,
                False,
                True,
                version_spectra,
                True,
                perf_crit=perf_crit,
                loc="home",
            )
            best_pred, conc, tt_collect, label_use = obj.get_performance_baseon_validation_perf(perf_crit, show=False)
            pred_s_model.append(best_pred)
        obj = GetPerformanceSIMUDataset(s_data, False, True, 0.08, 2, 0, [], save=False)
        vit_pred, _ = obj.get_quantification_results(show=False)
        pred_s_model.append(vit_pred)
        print([np.shape(v) for v in pred_s_model])
        pred_tot.append(pred_s_model)
    
    rsquare_status = []
    for i in range(len(data_type)):
        stat = pred_tot[i]
        _rsquare = []
        for j in range(len(model_group) + 1):
            s_stat = stat[j]
            _, _rsquare_status = utils.get_rsquare_curve_stat(conc, s_stat)
            _rsquare.append(_rsquare_status)
        rsquare_status.append(_rsquare)
    model_title = ["Xception", "U-CNN", "ResNet", "ViT"]    
    if show:
        x_axis = np.unique(conc)
        conc_unique = np.unique(conc)
        color_use = ["r", "g", "b", "m"]
        fig = vu.give_figure_specify_size(0.5, 3.0)
        ax_global = vu.ax_global_get(fig)
        ax_global.set_xticks([])
        ax_global.set_yticks([])
        for i in range(len(data_type)):
            ax = fig.add_subplot(len(data_type), 1, i + 1)
            for j in range(len(model_group) + 1):
                ax.plot([], [], color=color_use[j], label=model_title[j])
            if i == 0:
                ax.legend(ncol=len(model_group)+1, bbox_to_anchor=(0, 1),handlelength=1.0, columnspacing=1.6,
                        loc='lower left')
            stat = rsquare_status[i]
            for j in range(len(model_group) + 1):
                s_stat = stat[j]
                ax.errorbar(x_axis, s_stat[:, 0], s_stat[:, 1], color=color_use[j], 
                            capsize=3)            
            ax.plot(x_axis, conc_unique, color='gray', ls='-')
            ax.set_xscale("symlog", linthresh=np.sort(conc_unique)[2])
            ax.set_yscale("symlog", linthresh=np.sort(conc_unique)[2])
            ax.set_ylim((0, np.max(conc_unique)))
            ax.set_xticks(x_axis)
            ax.set_xticklabels(["%.4f" % v for v in conc_unique], rotation=90)

            axins = zoomed_inset_axes(ax, 2.1, loc="lower right") # zoom = 6
            zoom_start = 4
            for j in range(len(model_group) + 1):
                s_stat = stat[j]
                axins.plot(x_axis[-zoom_start:], s_stat[:, 0][-zoom_start:], color=color_use[j], alpha=0.5, marker='.')
            axins.plot(x_axis[-zoom_start:], conc_unique[-zoom_start:], color='gray', ls='-', alpha=0.5)
            axins.set_aspect(0.4)
            axins.set_xscale("symlog", linthresh=np.sort(conc_unique)[2])
            axins.set_yscale("symlog", linthresh=np.sort(conc_unique)[2])
            axins.set_xticks(x_axis[-zoom_start:])
            axins.set_xticklabels(["%.1f" % v for v in x_axis[-zoom_start:]])
            axins.set_yticks(x_axis[-zoom_start:])
            axins.set_yticklabels(["%.1f" % v for v in x_axis[-zoom_start:]])
            axins.tick_params(axis='both', pad=2)
            axins.set_ylim((x_axis[-zoom_start]-0.1, x_axis[-1] + 0.3))
            if i != len(data_type) - 1:
                ax.xaxis.set_major_formatter(plt.NullFormatter())
            if i == 0:
                ax.set_title(data_title[i]+"\n\n")
            else:
                ax.set_title(data_title[i])
        ax_global.set_xlabel("\n\n\n\nConcentration")
        ax_global.set_ylabel("Predicted concentration\n\n\n")
        plt.subplots_adjust(hspace=0.2)
        if save:
            plt.savefig(
                tds_dir + "/quantification_global_accuracy_comparision_simulated_data.pdf",
                pad_inches=0,
                bbox_inches="tight",
            )
    else:
        return model_title, data_title, np.unique(conc), rsquare_status


def combine_detection_and_quantification_results(perf_crit=["validata_loss", "log_rsquare"], tds_dir="../rs_dataset/paper_figure/", 
                                                 save=False):
    # _, _, _, detect_perf = get_boxplot_detection_performance_different_approaches(perf_crit[0], tds_dir=tds_dir, save=False, show=False)
    # model_title, data_title, x_axis, rsquare_status = get_boxplot_quantification_performance_different_approaches(perf_crit[1], [], [], 
    #                                                                                                               tds_dir=tds_dir, save=False, show=False)
    model_title = ["Xception", "U-CNN", "ResNet", "ViT"]
    data_title = [
        "Type 1\n(no contaminants)",
        "Type 2\n(random contaminants at a fixed Raman shift location)",
        "Type 3\n(random contaminants at random Raman shift locations)",
    ]
    x_axis = [0, 0.0025, 0.0050, 0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5]
    detect_perf = []
    rsquare_status = []
    for i in range(len(data_title)):
        _d, _r = [], []
        for j in range(len(model_title)):
            _d.append([np.random.random(len(x_axis)), np.random.random(len(x_axis))])
            _r.append(np.random.random([len(x_axis), 2]))
        detect_perf.append(_d)
        rsquare_status.append(_r)
    
    fig = vu.give_figure_specify_size(1.0, 0.5)
    ax_global = vu.ax_global_get(fig)
    ax_global.set_xticks([])
    ax_global.set_yticks([])
    color_group = ['r', 'g', 'b', 'm']
    for i, s_dataset in enumerate(data_title):
        s_stat = [detect_perf[i], rsquare_status[i]]
        for j, s_s_stat in enumerate(s_stat):
            ax = fig.add_subplot(2, len(data_title), i + 1+ j * 3)
            if j == 0:
                for q, s_model in enumerate(model_title):
                    ax.errorbar(x_axis, s_s_stat[q][0], s_s_stat[q][1], color=color_group[q], capsize=3, label=s_model)
                ax.set_ylim((0, 1.1))
            else:
                for q, s_model in enumerate(model_title):
                    ax.errorbar(x_axis, s_s_stat[q][:, 0], s_s_stat[q][:, 1], color=color_group[q], capsize=3, label=s_model)
                ax.plot(x_axis, x_axis, color='gray')
            ax.set_xscale("symlog", linthresh=x_axis[2])
            if j == 1:
                ax.set_xticks(x_axis)
                ax.set_xticklabels(["%.4f" % v for v in x_axis])
                ax.set_yscale("symlog", linthresh=x_axis[2])
            else:
                ax.xaxis.set_major_formatter(plt.NullFormatter())
            if i != 0:
                ax.yaxis.set_major_formatter(plt.NullFormatter())
            if i == 0 and j == 0:
                ax.set_ylabel("Detection accuracy")
            if i == 0 and j == 1:
                ax.set_ylabel("Predicted concentration")
        ax = fig.add_subplot(1, len(data_title), i+1, frameon=False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(s_dataset)
    ax_global.set_xlabel("\n\n\n\nconcentration")
    plt.subplots_adjust(wspace=0.01)
            
            

def check_detection_accuracy_at_each_concentration(perf_crit="global_accu", tds_dir="../rs_dataset/paper_figure/",
                                                   save=False):
    data_type = "SIMU_TYPE_2"
    model_use = ["xception", "unified_cnn", "resnet"]
    model_label = ["Xception", "U-CNN", "ResNet"]
    val_tt = ["validation"] #, "testing"]
    version_use = [20, 21, 22, 23, 24]
    fig = vu.give_figure_specify_size(1.0, 0.5)
    ax_global = vu.ax_global_get(fig)
    ax_global.set_xticks([])
    ax_global.set_yticks([])
    if perf_crit == "global_accu":
        vmin, vmax = 0, 100
    ax_group = []
    cmap_use = ["Blues" if perf_crit == "global_accu" else "Blues_r"][0]
    for i, s_model in enumerate(model_use):
        obj = GetSpectraPerformanceSIMU(data_type, s_model, True, False, version_use, True,
                                        perf_crit=perf_crit, loc="home")
        _, \
        ensemble_perf_conc, \
        _, _ = obj._get_perf_stat("top_peak", ["top_peak"], perf_crit=perf_crit, show=False)
        perc = obj.percentage
        for j, s_perf in enumerate(ensemble_perf_conc[:1]):
            s_perf = (s_perf * 100).astype(np.int8)
            print(np.min(s_perf), np.max(s_perf))
            ax_title = fig.add_subplot(len(model_use), 1, i+1, frameon=False)
            ax_title.set_xticks([])
            ax_title.set_yticks([])
            ax_title.set_title(model_label[i], pad=2)
            for q, s_conc in enumerate(np.unique(obj.tt_conc)):
                _perf = s_perf[0][:, :, q]
                ax = fig.add_subplot(len(model_use), len(np.unique(obj.tt_conc)),  i * len(np.unique(obj.tt_conc)) + q + 1)
                im = sns.heatmap(_perf,
                            annot=False, 
                            vmin=vmin, vmax=vmax, 
                            cmap=cmap_use, ax=ax, cbar=False)
                ax_group.append(im)
                ax.set_xticks([])
                ax.set_yticks([])
                if i == len(model_use) - 1:
                    ax.set_xlabel("%.4f" % s_conc)
                ax.tick_params(tick1On=False)
    cax = plt.axes([0.92, 0.13, 0.025, 0.75])
    mappable = im.get_children()[0]
    plt.colorbar(mappable, cax=cax, ax=ax_group)        
    ax_global.set_xlabel("\n# spectra used for testing")
    ax_global.set_ylabel("# spectra used for training")
    # ax_global.set_title("Detection accuracy on %s dataset at each concentration (%s)\n\n" % (val_tt[j], s_model))
    plt.subplots_adjust(wspace=0.03, hspace=0.23)
    if save:
        plt.savefig(tds_dir + "/detection_performance_per_conc_%s_%s_%s.pdf" % (val_tt[j], 
                                                                                s_model, data_type), 
                    pad_inches=0, bbox_inches='tight')


class GetSpectraPerformanceSIMU(object):
    def __init__(
        self,
        dataset,
        model,
        detection,
        quantification,
        version_group,
        avg_spectra=True,
        perf_crit="rsquare",
        loc="home",
    ):
        super(GetSpectraPerformanceSIMU).__init__()
        self.dataset = dataset
        self.model = model
        self.detection = detection
        self.quantification = quantification
        self.version_group = version_group
        self.avg_spectra = avg_spectra
        self.perf_crit = perf_crit
        _, [_, val_label, val_conc, _], [_, tt_label, tt_conc, _, _] = vu.get_tt_label(
            self.dataset, self.quantification
        )
        self.val_label, self.val_conc = val_label, val_conc
        self.tt_label, self.tt_conc = tt_label, tt_conc
        val_conc_update = self.val_conc.copy()
        val_conc_update[val_conc_update == 0] = 1e-6
        tt_conc_update = self.tt_conc.copy()
        tt_conc_update[tt_conc_update == 0.0] = 1e-6
        self.tt_conc_update = tt_conc_update
        self.val_conc_update = val_conc_update
        self.loc = loc
        self.percentage = [0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
        self.tt_method = ["top_peak", "top_std", "top_mean", "top_diff"]

    def _get_perf_stat(
        self,
        tr_selection_criteria,
        tt_selection_method=["top_peak", "top_std", "top_mean", "top_diff"],
        perf_crit="f1_score",
        show=False,
    ):
        prediction_group_val, prediction_group_tt = [], []
        label_group, conc_group = np.concatenate(
            [self.val_label, self.tt_label], axis=0
        ), np.concatenate([self.val_conc, self.tt_conc], axis=0)
        val_tt_group = np.concatenate(
            [np.ones([len(self.val_label)]), np.zeros([len(self.tt_label)])], axis=0
        )
        val_conc_update = self.val_conc.copy()
        val_conc_update[val_conc_update == 0.0] = 1e-6 
        tt_conc_update = self.tt_conc.copy()
        tt_conc_update[tt_conc_update == 0.0] = 1e-6 
        val_index = np.where(val_tt_group == 1)[0]
        tt_index = np.where(val_tt_group == 0)[0]
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
                leave_index=0,
                normalization="none",
                perf_crit=perf_crit_temp,
                version=s_version,
                loc=self.loc,
            )
            if self.quantification:
                raw_prediction = np.maximum(raw_prediction, 0)
            prediction_group_val.append(np.array(raw_prediction)[:, :, :, val_index])
            prediction_group_tt.append(np.array(raw_prediction)[:, :, :, tt_index])
        (
            ensemble_perf_val,
            ensemble_perf_val_conc,
            ensemble_predction_val,
        ) = self._get_performance(
            prediction_group_val,
            stat,
            tt_selection_method,
            perf_crit,
            label_use=self.val_label,
            conc_use=val_conc_update,
            show=show,
        )
        ensemble_perf_val = np.nan_to_num(ensemble_perf_val)
        (
            ensemble_perf_tt,
            ensemble_perf_tt_conc,
            ensemble_prediction_tt,
        ) = self._get_performance(
            prediction_group_tt,
            stat,
            tt_selection_method,
            perf_crit,
            label_use=self.tt_label,
            conc_use=tt_conc_update,
            show=show,
        )
        ensemble_perf_tt = np.nan_to_num(ensemble_perf_tt)
        return (
            [ensemble_perf_val, ensemble_perf_tt],
            [ensemble_perf_val_conc, ensemble_perf_tt_conc],
            [ensemble_predction_val, ensemble_prediction_tt],
            stat[:-2],
        )

    def _get_performance(
        self,
        prediction_group,
        stat,
        tt_selection_method,
        perf_crit,
        label_use=[],
        conc_use=[],
        show=False,
    ):
        ensemble_prediction = np.mean(
            np.array(prediction_group), axis=0
        )  # [len(tt_selection_method), 10, 10, ]
        ensemble_perf = np.zeros(
            [len(tt_selection_method), len(self.percentage), len(self.percentage)]
        )
        ensemble_perf_per_concentration = np.zeros(
            [
                len(tt_selection_method),
                len(self.percentage),
                len(self.percentage),
                len(np.unique(self.tt_conc)),
            ]
        )
        label_use_tensor = torch.from_numpy(label_use).long()
        ensemble_prediction_tensor = torch.from_numpy(ensemble_prediction).to(torch.float32)
        for i in range(len(tt_selection_method)):
            for j in range(len(self.percentage)):
                for m in range(len(self.percentage)):
                    if perf_crit == "validation_loss":
                        pred_t = ensemble_prediction_tensor[i, j, m]
                    else:
                        pred_t = ensemble_prediction[i, j, m]
                    pred = ensemble_prediction[i, j, m]
                    if self.detection:
                        accu = vu.get_performance_baseon_crit_detection(
                            label_use, pred_t, perf_crit, label_use_tensor, 
                        )
                        accu_per_conc, _, _ = vu.show_accuracy_over_concentration(
                            pred,
                            np.argmax(pred, axis=-1),
                            label_use,
                            conc_use,
                            show=False,
                            save=False,
                            tds_dir=None,
                        )
                        ensemble_perf_per_concentration[i, j, m] = (
                            np.array(accu_per_conc) 
                        )
                    if self.quantification:
                        accu = vu.get_performance_baseon_crit_quantification(
                            conc_use, pred, perf_crit
                        )
                    ensemble_perf[i, j, m] = accu
        if show:
            vu.show_accuracy_spectra_heatmap(
                ensemble_perf, stat[:-2], [], True, tt_selection_method
            )
            self._show_accuracy_at_diff_concentration(
                ensemble_perf_per_concentration, stat[:-2], tt_selection_method
            )
        return ensemble_perf, ensemble_perf_per_concentration, ensemble_prediction

    def _get_best_perf_index(
        self,
        performance_matrix,
        pred_matrix,
        label_use,
        best_index,
        perf_crit,
        show=False,
        use_tt_method=None,
    ):
        key_use = np.zeros_like(performance_matrix).astype(str)
        for i, s_method in enumerate([use_tt_method]):
            for j, s_per in enumerate(self.percentage):
                for q, s_tr_per in enumerate(self.percentage):
                    key_use[i, j, q] = "_".join(
                        ["%s" % v for v in [s_method, s_per, s_tr_per]]
                    )
        key_reshape = np.reshape(key_use, [-1])
        performance_reshape = np.reshape(performance_matrix, [-1])
        if self.detection:
            pred_reshape = np.reshape(
                pred_matrix, [-1, len(label_use), len(np.unique(self.tt_label))]
            )
        if self.quantification:
            pred_reshape = np.reshape(pred_matrix, [-1, len(label_use)])
        if len(best_index) == 0:
            if perf_crit == "global_accu" or perf_crit == "f1_score" or perf_crit == "rsquare" or perf_crit == "log_rsquare":
                best_index = np.argmax(performance_reshape)
            elif perf_crit == "mae" or perf_crit == "rmae" or perf_crit == "validation_loss":
                best_index = np.argmin(performance_reshape)
            # if self.detection:
            #     _calc_perf = vu.get_performance_baseon_crit_detection(
            #         self.val_label, pred_reshape[best_index], perf_crit
            #     )
            # if self.quantification:
            #     _calc_perf = vu.get_performance_baseon_crit_quantification(
            #         self.val_conc, pred_reshape[best_index], perf_crit
            #     )
            # if show:
            #     print(
            #         "The selected key and the corresponding performance",
            #         key_reshape[best_index],
            #         performance_reshape[best_index],
            #         _calc_perf,
            #     )
        else:
            best_index = best_index[0]
            # if self.detection:
            #     _calc_perf = vu.get_performance_baseon_crit_detection(
            #         self.tt_label, pred_reshape[best_index], perf_crit
            #     )
            # if self.quantification:
            #     _calc_perf = vu.get_performance_baseon_crit_quantification(
            #         self.tt_conc, pred_reshape[best_index], perf_crit,
            #     )
            # if show:
            #     print(
            #         "The selected key and the corresponding performance",
            #         key_reshape[best_index],
            #         performance_reshape[best_index],
            #         _calc_perf,
            #     )
            #     print(
            #         "The actual best performance and best key",
            #         key_reshape[np.argmax(performance_reshape)],
            #         np.max(performance_reshape),
            #     )
        return (
            best_index,
            key_reshape[best_index],
            performance_reshape[best_index],
            pred_reshape[best_index],
        )

    def get_performance_diagonal(self, perf_crit="global_accu", show=False):
        tt_selection_use = ["top_peak", "top_std", "top_mean", "top_diff"]
        val_perf_collect, val_pred_collect = [], []
        for i, s_selection in enumerate(tt_selection_use):
            (
                [ensemble_perf_val, ensemble_perf_tt],
                _,
                [ensemble_pred_val, ensemble_pred_tt],
                stat,
            ) = self._get_perf_stat(
                s_selection, [s_selection], perf_crit=perf_crit, show=False
            )
            val_perf_collect.append(ensemble_perf_val[0])
            val_pred_collect.append(ensemble_pred_val[0])
        if show:
            fig = plt.figure(figsize=(12, 3))
            for i, s_selection in enumerate(tt_selection_use):
                ax = fig.add_subplot(1, len(tt_selection_use), i+1)
                val_perf = [val_perf_collect[i] if perf_crit != "validation_loss" else (val_perf_collect[i])][0]
                sns.heatmap(val_perf, annot=False, cmap="Blues")
        return val_perf_collect, val_pred_collect

    def get_performance_baseon_validation_perf(
        self, perf_crit="global_accuracy", show=True, tds_dir=None, save=False, fix_percent=False,
    ):
        tt_selection_use = ["top_peak", "top_std", "top_mean", "top_diff"]
        label_use = ["Peak", "Std", "Mean", "Diff"]
        val_collect, tt_collect = [], []
        for i, s_selection in enumerate(tt_selection_use):
            (
                [ensemble_perf_val, ensemble_perf_tt],
                [ensemble_perf_val_conc, ensemble_perf_tt_conc], 
                [ensemble_pred_val, ensemble_pred_tt],
                stat,
            ) = self._get_perf_stat(
                s_selection, [s_selection], perf_crit=perf_crit, show=False
            )
            if show:
                print(
                    "================================================================"
                )
            (
                _best_val_index,
                _best_val_key,
                _best_val_perf,
                _best_val_pred,
            ) = self._get_best_perf_index(
                ensemble_perf_val,
                ensemble_pred_val,
                self.val_label,
                [],
                perf_crit,
                show=show,
                use_tt_method=s_selection,
            )
            if fix_percent:
                _best_val_index = len(self.percentage) * len(self.percentage) - 1
            _, _best_tt_key, _best_tt_perf, _best_tt_pred = self._get_best_perf_index(
                ensemble_perf_tt,
                ensemble_pred_tt,
                self.tt_label,
                [_best_val_index],
                perf_crit,
                show=show,
                use_tt_method=s_selection,
            )
            val_collect.append(
                [_best_val_index, _best_val_key, _best_val_perf, _best_val_pred]
            )
            tt_collect.append([_best_tt_key, _best_tt_perf, _best_tt_pred])
        if show:
            if self.detection:
                show_best_detection_performance_on_each_combination(tt_collect, self.tt_label, self.tt_conc, 
                                                                    label_use)
        val_perf_array = np.array([v[2] for v in val_collect])
        if perf_crit == "global_accu" or perf_crit == "f1_score" or perf_crit == "rsquare" or perf_crit == "log_rsquare":
            best_best = np.argmax(val_perf_array)
        elif perf_crit == "mae" or perf_crit == "rmae" or perf_crit == "validation_loss":
            best_best = np.argmin(val_perf_array)
        tt_best_perf = tt_collect[best_best]
        if self.detection:
            best_pred = tt_best_perf[-1]
            pred_label = np.argmax(tt_best_perf[-1], axis=-1)
            perf = np.sum(pred_label == self.tt_label) / len(self.tt_label)
            (
                accu_per_conc,
                std_per_conc,
                xtick_label,
            ) = vu.show_accuracy_over_concentration(
                best_pred, pred_label, self.tt_label, self.tt_conc, show=False
            )
            if show:
                print(
                    "================================================================"
                )
                print("The final best key", val_collect[best_best][:3])
                for i, s_tick in enumerate(xtick_label):
                    print(
                        "concentration: %s, ensemble accuracy %.2f(%.2f)"
                        % (s_tick, accu_per_conc[i], std_per_conc[i])
                    )
                print("Ensemble global accuracy: %.2f" % (perf * 100))
                print(
                    "Ensemble F1 score: %.2f"
                    % (f1_score(self.tt_label, pred_label) * 100)
                )
            return accu_per_conc, std_per_conc, xtick_label, tt_collect, label_use
        if self.quantification:
            return tt_best_perf[-1], self.tt_conc, tt_collect, label_use
        
        
def show_best_detection_performance_on_each_combination(tt_collect_group, tt_label, tt_conc, label_use):
    fig = vu.give_figure_specify_size(0.5)
    ax = fig.add_subplot(111)
    color_use = ['r', 'g', 'b', 'm']
    for i, s_collect in enumerate(tt_collect_group):
        pred = s_collect[-1]
        pred_label = np.argmax(pred, axis=-1)
        perf = np.sum(pred_label == tt_label) / len(tt_label)
        accu_per_conc, std_per_conc, xtick_label = vu.show_accuracy_over_concentration(pred, pred_label, 
                                                                                       tt_label, tt_conc, 
                                                                                       show=False)
        x_axis = np.arange(len(xtick_label) + 1)[1:]
        ax.plot(x_axis, accu_per_conc, color=color_use[i], marker='.', label=label_use[i] + ":%.1f" % (perf * 100))
        # ax.errorbar(x_axis, accu_per_conc, std_per_conc, color=color_use[i], capsize=3, label=label_use[i] + ":%.1f" % (perf * 100))
    # ax.set_xscale("log")
    ax.set_xticks(x_axis)
    ax.set_xticklabels(["%.4f" % v for v in np.unique(tt_conc)], rotation=90)
    ax.set_xlabel("Concentration")
    ax.set_ylabel("Detection accuracy")
    ax.legend(loc='best')
        

def get_calibration_curve(
    tt_label, tt_pred, x_axis="uncertainty", class_index=1, num_bins=10
):
    """Get the calibration curve
    tt_label: [num_samples]
    tt_pred: [num_samples, 2]
    class_index: which class am I going to calibration
    """
    if np.max(tt_pred) != 1:
        tt_pred = softmax(tt_pred, axis=-1)
    tt_pred_label = np.argmax(tt_pred, axis=-1)
    if x_axis == "entropy":
        tt_uncert = -np.sum(tt_pred * np.log(tt_pred + 1e-10), axis=-1)
        correct_or_wrong = (tt_label == tt_pred_label).astype(np.int32)
        bin_interval = np.linspace(np.min(tt_uncert), np.max(tt_uncert), num_bins)
    elif x_axis == "probability":
        tt_uncert = tt_pred[:, class_index]
        correct_or_wrong = (tt_label == class_index).astype(np.int32)
        # correct_or_wrong = np.logical_and(tt_label == class_index, tt_pred_label == class_index).astype(np.int32)
        bin_interval = np.linspace(0, 1, num_bins)
    elif x_axis == "nll":
        tt_uncert = -np.log(tt_pred[:, class_index] + 1e-10)
        correct_or_wrong = (tt_label == class_index).astype(np.int32)
        bin_interval = np.linspace(np.min(tt_uncert), np.max(tt_uncert), num_bins)
    accu_group = []
    x_axis_group = []
    tot = 0.0
    for i, s_value in enumerate(bin_interval[:-1]):
        index = np.logical_and(tt_uncert >= s_value, tt_uncert < bin_interval[i + 1])
        accu_group.append(np.sum(correct_or_wrong[index == 1]) / np.sum(index))
        x_axis_group.append(np.mean(bin_interval[i : i + 2]))
        tot += np.sum(index)
    if x_axis == "nll":
        x_axis_group = np.exp(-np.array(x_axis_group))
    return accu_group, x_axis_group, tt_uncert


class GetPerformanceSIMUDataset(object):
    def __init__(
        self,
        dataset,
        detection,
        quantification,
        lr=0.2,
        patch_size=2,
        concentration_float=0,
        version_use=[],
        save=False,
    ):
        super(GetPerformanceSIMUDataset).__init__()
        exp_dir = "../exp_data/VIT/%s/detection_%s_quantification_%s/" % (
            dataset,
            detection,
            quantification,
        )
        file2load = exp_dir + "stat_patch_%d_lr_%.3f.obj" % (patch_size, lr)
        stat_g = pickle.load(open(file2load, "rb"))
        self.tt_label = stat_g["label"]
        tt_conc = stat_g["concentration"]
        tt_conc_update = tt_conc.copy()
        tt_conc_update[tt_conc_update == 0.0] = concentration_float
        self.tt_conc_update = tt_conc_update
        self.tt_conc = tt_conc
        keys = list(stat_g.keys())
        if len(version_use) != 0:
            self.stat_group = [stat_g["version_%d" % k] for k in version_use]
            self.version_group = ["version_%d" % k for k in version_use]
        else:
            self.stat_group = [stat_g[k] for k in keys if "version_" in k]
            self.version_group = [k for k in keys if "version_" in k]
        print([v.split("version_")[1] for v in self.version_group])
        self.color = ["r", "g", "b", "c", "m", "orange"]
        self.dataset = dataset
        self.concentration_float=concentration_float

    def get_detection_results(self, show=False):
        perf_group = {}
        perf_keys = ["accu_per_conc", "f1_score", "ga"]
        for s_k in perf_keys:
            perf_group[s_k] = []
        num_measurement = len(self.stat_group)
        for i, s_pred in enumerate(self.stat_group):
            _pred_label = np.argmax(np.array(s_pred), axis=-1)[:, 0]
            accu_per_conc, _, xtick_label = vu.show_accuracy_over_concentration(
                s_pred, _pred_label, self.tt_label, self.tt_conc, show=False
            )
            _f1_score = f1_score(self.tt_label, _pred_label)
            _avg_accu = np.sum(self.tt_label == _pred_label) / len(self.tt_label)
            print(self.version_group[i], _avg_accu)
            for s_value, s_k in zip([accu_per_conc, _f1_score, _avg_accu], perf_keys):
                perf_group[s_k].append(s_value)

        for s_k in perf_keys:
            avg_value, std_value = np.mean(perf_group[s_k], axis=0), np.std(
                perf_group[s_k], axis=0
            ) * 1.95 / np.sqrt(num_measurement)
            perf_group[s_k + "_average"] = [avg_value, std_value]

        ensemble_prediction = np.mean(self.stat_group, axis=0)[:, 0]
        ensemble_pred_label = np.argmax(ensemble_prediction, axis=-1)
        accu_per_conc, std_per_conc, _ = vu.show_accuracy_over_concentration(
            ensemble_prediction,
            ensemble_pred_label,
            self.tt_label,
            self.tt_conc,
            show=False,
        )
        if show:
            ensemble_f1_score, ensemble_ga = f1_score(
                self.tt_label, ensemble_pred_label
            ), np.sum(self.tt_label == ensemble_pred_label) / len(ensemble_pred_label)
            print(
                "===================================================================="
            )
            for i, s_tick in enumerate(xtick_label):
                print(
                    "concentration: %s, average accuracy: %.2f ± %.2f, ensemble accuracy %.2f ± %.2f"
                    % (
                        s_tick,
                        perf_group[perf_keys[0] + "_average"][0][i] * 100,
                        perf_group[perf_keys[0] + "_average"][1][i] * 100,
                        accu_per_conc[i] * 100,
                        std_per_conc[i],
                    )
                )

            print(
                "Average F1 score: %.2f ± %.2f, Ensemble F1 score %.2f"
                % (
                    perf_group[perf_keys[1] + "_average"][0] * 100,
                    perf_group[perf_keys[1] + "_average"][1] * 100,
                    ensemble_f1_score * 100,
                )
            )
            print(
                "Average global accuracy: %.2f ± %.2f, Ensemble global accuracy %.2f"
                % (
                    perf_group[perf_keys[2] + "_average"][0] * 100,
                    perf_group[perf_keys[2] + "_average"][1] * 100,
                    ensemble_ga * 100,
                )
            )
            fig = plt.figure(figsize=(6, 3))
            ax = fig.add_subplot(111)
            ax.errorbar(
                np.arange(len(xtick_label) + 1)[1:],
                perf_group[perf_keys[0] + "_average"][0],
                perf_group[perf_keys[0] + "_average"][1],
                color="m",
                marker="o",
                capsize=3,
            )
            ax.grid(ls=":")
            for i, s_stat in enumerate(perf_group[perf_keys[0]]):
                ax.plot(
                    np.arange(len(xtick_label) + 1)[1:],
                    s_stat,
                    color=self.color[i],
                    marker=".",
                    ls="",
                )
            ax.set_xscale("log")
            ax.set_xticks(np.arange(len(xtick_label) + 1)[1:])
            ax.set_xticklabels(xtick_label)
            ax.set_xlabel("Concentration (uM)")
            ax.set_ylabel("Detection accuracy")
        return accu_per_conc, std_per_conc, np.sum(self.tt_label == ensemble_pred_label) / len(ensemble_pred_label), f1_score(self.tt_label, ensemble_pred_label)

    def get_detection_calibration_curve(self, criteria, class_index):
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)
        if self.dataset == "SIMU_TYPE_4":
            if criteria == "entropy":
                num_bins = 8
            elif criteria == "nll":
                num_bins = 7
            elif criteria == "probability":
                num_bins = 11
        ensemble_accu, x_axis, _ = get_calibration_curve(
            self.tt_label,
            np.mean(self.stat_group, axis=0)[:, 0],
            criteria,
            class_index,
            num_bins=num_bins,
        )
        ax.plot(x_axis, ensemble_accu, marker=".", label="%d" % num_bins)
        ax.grid(ls=":")
        if criteria == "probability":
            ax.plot([0, 1], [0, 1], ls=":")
        if criteria == "entropy":
            ax.set_xlabel("Entropy")
            ax.set_ylabel("Prediction accuracy")
        else:
            ax.set_xlabel("Predicted probability")
            ax.set_ylabel("True probability")

        fig = plt.figure(figsize=(5, 3))
        ax = fig.add_subplot(111)
        avg_logit = np.mean(self.stat_group, axis=0)[:, 0]
        avg_prob = softmax(avg_logit, axis=-1)
        ax.hist(avg_prob)

    def get_quantification_results(self, show=True):
        avg_pred, rsquare_group = [], []
        for i, s_pred in enumerate(self.stat_group):
            s_pred[s_pred < 0] = 0
            _rsquare = utils.calc_rsquare(self.tt_conc, s_pred)
            if self.concentration_float != 0:
                print(self.version_group[i], _rsquare, utils.calc_rsquare(np.log(self.tt_conc_update), np.log(s_pred)))
            else:
                print(self.version_group[i], _rsquare)
            avg_pred.append(s_pred)
            rsquare_group.append(_rsquare)
        print("=======================================================")
        print(
            "The average Rsquare: %.2f +- %.2f"
            % (
                np.mean(rsquare_group) * 100,
                1.95 * np.std(rsquare_group) / np.sqrt(len(rsquare_group)) * 100,
            )
        )
        ensemble_perf = np.mean(avg_pred, axis=0)
        print(
            "Ensemble Rsquare: %.2f"
            % (utils.calc_rsquare(self.tt_conc, ensemble_perf) * 100)
        )

        x_tick, quan_stat = utils.get_rsquare_curve_stat(self.tt_conc, ensemble_perf)
        if show:
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111)
            x_axis = np.arange(len(np.unique(self.tt_conc)) + 1)[1:]
            ax.errorbar(
                x_axis,
                quan_stat[:, 0],
                quan_stat[:, 1],
                capsize=3,
                color="m",
                marker="o",
            )
            ax.plot(x_axis[1:], x_tick[1:], ls=":", color="g")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(ls=":")
            ax.set_xticks(x_axis)
            ax.set_xticklabels(["%.4f" % v for v in x_tick], rotation=90)

            ax.set_xlabel("True concentration")
            ax.set_ylabel("Predicted concentration")
        else:
            return ensemble_perf, self.tt_conc


def get_concentration_prediction_summary_results(
    tds_dir="../rs_dataset/paper_figure/", save=False
):
    data_group = ["SIMU_TYPE_12", "SIMU_TYPE_13", "SIMU_TYPE_14"]
    title_group = ["Type 1", "Type 2", "Type 3"]
    perf_group = {}
    for i, s_data in enumerate(data_group):
        obj = GetPerformanceSIMUDataset(s_data, False, True, 0.08, 2, 10)
        ensemble_quan_pred, tt_conc = obj.get_quantification_results(show=False)
        perf_group[s_data] = ensemble_quan_pred

    fig = vu.give_figure_specify_size(1.3, 0.5)
    ax_global = vu.ax_global_get(fig)
    ax_global.set_xticks([])
    ax_global.set_yticks([])
    ax_group = [
        fig.add_subplot(1, len(data_group), i + 1) for i in range(len(data_group))
    ]
    x_axis = np.arange(len(np.unique(tt_conc)) + 1)[1:]
    color_use = ["r", "g", "b", "m"]
    for i, s_data in enumerate(data_group):
        x_tick, quan_stat = utils.get_rsquare_curve_stat(tt_conc, perf_group[s_data])
        rsquare = utils.calc_rsquare(tt_conc, perf_group[s_data])
        ax_group[i].errorbar(
            x_axis,
            quan_stat[:, 0],
            quan_stat[:, 1],
            color=color_use[0],
            marker=".",
            capsize=3,
            label="ViT: %.2f" % (rsquare * 100),
        )
    for i, s_ax in enumerate(ax_group):
        s_ax.grid(ls=":")
        s_ax.plot(x_axis[1:], x_tick[1:], ls=":", color="gray")
        for j, s_conc in enumerate(np.unique(tt_conc)):
            loc = np.where(tt_conc == s_conc)[0]
            s_ax.plot(
                [x_axis[j] for _ in range(len(loc))],
                perf_group[data_group[i]][loc],
                marker=".",
                color="gray",
                ls="",
                markersize=4,
            )
        s_ax.legend(loc="best")
        s_ax.set_xscale("log")
        s_ax.set_yscale("log")
        s_ax.set_xticks(x_axis)
        s_ax.set_xticklabels(["%.4f" % v for v in x_tick], rotation=90)
        s_ax.set_title(title_group[i])
    ax_global.set_ylabel("Predicted concentration\n\n\n")
    ax_global.set_xlabel("\n\n\n\nTrue concentration")
    if save:
        plt.savefig(
            tds_dir + "/quantification_simulate_data_performance.pdf",
            pad_inches=0,
            bbox_inches="tight",
        )
        
        
def produce_sbr_concentration_relation(tds_dir="../rs_dataset/paper_figure/", save=False):
    concentration  = np.array([0, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1] + list(np.linspace(0.2, 0.5, 4)))
    sbr = 1 / 100 * (np.exp(concentration) - 1) 
    fig = vu.give_figure_specify_size(0.5)
    ax = fig.add_subplot(111)
    ax.plot(concentration, sbr, 'r')
    ax.set_xscale("symlog", linthresh=concentration[1])
    ax.set_xticks(concentration)
    ax.set_xticklabels(["%.4f" % v for v in concentration], rotation=90)
    ax.set_xlabel("Concentration")
    ax.set_ylabel("SBR")
    if save:
        plt.savefig(tds_dir + "/relation_between_concentration_and_sbr.pdf", 
                    pad_inches=0, bbox_inches='tight')


if __name__ == '__main__':
    args = give_args()
    save = args.save
    pdf_pgf = args.pdf_pgf
    tds_dir = args.dir2save
    if not os.path.exists(tds_dir):
        os.makedirs(tds_dir)
    if args.index == "spectra_heatmap_detection":
        get_spectra_performance_different_percentage(
            detection=True,
            quantification=False,
            version_group=[20], #, 21, 22, 23, 24],
            perf_crit="validation_loss",
            save=save,
            tds_dir=tds_dir,
        )
    elif args.index == "spectra_heatmap_quantification":
        get_spectra_performance_different_percentage(
            detection=False,
            quantification=True, 
            version_group=[20, 21, 22, 23, 24],
            perf_crit="log_rsquare",
            save=save,
            tds_dir=tds_dir
        )
    elif args.index == "spectra_test_detection_each_selection_method":
        get_detection_performance_different_selection_methods(perf_crit="validation_loss", tds_dir=tds_dir, 
                                                              save=save, fix_percent=False)
    elif args.index == "compare_detection":
        get_boxplot_detection_performance_different_approaches(perf_crit="validation_loss", tds_dir=tds_dir,
                                                               save=save)
    elif args.index == "compare_quantification":
        get_boxplot_quantification_performance_different_approaches(perf_crit="rsquare", tds_dir=tds_dir, 
                                                                    save=save)
    elif args.index == "example_sersmap":
        get_example_sers_maps_and_spectra_figure(tds_dir=tds_dir, save=save)
        combine_sers_maps_example_map(tds_dir=tds_dir, save=save)
    elif args.index == "decompose_detection_accuracy":
        check_detection_accuracy_at_each_concentration(perf_crit="validation_loss", tds_dir=tds_dir,
                                                       save=save)
    elif args.index == "sers_map_and_attention_map":
        prepare_attention_map_example(quantile_percent=0.8, tds_dir=tds_dir, save=save)
    elif args.index == "save_map_for_figure":
        produce_map_for_architecture(tds_dir, save)
    elif args.index == "relation_between_sbr_and_concentration":
        produce_sbr_concentration_relation(tds_dir, save=save)
    elif args.index == "relation_between_peak_intensity_and_concentration":
        get_concentration_and_peak_intensity_multiple_datasets(save=save, tds_dir=tds_dir)
"""
Created on 12:48 at 29/11/2021
@author: bo
"""
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
from einops.layers.torch import Rearrange
import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns
from scipy.special import softmax
import data.prepare_sers_data as psd
import pickle


def calc_rsquare(gt_conc, pred_conc):
    top = np.sum((gt_conc - pred_conc)**2)
    bottom = np.sum((gt_conc - np.mean(gt_conc))**2)
    r_square = 1.0 - top / bottom
    # print("Rsquare", r_square)
    return r_square


def calc_rsquare_log(gt_conc, pred_conc):
    if np.min(gt_conc) == 0:
        gt_conc_update = gt_conc.copy()
        gt_conc_update[gt_conc_update == 0.0] = 1e-6 
    else:
        gt_conc_update = gt_conc 
    return calc_rsquare(np.log(gt_conc_update), np.log(pred_conc))


def calc_rae(gt_quan, pred_quan):
    out = np.sqrt(np.sum((gt_quan - pred_quan) ** 2 / gt_quan ** 2))
    return out 

def calc_rmae(gt_quan, pred_quan):
    out = np.sum(abs(gt_quan - pred_quan) / gt_quan)
    return out 


def calc_gap(tt_label, tt_pred):
    pred_label = np.argmax(tt_pred, axis=-1)
    return np.sum(pred_label == tt_label) / len(tt_label)


def calc_global_accuracy(pred, tt_label):
    if len(np.shape(pred)) == 2:
        pred_label = np.argmax(pred, axis=-1)
    elif len(np.shape(pred)) == 3:
        pred_label = np.argmax(pred, axis=-1)[:, 0]
    return np.sum(pred_label == tt_label) / len(tt_label)


def get_quantification_performance(gt_quan, pred_quan, print_info=True, concentration_float=1e-6):
    """This function gives the quantification performance 
    Args:
    gt_quan: [num_test]
    pred_quan: [num_test]
    """
    # rsquare 
    rsquare = calc_rsquare(gt_quan, pred_quan)
    if np.min(gt_quan) == 0:
        gt_quan_log = gt_quan.copy()
        gt_quan_log[gt_quan_log == 0] = concentration_float
    else:
        gt_quan_log = gt_quan    
    rsquare_subset = calc_rsquare(np.log(gt_quan_log), np.log(pred_quan))
    
    #relative absolute error 
    # rae = calc_rae(gt_quan, pred_quan)    
    #relative mean absolute error
    # rmae = calc_rmae(gt_quan, pred_quan)
    # rae = calc_rae(np.log(gt_quan_log), np.log(pred_quan))
    # rmae = calc_rmae(np.log(gt_quan_log), np.log(pred_quan))
    rae, rmae = 0, 0
    # if len(gt_quan) > 1:
    #     rae_subset = calc_rae(gt_quan[1:], pred_quan[1:])
    #     rmae_subset = calc_rmae(gt_quan[1:], pred_quan[1:])
    # else:
    rae_subset = rae
    rmae_subset = rmae
    
    mae = np.sum(abs(np.log(gt_quan_log) - np.log(pred_quan)))
    mse = np.sum((np.log(gt_quan_log) - np.log(pred_quan))**2)
    # mae = np.sum(abs(gt_quan - pred_quan))
    # mse = np.sum((gt_quan - pred_quan)**2)
    if print_info:
        print("================================================")
        print("R-square: %.2f" % (rsquare * 100), "Subset R-square: %.2f" % (rsquare_subset * 100))
        print("Relative absolute error: %.5f" % (rae), "Relative subset absolute error: %.5f" % (rae_subset))
        print("Relative mae: %.5f" % (rmae), "Relative subset mae: %.5f" % rmae_subset)
        print("mae: %.4f" % mae, " mse: %.4f" % mse)
    return [rsquare, rsquare_subset], [rae, rae_subset], [rmae, rmae_subset], [mae, mse]
        
        
def get_rsquare_curve_stat(tt_conc, quan_exp):
    stat = np.zeros([len(np.unique(tt_conc)), 2])
    for i, s_conc in enumerate(np.unique(tt_conc)):
        index = np.where(tt_conc == s_conc)[0]
        pred_value = quan_exp[index] #* 1000
        pred_value_log = np.log(pred_value)
        avg, std = np.mean(pred_value_log), np.std(pred_value_log) * 1.95 / np.sqrt(len(index))
        stat[i] = [avg, std]
    return np.unique(tt_conc), stat


def get_cosine_similarity_position_encoding(pos_encoding, im_shape, patch_shape, metric="cosine_similarity", show=False,
                                            tds_dir=None, save=False):
    im_h, im_w = im_shape
    patch_h, patch_w = patch_shape
    num_pat_h, num_pat_w = im_h // patch_h, im_w // patch_w
    print("num patches on the first dimension", num_pat_h)
    print("num patches on the second dimension", num_pat_w)

    if metric == "cosine_similarity":
        pos_encoding = pos_encoding[1:] / np.sqrt(np.sum(pos_encoding[1:] ** 2, axis=-1, keepdims=True))
    elif metric == "euclidean_distance":
        pos_encoding = pos_encoding[1:]
    pos_encoding = np.reshape(pos_encoding, [num_pat_h, num_pat_w, -1])
    patches = np.zeros([num_pat_h, num_pat_w, num_pat_h, num_pat_w])
    for pos_i in range(num_pat_h):
        for pos_j in range(num_pat_w):
            target_patch = pos_encoding[pos_i, pos_j]
            cos_sim_ij = np.zeros([num_pat_h, num_pat_w])
            for i in range(num_pat_h):
                for j in range(num_pat_w):
                    if metric == "euclidean_distance":
                        cos_sim_ij[i, j] = np.sum((target_patch - pos_encoding[i, j])**2)
                    elif metric == "cosine_similarity":
                        cos_sim_ij[i, j] = np.sum(target_patch * pos_encoding[i, j])
            patches[pos_i, pos_j] = cos_sim_ij

    if show:
        nrow = im_shape[0] // patch_h
        ncol = im_shape[1] // patch_w
        fig, axes = plt.subplots(figsize=(12, 10), nrows=nrow, ncols=ncol)
        for i in range(nrow):
            for j in range(ncol):
                im = axes[i, j].imshow(patches[i, j])
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        fig.colorbar(im, ax=axes.ravel().tolist())
        if save:
            plt.savefig(tds_dir + "/positional_encoding.pdf", pad_inches=0, bbox_inches='tight')
    return patches


def create_canvas(feature):
    """This function is used to create canvas
    Args:
        feature [out_channel, in_channel, kh, kw]
    """
    nx, ny, fh, fw = np.shape(feature)
    x_values = np.linspace(-3, 3, nx)
    y_values = np.linspace(-3, 3, ny)
    canvas = np.empty((fw * nx, fh * ny))
    for i, yi in enumerate(x_values):
        f_sub = feature[i]
        for j, xj in enumerate(y_values):
            f_use = f_sub[j]
            canvas[(nx - i - 1) * fh:(nx - i) * fh,
            j * fw:(j + 1) * fw] = f_use
    return canvas


def attention_rollout(attn_weights_list, apply_rollout=True):
    """attn_weights_list: [num_layers, num_seq, num_seq]"""
    eye = np.eye(attn_weights_list[0].shape[-1])
    if apply_rollout:
        attn_weights_rollout = [0.5 * v + 0.5 * eye for v in attn_weights_list]
        attn_weights_rollout = [v / np.sum(v, axis=-1, keepdims=True) for v in attn_weights_rollout]
    else:
        attn_weights_rollout = attn_weights_list
    joint_attentions = np.zeros(np.shape(attn_weights_rollout))
    joint_attentions[0] = attn_weights_rollout[0]
    for i, s_att in enumerate(attn_weights_rollout[1:]):
        a_tilde = np.matmul(s_att, joint_attentions[i])
        joint_attentions[i+1] = a_tilde
    return joint_attentions


def show_im_patch(im_new_patch, nrow, ncol, figsize, save=False, tds_dir=None):
    fig, axes = plt.subplots(figsize=figsize, nrows=nrow, ncols=ncol)
    for i in range(nrow):
        for j in range(ncol):
            axes[i, j].imshow(im_new_patch[i*ncol+j])
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    if save:
        plt.savefig(tds_dir + "/image_patch_example.jpg", pad_inches=0, bbox_inches='tight', dpi=500)
        

def show_attention_maps(s_npy, attention_map_imsize, concentration, wavecut, select_wave=0, tds_dir=None, save=False,
                        save_name=None):
    """Args:
    s_npy: [num_concentration, num_sers_maps, imh, imw]
    attention_map_imsize: [num_concentration, num_sers_maps, imh, imw]
    concentration: [num_concentration]
    wavecut: [num_sers_maps]
    select_wave: the selected wavenumbers
    tds_dir: str
    save: bool
    """
    ncol = 5
    nrow = len(s_npy) // ncol
    if nrow == 0:
        fig = plt.figure(figsize=(10, 4))
        for i, s_peak in enumerate(select_wave):
            ax = fig.add_subplot(1, len(select_wave)+1, i+1)
            ax.imshow(np.array(s_npy)[0, int(s_peak), :, :])
            ax.set_title("Wavenumber:%d" % (int(s_peak)))
        ax = fig.add_subplot(1, len(select_wave)+1, len(select_wave) + 1)
        l = ax.imshow(attention_map_imsize[-1, 0])
        ax.set_title("Attention")
        if save:
            plt.savefig(tds_dir + "/%s.pdf" % save_name, pad_inches=0, bbox_inches='tight')
    else:
        title_use = ["input", "attention"]
        fig = plt.figure(figsize=(10, 8))
        for i in range(nrow):
            for j in range(ncol):
                _index = i * ncol*2 + j + 1
                for q, _im in enumerate([s_npy[i*ncol+j], attention_map_imsize[i*ncol+j]]):
                    ax = fig.add_subplot(nrow, ncol * 2, i * ncol*2 + j * 2 + q + 1)
                    if select_wave[i * ncol + j] != 0 and q == 0:
                        _start_index = np.where(wavecut >= select_wave[i * ncol + j])[0][0]
                        _im_use = _im[_start_index]
                    else:
                        _im_use = np.mean(_im, axis=0)
                        if len(np.shape(_im_use)) == 3 and np.shape(_im_use)[-1] != 3:
                            _im_use = np.mean(_im_use, axis=0)
                    ax.imshow(_im_use)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if j == 0 and q == 0:
                        if np.sum(concentration) % 1 != 0:
                            ax.set_ylabel("%.3f" % concentration[i*ncol+j])
                    if i == 0:
                        ax.set_title(title_use[q], fontsize=8)
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        if save:
            if np.sum(select_wave) != 0:
                plt.savefig(tds_dir + "/attention_maps_on_all_concentrations_at_wave_%s.pdf" % save_name,
                            pad_inches=0, bbox_inches='tight')
            else:
                plt.savefig(tds_dir + "/attention_maps_on_all_concentrations_%s.pdf" % save_name,
                            pad_inches=0, bbox_inches='tight')


def reshape_mask_to_imsize(patch_size, mask):
    nrow = int(np.sqrt(len(mask)))
    ncol = int(np.sqrt(len(mask)))
    new_mask = np.zeros([nrow * patch_size, ncol * patch_size])
    for i in range(nrow):
        for j in range(ncol):
            new_mask[i * patch_size:(i+1)*patch_size,
            j*patch_size:(j+1)*patch_size] = np.zeros([patch_size, patch_size]) + mask[i * ncol + j]
    new_mask = new_mask - np.min(new_mask)
    new_mask = new_mask / (np.max(new_mask) + 1e-10)
    return new_mask


def reshape_mask_sers(mask, patch_size, imsize):
    nrow = imsize[0] // patch_size[0]
    ncol = imsize[1] // patch_size[1]
    new_mask = np.zeros(imsize)
    for i in range(nrow):
        for j in range(ncol):
            new_mask[i * patch_size[0]:(i+1)*patch_size[0],
                j*patch_size[1]:(j+1)*patch_size[1]] = np.zeros(patch_size) + mask[i * ncol + j]
    # new_mask = new_mask - np.min(new_mask)
    # new_mask = new_mask / (np.max(new_mask) + 1e-10)
    return new_mask


def im_to_patch(im, patch_height, patch_width):
    func = Rearrange('b c (h p1) (w p2) -> b (h w) p1 p2 c', p1=patch_height, p2=patch_width)
    im_patch = func(im)
    return im_patch.detach().cpu().numpy()[0]


def overlap_attention_cls_token_sers_multi(attn_maps, target_patch_index, patch_size, imsize):
    """Args:
    attn_maps: [num_wavenumbers, num_layers, num_heads, num_patch + 1, num_patch + 1]
    target_patch_index: int, which patch to look at
    patch_size: int, [5, 5]
    imsize: int, [45, 75]
    """
    attn_maps_all_heads_token = np.array(attn_maps)
    attn_maps_rollout = []
    for v in attn_maps_all_heads_token:
        _rollout = np.expand_dims(attention_rollout(np.mean(v, axis=1), True), axis=1)[:, :, target_patch_index, :]
        attn_maps_rollout.append(_rollout)
    attn_maps_all_heads_token = np.concatenate([attn_maps_all_heads_token[:, :, :, target_patch_index],
                                                np.array(attn_maps_rollout)], axis=2) # [num_wave, num_layer, num_head+1]
    attn_maps_imsize = []
    for i, s_im in enumerate(attn_maps_all_heads_token):
        _maps = []
        for j in range(len(s_im)):  # layer index
            _mask = reshape_mask_sers(s_im[j, -1, 1:], patch_size, imsize)
            _maps.append(_mask)
        attn_maps_imsize.append(_maps)
    return attn_maps_all_heads_token, np.array(attn_maps_imsize)


def apply_mask_on_im(im, mask, show=False):
    im = im / np.max(im)
    if len(np.shape(im)) == 2:
        im = np.expand_dims(im, axis=-1)
    if np.shape(im)[-1] == 1:
        im = np.repeat(im, 3, axis=-1)
    # im_wt_mask = np.concatenate([im, np.expand_dims(mask, axis=-1)], axis=2)  # [imh, imw, 4]
    im_wt_mask = np.repeat(np.expand_dims(mask, axis=-1), 3, axis=-1)
    im_orig = im
    # im_orig = np.concatenate([im, np.ones([im.shape[0], im.shape[1], 1])], axis=2)
    im_tot = np.concatenate([im_orig, im_wt_mask], axis=1)
    if show:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)
        ax.imshow(im_tot)
    return im_tot


def save_plotly_data(maps_group, conc_group, attn_maps_imsize_group, peak_group, tr_wave,
                     target_shape,
                     plotly_dir, save_name):
#     plotly_dir = "../exp_data/VIT/TOMAS/detection_%s_quantification_%s/plotly_data/" % (detection, quantification)
    if not os.path.exists(plotly_dir):
        os.makedirs(plotly_dir)
    data_obj = {}
    data_obj["sers_maps"] = []
    data_obj["wavenumber"] = []
    data_obj["imshape"] = []
    data_obj["sample"] = []
    data_obj["concentration"] = []
    data_obj["peak"] = []
    data_obj["attention_map"] = []
    imh, imw = np.shape(maps_group)[1:3]
    for i, s_conc in enumerate(np.unique(conc_group)):
        index = np.where(conc_group == s_conc)[0]
        s_map = maps_group[index]
        if len(attn_maps_imsize_group) > 0:
            s_attention = attn_maps_imsize_group[index, -1, 0]
        else:
            s_attention = np.ones([len(index), imh, imw])
        s_conc = conc_group[index]
        s_peak = [tr_wave[q] for q in peak_group[index].astype(np.int32)]
        data_obj["sers_maps"].append(s_map)
        data_obj["wavenumber"].append([tr_wave for _ in s_map])
        data_obj["concentration"].append(s_conc)
        data_obj["imshape"].append([target_shape for _ in s_map])
        data_obj["sample"].append(np.arange(len(s_map)))
        data_obj["peak"].append(s_peak)
        data_obj["attention_map"].append(s_attention)
    for k in data_obj.keys():
        data_obj[k] = [v for q in data_obj[k] for v in q]
    with open(plotly_dir + "/%s.obj" % save_name, "wb") as f:
        pickle.dump(data_obj, f)        
            





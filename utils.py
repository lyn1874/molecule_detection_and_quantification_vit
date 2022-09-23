"""
Created on 12:48 at 29/11/2021
@author: bo
"""
import matplotlib.pyplot as plt
import numpy as np
from einops.layers.torch import Rearrange


def calc_rsquare(gt_conc, pred_conc):
    top = np.sum((gt_conc - pred_conc)**2)
    bottom = np.sum((gt_conc - np.mean(gt_conc))**2)
    r_square = 1.0 - top / bottom
    return r_square


def calc_rsquare_log(gt_conc, pred_conc):
    if np.min(gt_conc) == 0:
        gt_conc_update = gt_conc.copy()
        gt_conc_update[gt_conc_update == 0.0] = 1e-6 
    else:
        gt_conc_update = gt_conc 
    return calc_rsquare(np.log(gt_conc_update), np.log(pred_conc))


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
    im_wt_mask = np.repeat(np.expand_dims(mask, axis=-1), 3, axis=-1)
    im_orig = im
    im_tot = np.concatenate([im_orig, im_wt_mask], axis=1)
    if show:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)
        ax.imshow(im_tot)
    return im_tot

        
            





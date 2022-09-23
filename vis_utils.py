#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   vis_utils.py
@Time    :   2022/04/05 08:20:53
@Author  :   Bo 
'''
from audioop import cross
import numpy as np 
        
        
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
    return accu_per_conc, std_per_conc, xticklabels
    
    
def ax_global_get(fig):
    ax_global = fig.add_subplot(111, frameon=False)
    ax_global.spines['top'].set_color('none')
    ax_global.spines['bottom'].set_color('none')
    ax_global.spines['left'].set_color('none')
    ax_global.spines['right'].set_color('none')
    ax_global.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')

    return ax_global
        
        
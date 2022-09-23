#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   ml_method.py
@Time    :   2022/08/18 16:13:59
@Author  :   Bo 
'''
from dataclasses import dataclass
import numpy as np 
import os 
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVR, SVC
import data.prepare_sers_data as psd 
import utils
import torch
import pickle
import matplotlib


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


def compare_regression_baseline(dataset, 
                                path_mom="../rs_dataset/",
                                tds_dir="../exp_data/ml_methods/"):
    method = ["random_forest", "decision_tree", "svm", "gradient_boost"]
    ml_obj = MLMethod(dataset, False, True, path_mom=path_mom,
                      tds_dir=tds_dir)
    out = ml_obj.model(method, "top_peak", save=True)
    
    
def compare_detection_baseline(dataset, 
                               path_mom="../rs_dataset/",
                               tds_dir="../exp_data/ml_methods/"):
    method = ["knn", "random_forest", "svm", "gradient_boost"]
    ml_obj = MLMethod(dataset, True, False, 
                      path_mom=path_mom, 
                      tds_dir=tds_dir)
    out = ml_obj.model(method, "top_peak", save=True)
    
    
def get_concentration_perf(dataset, path2load="../exp_data/"):
    file = path2load + "/ml_methods/%s_concentration_prediction.obj" % dataset 
    ml_quantify = pickle.load(open(file, "rb"))
    perf = [utils.calc_rsquare(ml_quantify[-3], v) for v in ml_quantify[-5]]
    method = list(ml_quantify[0][0].keys())
    return ml_quantify[-5], ml_quantify[-3], method, perf
        

def get_detection_perf(dataset, path2load="../exp_data/"):
    file = path2load + "/ml_methods/%s_detection.obj" % dataset 
    ml_detect = pickle.load(open(file, "rb"))
    method = list(ml_detect[0][0].keys())
    pred_prob = ml_detect[-1]
    pred_label = ml_detect[-5]
    gt_label = ml_detect[-2]
    conc = ml_detect[-3]    
    return pred_prob, pred_label, gt_label, conc, method


class MLMethod(object):
    def __init__(self, dataset, detection, quantification, path_mom="../rs_dataset/", 
                 tds_dir="../exp_data/ml_methods/"):
        self.dataset = dataset  
        self.path_mom = path_mom 
        if self.dataset == "TOMAS":
            targ_h = 56 
        elif self.dataset == "DNP":
            targ_h = 44 
        elif self.dataset == "PA":
            targ_h = 40 
        self.num_max_measurements = 30 if self.dataset != "PA" else 25 
        if quantification:
            self.num_max_measurements -= 5
        concentration_float = 1e-6 if self.dataset != "PA" else 1e-5 
        self.target_shape = [targ_h, targ_h]
        self.concentration_float = concentration_float
        self.detection = detection 
        self.quantification = quantification
        # self.normalization = "max" if self.detection == True and self.dataset != "PA" else "none"
        self.quantification_loss = "mse" if self.detection == False else "none"
        self.percentage = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
        self.tds_dir = tds_dir
        if not os.path.exists(self.tds_dir):
            os.makedirs(self.tds_dir)
    
    def _load_dataset(self, leave_index):
        data_obj = psd.ReadSERSData(self.dataset, self.target_shape, 
                                    percentage=0, top_selection_method="sers_maps",
                                    avg=False, concentration_float=self.concentration_float, 
                                    quantification=self.quantification,
                                    detection=self.detection, 
                                    leave_index=leave_index, 
                                    leave_method="leave_one_chip", 
                                    path_mom=self.path_mom)

        [tr_maps, tr_label, tr_conc, tr_peak, tr_wave], \
            [val_maps, val_label, val_conc, val_peak], \
                [tt_maps, tt_label, tt_conc, tt_peak, tt_wave], \
                    imshape, num_class = data_obj.forward_test()
        tr_conc_update = self._update_concentration(tr_conc)
        val_conc_update = self._update_concentration(val_conc)
        tt_conc_update = self._update_concentration(tt_conc)
        return [tr_maps, tr_label, tr_conc, tr_conc_update, tr_peak], \
                [val_maps, val_label, val_conc, val_conc_update, val_peak], \
                    [tt_maps, tt_label, tt_conc, tt_conc_update, tt_peak], tr_wave
                
    def _update_concentration(self, conc):
        conc_update = conc.copy()
        conc_update[conc_update == 0.0] = self.concentration_float
        return np.log(conc_update)
        
    def _extract_spectra(self, perc, method, sers_maps, wavenumber):
        spectra = psd.simple_selection(sers_maps, wavenumber, perc, method, self.dataset)[0]
        return spectra 
    
    def model(self, model_method, selection_method="top_peak", save=False):
        perf_val = {}
        perf_tt = {}
        val_stat = {}
        perf_tt_proba = {}
        perf_val_proba = {}
        for s in model_method:
            perf_val[s] = []
            perf_tt[s] = []
            val_stat[s] = []
            perf_tt_proba[s] = []
            perf_val_proba[s] = []
        tt_conc, tt_label = [], []
        val_conc, val_label = [], []
        for s_index in range(self.num_max_measurements):
            tr_data, val_data, tt_data, tr_wave = self._load_dataset(s_index)   
            tt_conc.append(tt_data[-2])
            tt_label.append(tt_data[1])
            val_conc.append(val_data[-2])
            val_label.append(val_data[-1])
            single_index_val_pred, single_index_tt_pred = [[] for _ in model_method], [[] for _ in model_method]
            single_val_stat = [[] for _ in model_method]
            single_index_tt_pred_proba = [[] for _ in model_method]
            single_index_val_pred_proba = [[] for _ in model_method]
            for s_percentage in self.percentage:
                tr_spectra = self._extract_spectra(s_percentage, selection_method, 
                                                   tr_data[0], tr_wave) 
                val_spectra_group = [self._extract_spectra(q, selection_method, 
                                                           val_data[0], tr_wave) for q in self.percentage]
                tt_spectra_group = [self._extract_spectra(q, selection_method, 
                                                          tt_data[0], tr_wave) for q in self.percentage]
                for model_index, s_model in enumerate(model_method):
                    if self.quantification:
                        _predictor = self._get_regressor(s_model)
                        _predictor.fit(tr_spectra, tr_data[-2])
                    elif self.detection:
                        _predictor = self._get_classifier(s_model)
                        _predictor.fit(tr_spectra, tr_data[1])
                    _val_pred = [_predictor.predict(_s_val) for _s_val in val_spectra_group]
                    _tt_pred = [_predictor.predict(_s_tt) for _s_tt in tt_spectra_group]
                    if self.detection:
                        _val_pred_prob = [_predictor.predict_proba(_s_tt) for _s_tt in val_spectra_group]
                        _tt_pred_prob = [_predictor.predict_proba(_s_tt) for _s_tt in tt_spectra_group]
                        single_index_tt_pred_proba[model_index].append(_tt_pred_prob)
                        single_index_val_pred_proba[model_index].append(_val_pred_prob)
                    single_index_val_pred[model_index].append(_val_pred)
                    single_index_tt_pred[model_index].append(_tt_pred)
                    if self.quantification:
                        single_val_stat[model_index].append(self._calc_quan_stat(_val_pred, val_data[-2]))
                    elif self.detection:
                        single_val_stat[model_index].append(self._calc_detect_stat(_val_pred, 
                                                                                   val_data[1]))
            for model_index, s_model in enumerate(model_method):
                perf_val[s_model].append(single_index_val_pred[model_index])
                perf_tt[s_model].append(single_index_tt_pred[model_index])
                val_stat[s_model].append(single_val_stat[model_index])
                perf_tt_proba[s_model].append(single_index_tt_pred_proba[model_index])
                perf_val_proba[s_model].append(single_index_val_pred_proba[model_index])
        for s_model in model_method:
            perf_val[s_model] = np.array(perf_val[s_model])
            perf_tt[s_model] = np.array(perf_tt[s_model])
            val_stat[s_model] = np.array(val_stat[s_model])
            perf_tt_proba[s_model] = np.array(perf_tt_proba[s_model])
            perf_val_proba[s_model] = np.array(perf_val_proba[s_model])
            
        tt_perf, tt_act_pred, tt_act_prob = [], [], []
        tt_label = np.concatenate(tt_label, axis=0)
        tt_conc = np.concatenate(tt_conc, axis=0)
        for s_model in model_method:
            if self.quantification:
                _tt_opt, _tt_perf = self._find_best_index(val_stat[s_model], perf_tt[s_model], tt_conc)
            elif self.detection:
                print("The shape of the tt probablity", np.shape(perf_tt_proba[s_model]))
                _tt_opt, _tt_perf, _tt_prob = self._find_best_detect_index(val_stat[s_model], perf_tt[s_model], perf_tt_proba[s_model], tt_label)
                tt_act_prob.append(_tt_prob)
            tt_act_pred.append(_tt_opt)
            tt_perf.append(_tt_perf)  
                  
        stat_group = [[perf_val, perf_val_proba, val_label, val_conc], perf_tt, val_stat, tt_act_pred, tt_perf, np.array(tt_conc), np.array(tt_label), np.array(tt_act_prob)]
        if save:
            if self.quantification:
                with open(self.tds_dir + "/%s_concentration_prediction.obj" % (self.dataset), "wb") as f:
                    pickle.dump(stat_group, f)
            elif self.detection:
                with open(self.tds_dir + "/%s_detection.obj" % self.dataset, "wb") as f:
                    pickle.dump(stat_group, f)        
        return [perf_val, perf_val_proba], perf_tt, val_stat, tt_act_pred, tt_perf, tt_act_prob
    
    def _find_best_index(self, val_stat, tt_pred, tt_conc):
        """Args:
        val_stat: [num_leave_index, num_perf, num_perf]
        tt_pred: [num_leave_index, num_perf, num_perf, 1]
        """
        val_stat_avg = np.reshape(np.mean(val_stat, axis=0), [-1])
        tt_stat_reshape = np.reshape(tt_pred, [len(tt_pred), -1])
        tt_opt = tt_stat_reshape[:, np.argmax(val_stat_avg)]
        tt_rsquare = utils.calc_rsquare(tt_conc, tt_opt)
        return tt_opt, tt_rsquare        
    
    def _find_best_detect_index(self, val_stat, tt_pred, tt_pred_prob, tt_label):
        val_stat_avg = np.reshape(np.mean(val_stat, axis=0), [-1])
        tt_stat_reshape = np.reshape(tt_pred, [len(tt_pred), -1])
        tt_prob_reshape = np.reshape(tt_pred_prob, [len(tt_pred), -1, 2])
        tt_opt = tt_stat_reshape[:, np.argmax(val_stat_avg)]
        tt_prob_opt = tt_prob_reshape[:, np.argmax(val_stat_avg)]
        
        accu = np.sum(tt_opt == tt_label) / len(tt_label)
        return tt_opt, accu, tt_prob_opt 
    
    def _calc_quan_stat(self, pred, gt_concentration):
        return [utils.calc_rsquare(gt_concentration, v) for v in pred]
    
    def _calc_detect_stat(self, pred, gt_label):
        accu_g = [np.sum(v == gt_label) / len(gt_label) for v in pred]
        return accu_g 
                
    def _get_regressor(self, s_method):
        if s_method == "random_forest":
            return RandomForestRegressor()
        elif s_method == "decision_tree":
            return DecisionTreeRegressor()
        elif s_method == "adaboost":
            return AdaBoostRegressor()
        elif s_method == "gradient_boost":
            return GradientBoostingRegressor()
        elif s_method == "svm":
            return SVR()
        
    def _get_classifier(self, s_method):
        if s_method == "knn":
            return KNeighborsClassifier()
        elif s_method == "random_forest":
            return RandomForestClassifier()
        elif s_method == "adaboost":
            return AdaBoostClassifier()
        elif s_method == "gradient_boost":
            return GradientBoostingClassifier()
        elif s_method == "svm":
            return SVC(probability=True)
        
        
def get_detection_heatmap_for_ml(path2load_exp="../exp_data/", path2load_label="../rs_dataset/"):
    heatmap_all_data = {}
    detection, quantification = True, False
    for dataset in ["TOMAS", "DNP", "PA"]:
        de_path = path2load_exp + "/ml_methods/%s_detection.obj" % dataset 
        value = pickle.load(open(de_path, "rb"))
        label, concentration = pickle.load(open(path2load_label + "/%s_label_conc_detection_%s_quantification_%s.obj" % (
            dataset, detection, quantification), "rb"))
        val_prob = value[0][1]
        val_label = label[0]
        key_use = list(value[0][1].keys())
        heatmap_g = []
        for s_key in key_use:
            s_perf = get_val_map(True, np.array(val_prob[s_key]), val_label, [])
            heatmap_g.append(s_perf)
        heatmap_all_data[dataset] = heatmap_g
    with open(de_path.split("%s_detection" % dataset)[0] + "detection_heatmap.obj", "wb") as f:
        pickle.dump([heatmap_all_data, key_use], f)
        
        
def get_quantification_heatmap_for_ml(path2load_exp="../exp_data/", path2load_label="../rs_dataset/"):
    heatmap_all_data = {}
    detection, quantification = False, True
    for dataset in ["TOMAS", "DNP", "PA"]:
        de_path = path2load_exp + "/ml_methods/%s_concentration_prediction.obj" % dataset 
        value = pickle.load(open(de_path, "rb"))
        label, concentration = pickle.load(open(path2load_label + "/%s_label_conc_detection_%s_quantification_%s.obj" % (
            dataset, detection, quantification), "rb"))
        if dataset != "TOMAS":
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
        val_conc = concentration[0]
        val_label = label[0]
        val_prob = value[0][0]
        key_use = list(value[0][0].keys())
        heatmap_g = []
        for s_key in key_use:
            s_perf = get_val_map(False, np.exp(val_prob[s_key]), val_label, val_conc)
            heatmap_g.append(s_perf)
        heatmap_all_data[dataset] = heatmap_g
    with open(de_path.split("%s_conc" % dataset)[0] + "quantification_heatmap.obj", "wb") as f:
        pickle.dump([heatmap_all_data, key_use], f)
                       
                   
def get_val_map(detection, val_prob, val_label, val_conc):
    if detection:
        num_measure, num_tr, num_tt, num_val_data, num_cls = np.shape(val_prob)
        label_reshape = np.reshape(val_label, [-1])
        label_tensor = torch.from_numpy(label_reshape).long()
    else:
        num_measure, num_tr, num_tt, num_val_data = np.shape(val_prob)
    val_perf = np.zeros([num_tr, num_tt])    
    for i in range(num_tr):
        for j in range(num_tt):
            if detection:
                s_prob = np.reshape(val_prob[:, i, j], [num_measure * num_val_data, num_cls])
                s_prob_tt = torch.from_numpy(s_prob).to(torch.float32)
                val_loss = torch.nn.CrossEntropyLoss(reduction='sum')(s_prob_tt, label_tensor).cpu().detach().numpy()
                val_perf[i, j] = val_loss / len(label_reshape)        
            else:
                s_pred = val_prob[:, i, j]
                log_rsquare = [utils.calc_rsquare_log(_gt, _pred) for _gt, _pred in zip(val_conc, s_pred)]
                val_perf[i, j] = np.mean(log_rsquare, axis=0)
    return val_perf                
                   
    
"""
Created on 10:56 at 04/01/2022
@author: bo
"""
import numpy as np
import os
import utils
import data.prepare_sers_data as psd
import configs.common as common
import matplotlib.pyplot as plt
import vis_utils as vu


log_base=2.3
if log_base==2:
    log_fun = np.log2 
elif log_base==10:
    log_fun=np.log10 
elif log_base==2.3:
    log_fun = np.log 


def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argsort(memory_available)


num_gpu = 1
free_id = get_freer_gpu()
use_id = free_id[-num_gpu:]
use_id_list = ",".join(["%d" % i for i in use_id])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = use_id_list


def get_sers_modeldir(version=0, lr=0.01, 
                      quantification=True, detection=True, model_cls="VIT",
                      percentage=0.0, top_selection_method="all",
                      concentration_float=0.0,
                      dataset="Marlitt", avg_spectra=True, 
                      leave_index=0, 
                      quantification_loss="rmae", 
                      loc="home"):
    if loc == "home":
        path_start = "../"
    elif loc == "scratch":
        path_start = "/scratch/blia/"
    elif loc == "nobackup":
        path_start = "/nobackup/blia/"
    else:
        path_start = loc
    path_mom = path_start + "exp_data/%s/%s/" % (model_cls, dataset)
    path_mom += "detection_%s_quantification_%s" % (detection, quantification)
    if "Spectra" in model_cls:
        path_mom += "_average_spectra_%s/" % avg_spectra
    path_mom += "/"
    subset = [v for v in os.listdir(path_mom) if "version_%d_" % version in v]
    subset = [v for v in subset if  "learning_rate_%.4f" % lr in v] # or "learning_rate_%.3f" % lr in v]
    if model_cls != "VIT":
        subset = [v for v in subset if "selection_method_%s_select_percentage_%.3f" % (top_selection_method,
                                                                                           percentage) in v]
    if quantification:
        subset = [v for v in subset if "concentration_float_%.4f" % concentration_float in v]
    
    if model_cls == "VIT":
        if quantification:
            subset = [v for v in subset if "quantification_loss_%s" % quantification_loss in v]
    
    if len(subset) == 1:
        model_dir = path_mom + subset[0] + "/"
        print(model_dir)
        repeat_dir = [v for v in os.listdir(model_dir) if "repeat_" in v and int(v.split("repeat_")[1]) == leave_index]
        model_dir += repeat_dir[0]
        model_dir += "/"
        tds_dir = model_dir.split("version_")[0] + "/tds/version_" + model_dir.split("version_")[1]
        if not os.path.exists(tds_dir):
            os.makedirs(tds_dir)
        print(tds_dir)
        return model_dir, tds_dir
    elif len(subset) == 0:
        print("Model does not exist")
        return [], []
    else:
        return [path_mom + v + "/" for v in subset], []
    
    
def get_ckpt(dataset, detection, quantification, const, exp_dir="/nobackup/blia/exp_data/VIT/"):
    data_path = exp_dir + "/%s/detection_%s_quantification_%s" % (dataset, detection, quantification)
    if quantification:
        data_path += "_nozero"
    data_path + "/"
    model_dir = [v for v in sorted(os.listdir(data_path)) if "version_" in v and "learning_rate_%.4f" % const.lr in v]
    print("There are %d models for ensemble calculation" % len(model_dir))
    model_dir = [v + "/%s_repeat_%d/" % (const.leave_method, const.leave_index) for v in model_dir]
    model_dir = [data_path + "/" + v + "/" for v in model_dir]
    ckpt_group = []
    for s_model_path in model_dir:
        if quantification:
            all_ckpts = sorted([v for v in os.listdir(s_model_path) if ".ckpt" in v and "validation_quantification_rmae" in v and '-v1.ckpt' not in v])    
            accu = [float(v.split("validation_quantification_rmae=")[1].split(".ckpt")[0]) for v in all_ckpts]
            s_ckpt = np.array(all_ckpts)[np.argsort(accu)[0]]
        else:
            all_ckpts = sorted([v for v in os.listdir(s_model_path) if ".ckpt" in v and "validation_accuracy" in v and '-v1.ckpt' not in v])
            accu = [float(v.split("validation_accuracy=")[1].split(".ckpt")[0]) for v in all_ckpts]
            s_ckpt = np.array(all_ckpts)[np.argsort(accu)[-1]]
        model_ckpt = s_model_path + "/" + s_ckpt
        ckpt_group.append(model_ckpt)
    return ckpt_group 


class Test(object):
    def __init__(self, sers_maps, labels, concentration, peaks,
                 model_use, patch_size, image_size,
                 cast_quantification_to_classification=False,
                 tds_dir=None, save=False):
        super(Test, self).__init__()
        self.sers_maps = sers_maps
        self.peaks = peaks
        self.labels = labels
        self.concentration = concentration
        self.wavenumber = np.arange(np.shape(self.sers_maps)[-1])
        self.model_use = model_use
        self.patch_size = patch_size
        self.image_size = image_size
        self.cast_quantification_to_classification = cast_quantification_to_classification
        self.tds_dir = tds_dir 
        self.save = save
        print("The concentration float ", model_use.hparams.concentration_float)

    def get_spectra_accuracy(self):
        out_g, quan_g = [], []
        batch_size = [1600 if len(self.sers_maps) > 400 else len(self.sers_maps)][0]
        num_iter = len(self.sers_maps) / batch_size
        for i in range(int(np.ceil(num_iter))):
            s_t, _ = psd.get_test_sers_tensor(False, self.sers_maps[i * batch_size:(i + 1) * batch_size],
                                              self.model_use)
            feat, _out_g, _quan_g = self.model_use.model.forward_test(s_t)
            if self.model_use.hparams.detection is True:
                out_g.append(_out_g.detach().cpu().numpy())
            if self.model_use.hparams.quantification is True:
                quan_g.append(_quan_g.detach().cpu().numpy())
        if self.model_use.hparams.detection:
            out_g = np.array([v for q in out_g for v in q])
            out_g = np.reshape(out_g, [len(self.labels), -1, len(np.unique(self.labels))])
            self._give_detection_performance(out_g)
        if self.model_use.hparams.quantification:
            rsquare, quan_g = self._give_quantification_performance(quan_g)
        else:
            rsquare = 0.0

        return out_g, quan_g, rsquare

    def get_vit_accuracy(self):
        out_g, quan_g = [], []
        for i, s_map in enumerate(self.sers_maps):
            s_t, _ = psd.get_test_sers_tensor(False, np.expand_dims(s_map, 0), self.model_use)
            feat, _out_g, _quan_g = self.model_use.vit(s_t)
            if self.model_use.hparams.detection is True:
                out_g.append(_out_g.detach().cpu().numpy())
            if self.model_use.hparams.quantification is True:
                quan_g.append(_quan_g.detach().cpu().numpy())
        if self.model_use.hparams.detection is True:
            self._give_detection_performance(out_g)
        if self.model_use.hparams.quantification:
            rsquare, quan_g = self._give_quantification_performance(quan_g)
        else:
            rsquare = 0.0

        return out_g, quan_g, rsquare
    
    def _give_detection_performance(self, out_g):
        if np.shape(out_g)[1] == 1:
            print("Accuracy: %.4f" % (
                    (np.argmax(out_g, axis=-1)[:, 0] == self.labels).sum() / len(self.labels)))
            vu.show_accuracy_over_concentration(np.array(out_g)[:, 0], 
                                                np.argmax(out_g, axis=-1)[:, 0], 
                                                self.labels, 
                                                self.concentration, show=True, 
                                                save=False, tds_dir=None)
        else:
            print("Accuracy: %.4f" % (
                    (np.argmax(np.mean(out_g, axis=1), axis=-1) == self.labels).sum() / len(self.labels)))
    
    def _give_quantification_performance(self, pred_quan):
        quan_exp = np.array([v for g in pred_quan for v in g])
        if 0.0 < self.model_use.hparams.concentration_float < 1.0:
            quan_exp = np.maximum(quan_exp, log_fun(self.model_use.hparams.concentration_float))
            if log_base != 2.3:
                quan_exp = np.power(log_base, quan_exp) #- self.model_use.hparams.concentration_float
            else:
                quan_exp = np.exp(quan_exp)

        elif self.model_use.hparams.concentration_float == 1.0:
            quan_exp[np.where(quan_exp < 0)[0]] = 0 + 1e-5
            quan_exp = np.log(quan_exp)
        elif self.model_use.hparams.concentration_float > 1.0:
            quan_exp = quan_exp / self.model_use.hparams.concentration_float                
        else:
            quan_exp = quan_exp
            
        if not self.cast_quantification_to_classification:
            rsquare = utils.calc_rsquare(self.concentration, quan_exp)
        else:
            pred_quan = np.unique(self.concentration)[np.argmax(pred_quan, axis=-1)[:, 0]]
            rsquare = utils.calc_rsquare(self.concentration, pred_quan)
        return rsquare, quan_exp

    def _get_single_attention_map(self, s_map):
        s_tensor, s_npy = psd.get_test_sers_tensor(False, s_map, self.model_use)
        if len(s_tensor) > 1:
            attn_maps = []
            for i in range(len(s_tensor)):
                _, _, _, _, _attn_map = self.model_use.vit.forward_test(s_tensor[i:(i + 1)])
                attn_maps.append(_attn_map)
        else:
            _, _, _, _, attn_maps = self.model_use.vit.forward_test(s_tensor)
        if len(s_tensor) > 1:
            attn_maps = np.squeeze(np.array(attn_maps), axis=2)
        del s_tensor
        return attn_maps, np.transpose(s_npy, (0, 3, 1, 2))

    def get_query_key_value(self, s_map):
        s_tensor, s_npy = psd.get_test_sers_tensor(False, s_map, self.model_use)
        if len(s_tensor) > 1:
            query_key_value_g, attn_maps = [], []
            for i in range(len(s_tensor)):
                _, _, _, _query_key_value, _attn_map = self.model_use.vit.forward_test(s_tensor[i:(i + 1)])
                attn_maps.append(_attn_map)
                query_key_value_g.append(_query_key_value)
        else:
            _, _, _, query_key_value_g, attn_maps = self.model_use.vit.forward_test(s_tensor)
        if len(s_tensor) > 1:
            attn_maps = np.squeeze(np.array(attn_maps), axis=2)
        del s_tensor
        return attn_maps, query_key_value_g, np.transpose(s_npy, (0, 3, 1, 2))

    def _reorganize_attention_map_to_image(self, s_npy, s_attention_maps, s_peak, show=False, save=False):
        attn_maps_rollout, attn_maps_imsize = utils.overlap_attention_cls_token_sers_multi(s_attention_maps,
                                                                                           0,
                                                                                           self.patch_size,
                                                                                           self.image_size)
        if show:
            utils.show_attention_maps(s_npy, np.transpose(attn_maps_imsize[:, -1:], (1, 0, 2, 3)), self.concentration,
                                      self.wavenumber, select_wave=s_peak, tds_dir=self.tds_dir, save=save, 
                                      save_name="_".join(["%d" % q for q in s_peak]))
        return attn_maps_rollout, attn_maps_imsize

    def get_attention_maps_multiple_tensor(self):
        attention_maps_group, attention_maps_imsize_group = [], []
        for i, v in enumerate(self.sers_maps):
            attn_maps, s_npy = self._get_single_attention_map(np.expand_dims(v, axis=0))
            attn_maps_rollout, attn_maps_imsize = self._reorganize_attention_map_to_image(s_npy, attn_maps, self.peaks[i])
            attention_maps_group.append(attn_maps_rollout)
            attention_maps_imsize_group.append(attn_maps_imsize)
        return np.array(attention_maps_group), np.array(attention_maps_imsize_group)
    

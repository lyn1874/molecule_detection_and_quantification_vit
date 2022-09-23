"""
Created on 10:56 at 04/01/2022
@author: bo
"""
from unittest import skip
from cv2 import normalize
import torch
import numpy as np
import os
import utils
import data.prepare_sers_data as psd
import data.read_tomas as read_tomas 
import data.read_dnp as read_dnp 
import data.read_pa as read_pa 
import train_spectra as train_spectra
import pickle
import configs.common as common
import train_vit as train_vit
from sklearn.metrics import f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import csv
import vis_utils as vu
import sys
import csv
import shutil



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


def run_prediction_performance(dataset, model_vit, version_group=[10], avg_spectra=True, 
                               lr=0.0005, detection=True, quantification=False,
                               normalization="none", quantification_loss="mae", 
                               select_method=["top_peak"],
                               leave_index_subset=[0, 1, 2, 3, 4], loc="scratch"):
    if len(select_method) == 0:
        select_method = ["avg_map_dim", "top_std", "top_diff", "top_mean", "top_peak"] 
    if "SIMU" in dataset:
        for version in version_group:
            for s_select in select_method:
                if "SIMU" in dataset:
                    get_prediction_and_performance_for_spectra_experiment(s_select, model_vit=model_vit,
                                                                        quantification=quantification, detection=detection, 
                                                                        dataset=dataset, version=version,
                                                                        avg_spectra=avg_spectra, lr=lr, 
                                                                        normalization=normalization,
                                                                        quantification_loss=quantification_loss, 
                                                                        concentration_float=1e-6, 
                                                                        loc=loc)
    else:
        if len(leave_index_subset) == 0:
            max_leave = 30 if dataset != "PA" else 25
            if loc == "scratch" and quantification == True:
                max_leave = max_leave - 5
            leave_index_subset = np.arange(max_leave).astype(np.int32)
        
        if dataset == "TOMAS":
            targ_shape = [56, 56]
        elif dataset == "DNP":
            targ_shape = [44, 44]
        elif dataset == "PA":
            targ_shape = [40, 40]
        concentration_float = 1e-6 if dataset != "PA" else 1e-5 
        for leave_index in leave_index_subset:
            data_obj = psd.ReadSERSData(dataset, target_shape=targ_shape,
                                        bg_method="ar", tr_limited_conc=[0],
                                        percentage=0, top_selection_method="sers_maps",
                                        path_mom="../rs_dataset/", quantification=quantification, detection=detection, 
                                        normalization=normalization, leave_index=leave_index, leave_method="leave_one_chip", 
                                        skip_value=1)
            _, _val_data, _tt_data, imshape, num_class = data_obj.forward_test()
            for version in version_group:
                for s_select in select_method:
                    obj_name = "version_%d_selection_method_%s_leave_chip_%d_normalization_%s" % (version, s_select, leave_index, normalization)
                    tds_dir = "../exp_data/Spectra_%s/%s/detection_%s_quantification_%s_average_spectra_True/tds/" % (model_vit, dataset, detection, quantification)
                    if os.path.isfile(tds_dir.split("tds/")[0] + "/tds/" + "%s.obj" % obj_name):
                        print("exist")
                        continue     
                    get_prediction_and_performance_for_spectra_experiment(s_select, model_vit=model_vit, 
                                                                          quantification=quantification, detection=detection, dataset=dataset,
                                                                          version=version, avg_spectra=avg_spectra, lr=lr, 
                                                                          normalization=normalization, concentration_float=concentration_float,    
                                                                          leave_index=leave_index, 
                                                                          leave_method="leave_one_chip", 
                                                                          quantification_loss=quantification_loss, 
                                                                          data_group=[_val_data, _tt_data, imshape, num_class], loc=loc)
    
    
def save_attention_maps_tomas_dnp_pa(dataset, detection=False):
    if dataset == "PA":
        targ_shape = 40 
    elif dataset == "DNP":
        targ_shape = 44
    elif dataset == "TOMAS":
        targ_shape = 56
    target_shape = [targ_shape, targ_shape]
    if dataset == "DNP":
        if detection == False:
            detection, quantification, concentration_float, normalization, quantification_loss = False, True, 1e-6, "none", "mse"
            version, lr, loc = 12266, 0.006, "nobackup"
            leave_index = 0
        else:
            detection, quantification, concentration_float, normalization, quantification_loss = True, False, 0, "none", "none"
            version, lr, loc = 32619, 0.005, "nobackup"
            leave_index = 15
        sers_maps, label, \
            concentration, wavenumber, \
                mapsize = read_dnp.prepare_dnp_data(target_shape=target_shape, skip_value=1, 
                                                    padding_approach="zero", leave_index=0, leave_method="leave_one_chip",
                                                    path="../rs_dataset/DNP_Ag_obj/", check_filename=False, 
                                                    testing=True)
        model_init, lr_schedule="xavier", "cosine"
    elif dataset == "TOMAS":
        detection, quantification, concentration_float, normalization, quantification_loss = False, True, 1e-6, "none", "mae"
        sers_maps, label, \
            concentration, wavenumber, \
                mapsize = read_tomas.prepare_tomas_data([56, 56], skip_value=1, 
                                                        padding_approach="zero", leave_index=0, 
                                                        leave_method="leave_one_chip", path="../rs_dataset/Tomas_obj/", 
                                                        check_filename=False, testing=True)
        version, lr, loc = 21, 0.08, "home"
        model_init, lr_schedule="xavier", "cosine"
        leave_index = 0

    elif dataset == "PA":
        if detection == False:
            detection, quantification, concentration_float, normalization, quantification_loss = False, True, 1e-5, "none", "mse"
            version, lr, loc=20042, 0.0004, "nobackup"
            leave_index=14
        else:
            detection, quantification, concentration_float, normalization, quantification_loss = True, False, 0, "none", "none"
            version, lr, loc=20887, 0.0006, "nobackup"
            leave_index=4
        model_init="xavier"
        lr_schedule="cosine"
        sers_maps, label, \
            concentration, wavenumber, \
                mapsize = read_pa.prepare_pa_data(target_shape=target_shape, skip_value=1, 
                                                    padding_approach="zero", leave_index=0, leave_method="leave_one_chip",
                                                    path="../rs_dataset/PA_obj/", check_filename=False, 
                                                    testing=True)

    imshape = target_shape + [np.shape(sers_maps)[-1]]
    model_dir, tds_dir = get_sers_modeldir(2, 2, version=version, lr=lr,  conc=[0], quantification=quantification,
                                            detection=detection,
                                            model="eetti16", model_cls="VIT", 
                                            percentage=0, top_selection_method="sers_maps",
                                            concentration_float=concentration_float,
                                            dataset=dataset, avg_spectra=True, 
                                            cast_quantification_to_classification=False,
                                            leave_index=leave_index,
                                            quantification_loss=quantification_loss,
                                            normalization=normalization,
                                            target_shape=target_shape,
                                            model_init=model_init, lr_schedule=lr_schedule,
                                            loc=loc) 
    print(model_dir)
    if quantification:
        ckpt_group = sorted([v for v in os.listdir(model_dir) if ".ckpt" in v and "validation_quantification_rmae" in v and '-v1.ckpt' not in v])    
        accu = [float(v.split("validation_quantification_rmae=")[1].split(".ckpt")[0]) for v in ckpt_group]
        use_ckpt = np.array(ckpt_group)[np.argsort(accu)[0]]
        use_ckpt = [v for v in os.listdir(model_dir) if "validation" not in v and ".ckpt" in v and "-v1.ckpt" not in v][0]
    else:
        ckpt_group = sorted([v for v in os.listdir(model_dir) if ".ckpt" in v and "validation_accuracy" in v and '-v1.ckpt' not in v])
        accu = [float(v.split("validation_accuracy=")[1].split(".ckpt")[0]) for v in ckpt_group]
        use_ckpt = np.array(ckpt_group)[np.argsort(accu)[-1]]
    model_ckpt = model_dir + "/" + use_ckpt
    const = common.give_tt_dict("eetti16")
    model_use = train_vit.ViTLightning(2, tuple([2, 2]), 
                                       const["input_feature"], const["num_layers"], const["num_heads"], 
                                       const["mlp_dim"], 0, 2, 10, 10, num_gpu, 0.03, add_positional_encoding=True, 
                                       quantification=quantification, detection=detection, imshape=imshape,
                                       concentration_float=concentration_float,
                                       cast_quantification_to_classification=False)
    model_use = model_use.load_from_checkpoint(model_ckpt) #, hyperparameters_to_override={"within_dropout"})
    model_use.eval()
    model_use.requires_grad_(False)
    model_use.to(torch.device("cuda"))
    
    test_obj = Test(sers_maps, label, concentration, np.ones([len(sers_maps), 1]), model_use,
                    [2, 2], imshape[:-1],
                    model_type="VIT", 
                    cast_quantification_to_classification=False, 
                    tds_dir=tds_dir, save=False)
    attn_maps_group, attn_maps_imsize_group = test_obj.get_attention_maps_multiple_tensor()
    attn_maps_imsize_group = np.array(attn_maps_imsize_group)
    plotly_dir = "../exp_data/VIT/%s/detection_%s_quantification_%s/" % (dataset, detection, quantification)
    print("==========================saving data===================================")
    utils.save_plotly_data(sers_maps, concentration, attn_maps_imsize_group, np.ones([len(sers_maps), 1]), wavenumber,
                           target_shape,
                            plotly_dir, "%s_%d_norm_%s" % (dataset, leave_index, normalization))
    

    
    
def save_attention_maps(dataset, detection=True, quantification=False, leave_index=0):
    if "SIMU_TYPE_" in dataset:
        target_shape = [30, 30]
        if int(dataset.split("_")[-1]) < 10:
            enc_lr = 0.2
            concentration_float = 0
        else:
            enc_lr = 0.08
            concentration_float = d10
    else:
        target_shape = [56, 56]
        concentration_float = 1e-6
    if "SIMU_TYPE" in dataset:
        normalization, leave_index, skip_value, leave_method = "none", 0, 0, "none"
    else:
        normalization = ["max" if detection else "none"][0]
    quantification_loss = ["mae" if quantification else "none"][0]
    l_file = "../exp_data/VIT/%s/detection_%s_quantification_%s/performance_patch_2_lr_%.3f.csv" % (dataset, detection, quantification, enc_lr)
    ll = csv.reader(open(l_file))
    rows = np.concatenate([list(ll)[1:]], axis=0)
    if detection:
        accuracy = rows[:, -3].astype(np.float32)
    else:
        accuracy = rows[:, 4].astype(np.float32)
    version = rows[:, 0].astype(np.float32)[np.argmax(accuracy)]
    if dataset == "SIMU_TYPE_4":
        version = 1905
    data_obj = psd.ReadSERSData(dataset, target_shape=target_shape, 
                                bg_method="ar",
                                tr_limited_conc=[0], 
                                percentage=0, top_selection_method="sers_maps", 
                                path_mom="../rs_dataset/", use_map=False, quantification=quantification,
                                detection=detection,
                                cast_quantification_to_classification=False,
                                normalization=normalization, leave_index=leave_index, 
                                skip_value=skip_value, leave_method="leave_one_chip")
    [tr_maps, tr_label, tr_conc, tr_peak, tr_wave], [val_maps, val_label, val_conc, val_peak], \
        [tt_maps, tt_label, tt_conc, tt_peak, tt_wave], imshape, num_class = data_obj.forward_test()
    model_dir, tds_dir = get_sers_modeldir(2, 2, version, enc_lr,  [0], False, quantification=quantification,
                                          detection=detection,
                                          model="eetti16", model_cls="VIT", 
                                          percentage=0, top_selection_method="sers_maps",
                                          concentration_float=concentration_float,
                                          dataset=dataset, avg_spectra=True, 
                                          cast_quantification_to_classification=False,
                                          leave_index=leave_index,
                                          quantification_loss=quantification_loss,
                                          normalization=normalization,
                                          target_shape=target_shape,
                                          loc="home")
    print(model_dir)
    if not quantification:
        ckpt_group = sorted([v for v in os.listdir(model_dir) if ".ckpt" in v and "accuracy=" in v])
        accu = [float(v.split("accuracy=")[1].split(".ckpt")[0]) for v in ckpt_group if "accuracy=" in v]
        s_index = np.argsort(accu)[-1]
    else:
        ckpt_group = sorted([v for v in os.listdir(model_dir) if ".ckpt" in v and "rsquare" in v])
        if "TOMAS" not in dataset:
            accu = [float(v.split("rsquare=")[1].split(".ckpt")[0]) for v in ckpt_group]
            s_index = np.argsort(accu)[-1]
        else:
            accu = [float(v.split("validation_quantification_rmae=")[1].split(".ckpt")[0]) for v in ckpt_group]
            s_index = np.argsort(accu)[0]
    use_ckpt = np.array(ckpt_group)[s_index]
    model_ckpt = model_dir + "/" + use_ckpt
    const = common.give_tt_dict("eetti16")
    model_use = train_vit.ViTLightning(num_class, tuple([2, 2]), 
                                   const["input_feature"], const["num_layers"], const["num_heads"], 
                                   const["mlp_dim"], 0, 2, 10, 10, num_gpu, 0.03, add_positional_encoding=True, 
                                   quantification=quantification, detection=detection, imshape=imshape,
                                      cast_quantification_to_classification=False)
    model_use = model_use.load_from_checkpoint(model_ckpt) #, hyperparameters_to_override={"within_dropout"})
    model_use.eval()
    model_use.requires_grad_(False)
    model_use.to(torch.device("cuda"))
    
    maps_group = np.concatenate([tr_maps, tt_maps], axis=0)
    label_group = np.concatenate([tr_label, tt_label], axis=0)
    conc_group = np.concatenate([tr_conc, tt_conc], axis=0)
    peak_group = np.concatenate([tr_peak, tt_peak], axis=0)

    test_obj = Test(maps_group, label_group, conc_group, peak_group, tr_wave, model_use,
                    [2, 2], imshape[:-1],
                   model_type="VIT", 
                   cast_quantification_to_classification=False, 
                   tds_dir=tds_dir, save=False)
    attn_maps_group, attn_maps_imsize_group = test_obj.get_attention_maps_multiple_tensor()
    attn_maps_imsize_group = np.array(attn_maps_imsize_group)
    plotly_dir = l_file.split("performance")[0] + "/"
    print("==========================saving data===================================")
    utils.save_plotly_data(maps_group, conc_group, attn_maps_imsize_group, peak_group, tr_wave,
                           target_shape,
                     plotly_dir, "%s_%d" % (dataset, version))


def get_prediction_and_performance_for_spectra_experiment(selection_method, model_vit="unified_cnn",
                                                          quantification=False, detection=True,
                                                          dataset="SIMU_CONC_MULTI_ANALYTE", version=0, 
                                                          avg_spectra=True, lr=0.0005, normalization="max", 
                                                          concentration_float=0.0, leave_index=0,
                                                          leave_method="leave_one_chip",
                                                          quantification_loss="mae", data_group=[], loc="scratch"):
    if selection_method == "avg_map_dim" or selection_method == "all":
        percentage = [1.0]
    else:
        percentage = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
        # percentage = [0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75]
    percentage_select = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    if "SIMU" in dataset:    
        target_shape = [30, 30]
    elif "TOMAS" in dataset:
        target_shape = [56, 56]
    elif dataset == "DNP":
        target_shape = [44, 44]
    elif dataset == "PA":
        target_shape = [40, 40]
    model_cls = "Spectra_%s" % model_vit
    np.random.seed(24907)
    
    data_obj = psd.ReadSERSData(dataset, target_shape=target_shape,
                                bg_method="ar", tr_limited_conc=[0],
                                percentage=0, top_selection_method="sers_maps",
                                path_mom="../rs_dataset/", quantification=quantification, detection=detection, 
                                normalization=normalization, leave_index=leave_index, leave_method=leave_method, 
                                skip_value=1)
    if len(data_group) == 0:
        _, [val_maps, val_label, val_conc, val_peak], [tt_maps, tt_label, tt_conc, tt_peak, tt_wave], imshape, num_class = data_obj.forward_test()
    else:
        [val_maps, val_label, val_conc, val_peak], [tt_maps, tt_label, tt_conc, tt_peak, tt_wave], imshape, num_class = data_group
    model_use = train_spectra.XceptionLightning(num_class, batch_size=512, num_samples=45000, warmup_epochs=10,
                                                num_gpus=1, enc_lr=0.03, quantification=quantification,
                                                wavenumber=110, strategy="dp", epochs=50,
                                                reduce_channel_first=True,
                                                model_type=model_vit)
    performance_obj = {}
    for i, s_per in enumerate(percentage):
        print("----------------------------------------------------------")
        print("         Work on method %s percentage %.3f" % (selection_method, s_per))
        print("----------------------------------------------------------")
        
        model_dir, tds_dir = get_sers_modeldir(version=version, lr=lr,
                                               quantification=quantification, detection=detection,
                                               model_cls=model_cls,
                                               concentration_float=concentration_float,
                                               percentage=s_per, top_selection_method=selection_method,
                                               dataset=dataset, leave_index=leave_index, 
                                               normalization=normalization, 
                                               quantification_loss=quantification_loss,
                                               cast_quantification_to_classification=False,
                                               target_shape=target_shape, loc=loc)
        print(model_dir)
        # if len([v for v in os.listdir(model_dir) if '.ckpt' in v and "validation" in v]) == 0:
        #     continue
        if "SIMU" in model_dir or "TOMAS" in model_dir or "DNP" in model_dir or "PA" in model_dir:
            # if quantification is False:        
            #     ckpt_group = sorted([v for v in os.listdir(model_dir) if ".ckpt" in v and "accuracy=" in v])
            #     accu = [float(v.split("accuracy=")[1].split(".ckpt")[0]) for v in ckpt_group]
            #     ckpt_dir = model_dir + np.array(ckpt_group)[np.argsort(accu)[-1]]
            # else:
            #     ckpt_group = sorted([v for v in os.listdir(model_dir) if ".ckpt" in v and "quantification_rmae" in v])
            #     accu = [float(v.split("quantification_rmae=")[1].split(".ckpt")[0]) for v in ckpt_group]
            #     ckpt_dir = model_dir + np.array(ckpt_group)[np.argsort(accu)[0]]
            # print(ckpt_group)
            if len([v for v in os.listdir(model_dir) if ".ckpt" in v and "validation" not in v]) >= 1:
                ckpt_dir = model_dir + [v for v in os.listdir(model_dir) if ".ckpt" in v and "validation" not in v and "v1-" not in v][0]
            else:
                ckpt_dir = model_dir + sorted(np.array([v for v in os.listdir(model_dir) if ".ckpt" in v]))[-1]
            if "PA" in model_dir and detection == True and "Spectra_resnet" in model_dir:
                ckpt_group = sorted([v for v in os.listdir(model_dir) if ".ckpt" in v and "accuracy=" in v])
                accu = [float(v.split("accuracy=")[1].split(".ckpt")[0]) for v in ckpt_group]
                ckpt_dir = model_dir + np.array(ckpt_group)[np.argsort(accu)[-1]]
            print(ckpt_dir)
        if not os.path.isfile(ckpt_dir):
            continue
        model_use = model_use.load_from_checkpoint(ckpt_dir)
        model_use.eval()
        model_use.to(torch.device("cuda"))
        print(model_use.hparams.concentration_float)
        if selection_method == "avg_map_dim":
            tt_str_group = ["top_peak", "top_std", "top_diff", "top_mean"]
        else:
            tt_str_group = [selection_method]
        for tt_selection_method in tt_str_group:
            percentage_g = [1.0] + percentage_select
            quantile_g = [False]
            for tt_selection_percentage in percentage_g:
                for quantile in quantile_g:
                    if tt_selection_percentage == 1.0:
                        use_selection_method = "avg_map_dim"
                    else:
                        use_selection_method = tt_selection_method 
                    val_maps_update, _, _, _ = data_obj.select_top_spectra(val_maps, val_label, val_conc, val_peak, 
                                                                           tt_selection_percentage, use_selection_method, 
                                                                           use_map=False, avg=avg_spectra, wavenumber=tt_wave)
                    tt_maps_update, _, _, _ = data_obj.select_top_spectra(tt_maps, tt_label, tt_conc, tt_peak, 
                                                                          tt_selection_percentage, use_selection_method, 
                                                                          use_map=False, avg=avg_spectra, wavenumber=tt_wave)    
                    # print("validation shape", np.shape(val_maps_update), np.shape(tt_maps_update))
                    # print("label shape", np.shape(val_label), np.shape(tt_label))
                    # print("concentration group", np.shape(val_conc), np.shape(tt_conc))
                    maps_group = np.concatenate([val_maps_update, tt_maps_update], axis=0)
                    label_group = np.concatenate([val_label, tt_label], axis=0)
                    concentration_group = np.concatenate([val_conc, tt_conc], axis=0)
                    if len(np.shape(val_peak)) != len(np.shape(tt_peak)):
                        val_peak = np.expand_dims(val_peak, axis=1)
                    peak_group = np.concatenate([val_peak, tt_peak], axis=0)
                    print("Using selection method %s with percentage %.2f with quantile %s: %d" % (use_selection_method, tt_selection_percentage, quantile, len(tt_maps_update)))
                    wavecut = np.arange(np.shape(tt_maps_update)[-1])
                    test_obj = Test(maps_group, label_group, concentration_group, peak_group, model_use,
                                    [2, 2], imshape[:-1],
                                    model_type=model_cls)
                    pred, quan, rsquare = test_obj.get_spectra_accuracy()
                    if detection is True:
                        performance_obj["method_%s_percentage_%.3f_tt_method_%s_tt_percent_%.3f_tt_quantile_%s" % (selection_method,
                                                                    s_per, tt_selection_method, tt_selection_percentage, quantile)] = pred
                    if quantification is True:
                        performance_obj["method_%s_percentage_%.3f_tt_method_%s_tt_percent_%.3f_tt_quantile_%s" % (selection_method,
                                                                    s_per, tt_selection_method, tt_selection_percentage, quantile)] = quan
                    plt.close('all')
    if "SIMU" in dataset:
        obj_name = "version_%d_selection_method_%s" % (version, selection_method)
    elif dataset == "TOMAS" or dataset == "DNP" or dataset == "PA":
        obj_name = "version_%d_selection_method_%s_leave_chip_%d_normalization_%s" % (version, selection_method, leave_index, normalization)
    if "/scratch/blia/" in tds_dir:
        tds_dir = tds_dir.replace("/scratch/blia/", "../")
    if "/nobackup/blia/" in tds_dir:
        tds_dir = tds_dir.replace("/nobackup/blia/", "../")
    if not os.path.exists(tds_dir.split("tds/")[0] + "/tds/"):
        os.makedirs(tds_dir.split("tds/")[0] + "/tds/")
    with open(tds_dir.split("tds/")[0] + "/tds/" + "%s.obj" % obj_name, "wb") as f:
        pickle.dump(performance_obj, f)
        
def find_fail(model, dataset="TOMAS", detection=True, quantification=False, version_g=[10, 11]):
    stdoutOrigin = sys.stdout
    if model != "VIT":
        if not os.path.exists("../exp_data/Spectra_%s/%s/" % (model, dataset)):
            os.makedirs("../exp_data/Spectra_%s/%s/" % (model, dataset))
        sys.stdout = open("../exp_data/Spectra_%s/%s/" % (model, dataset) + "stat_%s_%s.txt" % (detection, quantification), 'w')
    else:
        path = "../exp_data/VIT/%s/" % dataset 
        if not os.path.exists(path):
            os.makedirs(path)
        sys.stdout = open(path + "/stat_%s_%s.txt" % (detection, quantification), 'w') 
    for version in version_g:
        if model != "VIT":
            for method in ["top_peak", "avg_map_dim"]: #, "top_std", "top_mean", "top_diff", "avg_map_dim"]:
                if method != "avg_map_dim":
                    percentage = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
                else:
                    percentage = [1.0]            
                for s in percentage:
                    print("percentage==================%.3f %s version %d" % (s, method, version))
                    if "SIMU" not in dataset:
                        fail_tomas_dnp_pa(s, model, method, version, detection, quantification, dataset=dataset)
                    else:
                        fail_simu(s, model, method, version, dataset, detection, quantification)
        else:
            if dataset == "TOMAS":
                fail_vit(version, detection, quantification, dataset)
    sys.stdout.close()
    sys.stdout = stdoutOrigin
    
    
def remove_folder():
    path = "/nobackup/blia/exp_data/Spectra_resnet/TOMAS/detection_False_quantification_True_average_spectra_True/"
    sub_folder = [v for v in os.listdir(path) if "version_" in v]
    for s_folder in sub_folder:
        if int(s_folder.split("version_")[1].split("_")[0]) in [20, 21, 22, 23, 24, 5, 6, 7, 8, 9]:
    #         print(s_folder)
            pass
        else:
    #         print(s_folder)
            shutil.rmtree(path + s_folder)
            
                
def fail_tomas_dnp_pa(percentage, model, method, version, detection=True, quantification=False, dataset="TOMAS"):
    path_mom = "/nobackup/blia/exp_data/Spectra_%s/%s/detection_%s_quantification_%s_average_spectra_True/" % (model, dataset, detection, quantification)
    path_use = [v for v in os.listdir(path_mom) if "version_%d_" % version in v and "method_%s_" % method in v and "percentage_%.3f" % percentage in v][0]
    path2read = path_mom + path_use 
    sub = [v for v in os.listdir(path2read) if "leave_one_chip" in v]
    num_max = 30 if dataset != "PA" else 25
    if dataset == "TOMAS" or dataset == "DNP" or dataset == "PA":
        if len(sub) < num_max:
            sub_int = [int(v.split("chip_repeat_")[1]) for v in sub]
            print([v for v in np.arange(num_max) if v not in sub_int])            
    for v in sub:
        a = len([q for q in os.listdir(path2read + "/" + v) if ".ckpt" in q])
        if a < 1:
            print(v)


def fail_simu(percentage, model, method, version, dataset, detection=True, quantification=False):
    path_mom = "/nobackup/blia/exp_data/Spectra_%s/%s/detection_%s_quantification_%s_average_spectra_True/" % (model, dataset, detection, quantification)
    path_use = [v for v in os.listdir(path_mom) if "version_%d" % version in v and "method_%s_" % method in v and "percentage_%.3f" % percentage in v][0]
    path2read = path_mom + path_use 
    a = len([v for v in os.listdir(path2read) if ".ckpt" in v])
    if a != 4:
        print(path_use)
        

def fail_vit(version, detection=False, quantification=True, dataset="PA", model_init="xavier", lr_schedule="cosine"):
    path_mom = "/nobackup/blia/exp_data/VIT/%s/detection_%s_quantification_%s/" % (dataset, detection, quantification)
    if dataset != "PA" and dataset != "DNP" and dataset != "TOMAS":
        path_use = [v for v in os.listdir(path_mom) if "version_%d" % version in v][0]
    else:
        path_use = [v for v in os.listdir(path_mom) if "version_%d" % version in v and model_init in v]
        if dataset == "PA":
            path_use = [v for v in path_use if lr_schedule in v]
        path_use = path_use[0]
    path2read = path_mom + path_use
    sub = [v for v in os.listdir(path2read) if "leave_one_chip" in v]
    for v in sub:
        a = len([q for q in os.listdir(path2read + "/" + v) if ".ckpt" in q])
        if a < 5:
            print(version, v)
        
    
        
def get_tomas_multiple(version_group, lr=0.08, quantification=True, detection=False, 
                       concentration_float=1e-6, normalization="none", 
                       quantification_loss="mse",
                       leave_method="leave_one_chip",
                       loc="nobackup"):
    for s_version in version_group:
        get_vit_performance([s_version], target_shape=[56, 56], patch_size=2, lr=lr, model_type="eetti16",
                            quantification=quantification, detection=detection, 
                                  concentration_float=concentration_float, cast_quantification_to_classification=False, 
                                  normalization=normalization, quantification_loss=quantification_loss, dataset="TOMAS",
                                  leave_method=leave_method, 
                                  model_init="xavier", lr_schedule="cosine",
                                  loc=loc)
        

def get_dnp_multiple_performance(version_group, lr=0.005, quantification=False, detection=True, 
                                 concentration_float=1e-6, normalization="max", quantification_loss="none",
                                 loc="nobackup"):
    for s_version in version_group:
        get_vit_performance([s_version], target_shape=[44, 44], patch_size=2, lr=lr, 
                                  model_type="eetti16", quantification=quantification, detection=detection, 
                                  concentration_float=concentration_float, cast_quantification_to_classification=False, 
                                  normalization=normalization, quantification_loss=quantification_loss, 
                                  dataset="DNP", skip_value=0, leave_method="leave_one_chip", loc=loc)
        
        
def get_pa_multiple_performance(version_group, quantification=False, 
                                 model_init="xavier", lr_schedule="cosine",
                                 loc="nobackup"):
    if quantification:
        detection=False
        lr, concentration_float, normalization, quantification_loss, lr_schedule=0.0006, 1e-5, "none", "mse", "cosine"
    else:
        detection=True
        lr, concentration_float, normalization, quantification_loss=0.0006, 0, "none", "none"
    for s_version in version_group:
        get_vit_performance([s_version], target_shape=[40, 40], patch_size=2, lr=lr, 
                                  model_type="eetti16", quantification=quantification, detection=detection, 
                                  concentration_float=concentration_float, cast_quantification_to_classification=False, 
                                  normalization=normalization, quantification_loss=quantification_loss, 
                                  dataset="PA", skip_value=0, leave_method="leave_one_chip", 
                                  model_init=model_init, lr_schedule=lr_schedule,
                                  loc=loc)
        
def get_vit_performance(version_group=[9907,7770,12596,21201,23128,16621], 
                              target_shape=[30, 30], patch_size=2, lr=0.08, model_type="eetti16", 
                              quantification=True, detection=False, 
                              concentration_float=1e-6, cast_quantification_to_classification=False, 
                              normalization="none", 
                              quantification_loss="mae",
                              dataset="TOMAS",
                              skip_value=1, 
                              leave_method="leave_one_chip", 
                              model_init="xavier", lr_schedule="cosine",                              
                              loc="nobackup"):
    if leave_method == "leave_one_chip_per_conc":
        leave_group = np.arange(5).astype(np.int32)
    elif leave_method == "leave_one_chip":
        max_index = 30 if dataset != "PA" else 25
        max_index = max_index - 5 if quantification == True else max_index
        leave_group = np.arange(max_index).astype(np.int32)
    perf_tot = []
    pred_tot = []
    for leave_index in leave_group:
        performance_per_index, \
            pred_per_index, \
                col_name = get_performance_for_vit_experiment(version_group, dataset, 
                                                              target_shape=target_shape,
                                                              patch_size=patch_size, lr=lr, model_type=model_type,
                                                            quantification=quantification, detection=detection, concentration_float=concentration_float,
                                                            cast_quantification_to_classification=cast_quantification_to_classification,
                                                            normalization=normalization, leave_index=leave_index, 
                                                            skip_value=skip_value, 
                                                            quantification_loss=quantification_loss, 
                                                            leave_method=leave_method, 
                                                            model_init=model_init, lr_schedule=lr_schedule,
                                                            save_result=False, loc=loc)
        perf_tot.append(performance_per_index)
        pred_tot.append(pred_per_index)
    perf_tot = np.array([v for q in perf_tot for v in q])
    
    
    csv_filename = "performance_patch_%d_lr_%.4f_normalization_%s_%s_target_h_%d_%s_v%d.csv" % (patch_size, lr, 
                                                                                             normalization,
                                                                                             quantification_loss,
                                                                                             target_shape[0], 
                                                                                             leave_method, version_group[0])
    obj_filename = "stat_patch_%d_lr_%.4f_normalization_%s_%s_target_h_%d_%s_v%d.obj" % (patch_size, lr, normalization, 
                                                                                      quantification_loss, target_shape[0], 
                                                                                             leave_method, version_group[0])
    
    csv_filename = csv_filename.replace("lr_%.4f" % lr, "lr_%.4f_%s_initialisation_%s" % (lr, lr_schedule, model_init))
    obj_filename = obj_filename.replace("lr_%.4f" % lr, "lr_%.4f_%s_initialisation_%s" % (lr, lr_schedule, model_init))

    if not os.path.exists("../exp_data/VIT/%s/detection_%s_quantification_%s/" % (dataset, detection, 
                                                                                    quantification)):
        os.makedirs("../exp_data/VIT/%s/detection_%s_quantification_%s/" % (dataset, detection, 
                                                                            quantification))

    
    with open("../exp_data/VIT/%s/detection_%s_quantification_%s/%s" % (dataset, 
                                                                        detection,
                                                                        quantification,
                                                                        obj_filename), "wb") as f:
        pickle.dump(pred_tot, f)
    with open("../exp_data/VIT/%s/detection_%s_quantification_%s/%s" % (dataset, 
                                                                        detection, 
                                                                        quantification, 
                                                                        csv_filename), "w") as f:
        writer = csv.writer(f)
        writer.writerow(col_name)
        for v in perf_tot:
            writer.writerow(v)        

        
        
def get_performance_for_vit_experiment(version_group, dataset, target_shape=[30, 30],
                                       patch_size=2, lr=0.04, model_type="eetti16", 
                                       quantification=False, detection=True, concentration_float=0.0,
                                       cast_quantification_to_classification=False, normalization="max", leave_index=0,
                                       quantification_loss="mae",  skip_value=1,
                                       leave_method="leave_one_chip_per_conc", 
                                       model_init="xavier", lr_schedule="cosine",
                                       save_result=False, loc="home"):
    path_mom="../rs_dataset/"
    data_obj = psd.ReadSERSData(dataset, target_shape=target_shape, bg_method="ar",
                                tr_limited_conc=[0], percentage=0,
                                top_selection_method="sers_maps", 
                                path_mom=path_mom, use_map=False,
                                detection=detection,
                                quantification=quantification, 
                                normalization=normalization, leave_index=leave_index,
                                skip_value=skip_value, 
                                leave_method=leave_method)
    _, _, tt_out, imshape, num_class = data_obj.forward_test()
    tt_maps, tt_label, tt_conc, tt_peak, tt_wave = tt_out 
    const = common.give_tt_dict(model_type)
    print(version_group)
    performance_group = []
    performance_per_conc = []
    performance_obj = {}
    for i, s_version in enumerate(version_group):
        _perf_per_model = []
        model_dir, tds_dir = get_sers_modeldir(patch_height=patch_size,
                                               patch_width=patch_size, 
                                               version=s_version, lr=lr, 
                                               quantification=quantification, detection=detection,
                                               concentration_float=concentration_float, 
                                               model=model_type, model_cls="VIT", 
                                               dataset=dataset,
                                               cast_quantification_to_classification=cast_quantification_to_classification,
                                               leave_index=leave_index,
                                               quantification_loss=quantification_loss,
                                               normalization=normalization,
                                               target_shape=target_shape, 
                                               model_init=model_init, lr_schedule=lr_schedule,
                                               loc=loc)
        print(s_version, model_dir)
        if quantification:
            use_str = "validation" # it is always valiation_quantification_rmae
            ckpt_group = sorted([v for v in os.listdir(model_dir) if ".ckpt" in v and use_str in v and '-v1.ckpt' not in v])    
            #accu = [float(v.split(use_str)[1].split(".ckpt")[0]) for v in ckpt_group]
            ckpt_group = np.array(ckpt_group)#[np.argsort(accu)[:1]]
        else:
            use_str="validation" # it is always validation_accuracy
            ckpt_group = sorted([v for v in os.listdir(model_dir) if ".ckpt" in v and use_str in v and '-v1.ckpt' not in v])
            #accu = [float(v.split("validation_accuracy=")[1].split(".ckpt")[0]) for v in ckpt_group]
            ckpt_group = np.array(ckpt_group)#[np.argsort(accu)[-1:]]
        # ckpt_group = []
        ckpt_group = list(ckpt_group) + [v for v in os.listdir(model_dir) if ".ckpt" in v and "validation" not in v and '-v1.ckpt' not in v]
        print(len(ckpt_group))
        performance_obj_per_version = []
        for ckpt_index, s_ckpt in enumerate(ckpt_group):
            if "validation" in s_ckpt:
                ckpt_step = int(s_ckpt.split("epoch=")[1].split("-validation")[0])
                ckpt_perf = s_ckpt.split("-validation_")[1].split(".ckpt")[0]
            else:
                ckpt_step = int(s_ckpt.split("epoch=")[1].split(".ckpt")[0])
                ckpt_perf = "end_of_training"
            model_ckpt = model_dir + "/" + s_ckpt
            model_use = train_vit.ViTLightning(num_class, tuple([patch_size, patch_size]), 
                                    const["input_feature"], const["num_layers"], const["num_heads"], 
                                    const["mlp_dim"], 0, 2, 10, 10, num_gpu, 0.03, 
                                    concentration_float=concentration_float,
                                    add_positional_encoding=True, 
                                    quantification=quantification, detection=detection, imshape=imshape)
            model_use = model_use.load_from_checkpoint(model_ckpt)
            model_use.eval()
            model_use.to(torch.device("cuda"))
            if quantification is True and model_use.hparams.concentration_float < 1:
                tt_conc_update = tt_conc.copy()
                tt_conc_update[tt_conc_update == 0] = model_use.hparams.concentration_float 
            else:
                tt_conc_update = tt_conc 
            test_obj = Test(tt_maps, tt_label, tt_conc_update, tt_peak, 
                            model_use, [patch_size, patch_size], imshape[:-1], 
                            model_type="VIT")
            
            pred, quan, rsquare = test_obj.get_vit_accuracy()
            if detection:
                pred_label = np.argmax(np.array(pred), axis=-1)[:, 0]
                accuracy = np.sum(tt_label == pred_label) / len(tt_label)
                performance_obj["version_%d" % s_version] = pred
            if detection:
                accu_per_conc_per_class, _, xtick_label = vu.show_accuracy_over_concentration(pred, pred_label, tt_label,
                                                                                        tt_conc, show=False)
                _f1_score = f1_score(tt_label, np.argmax(pred, axis=-1)[:, 0])
                _pred_label = np.argmax(pred, axis=-1)[:, 0]
                correct = np.sum(_pred_label[tt_label == 0] == 0) / np.sum(tt_label == 0)
                _perf_per_model.append([s_version, leave_index, ckpt_step, accuracy, _f1_score, correct])
                performance_per_conc.append(accu_per_conc_per_class)
                performance_obj_per_version.append(pred)
                
            if quantification:
                rsquare_g, rae_g, rmae_g, mae_mse_g = utils.get_quantification_performance(tt_conc_update, quan, concentration_float=concentration_float)
                _perf_per_model.append([s_version, leave_index] + [ckpt_step, ckpt_perf] + rsquare_g + rae_g + rmae_g + mae_mse_g)
                # performance_group[i, ckpt_index] = [s_version] + [ckpt_step] + rsquare_g + rae_g + rmae_g
                performance_obj_per_version.append(quan)
        _perf_per_model = np.array(_perf_per_model)
        print("The shape of the performance group", np.shape(_perf_per_model))
        if quantification:
            if "SIMU" in dataset:
                best_index = np.argsort((_perf_per_model[:, 4]).astype(np.float32))[-1]
            elif "TOMAS" in dataset or "DNP" in dataset or "PA" in dataset:
                best_index = np.argsort((_perf_per_model[:, -1]).astype(np.float32))[0]
        if detection:
            best_index = np.argsort((_perf_per_model[:, -3]).astype(np.float32))[-1]            
        performance_obj["version_%d" % s_version] = performance_obj_per_version[best_index]    
        performance_group.append(_perf_per_model)
    performance_group = np.array(performance_group)
    print("The shape of the performance_group", np.shape(performance_group))
    performance_obj["concentration"] = tt_conc 
    performance_obj["label"] = tt_label
    if detection:
        performance_tot = np.reshape(performance_group, [-1, np.shape(performance_group)[-1]])
        first_row = ["version", "leave_index", "ckpt step", "accuracy", "f1"] #+ xtick_label
    if quantification:
        performance_tot = np.reshape(performance_group, [-1, np.shape(performance_group)[-1]])
        first_row = ["version", "leave_index", "ckpt step", "ckpt performance", "rsquare", 
                     "rsquare subset", "rae", "rae subset", "rmae", "rmae subset"]
    if save_result:
        if "SIMU" in dataset:
            csv_filename = "performance_patch_%d_lr_%.3f.csv" % (patch_size, lr)
            obj_filename = "stat_patch_%d_lr_%.3f.obj" % (patch_size, lr)
        elif dataset == "TOMAS" or dataset == "DNP" or dataset == "PA":
            csv_filename = "performance_patch_%d_leave_index_%d_lr_%.3f_normalization_%s_v%d.csv" % (patch_size, leave_index, lr, normalization, s_version)
            obj_filename = "stat_patch_%d_leave_index_%d_lr_%.3f_normalization_%s_v%d.obj" % (patch_size, leave_index, lr, normalization, s_version)
            
        if not os.path.exists("../exp_data/VIT/%s/detection_%s_quantification_%s/" % (dataset, detection, 
                                                                                      quantification)):
            os.makedirs("../exp_data/VIT/%s/detection_%s_quantification_%s/" % (dataset, detection, 
                                                                                quantification))
        with open("../exp_data/VIT/%s/detection_%s_quantification_%s/%s" % (dataset, 
                                                                            detection,
                                                                            quantification,
                                                                            obj_filename), "wb") as f:
            pickle.dump(performance_obj, f)
        with open("../exp_data/VIT/%s/detection_%s_quantification_%s/%s" % (dataset, 
                                                                            detection, 
                                                                            quantification, 
                                                                            csv_filename), "w") as f:
            writer = csv.writer(f)
            writer.writerow(first_row)
            for v in performance_tot:
                writer.writerow(v)        
    else:
        plt.close('all')
        return performance_tot, performance_obj, first_row
        

def get_sers_modeldir(patch_height=2, patch_width=2,
                      version=0, lr=0.01, conc=[0],
                      quantification=True, detection=True, model="s16", model_cls="VIT",
                      percentage=0.0, top_selection_method="all",
                      concentration_float=0.0,
                      dataset="Marlitt", avg_spectra=True, 
                      leave_index=0, cast_quantification_to_classification=False, 
                      quantification_loss="rmae", normalization="none", 
                      model_init="xavier", lr_schedule="cosine",
                      target_shape=[30, 30], loc="home"):
    if loc == "home":
        path_start = "../"
    elif loc == "scratch":
        path_start = "/scratch/blia/"
    elif loc == "nobackup":
        path_start = "/nobackup/blia/"
    path_mom = path_start + "exp_data/%s/%s/" % (model_cls, dataset)
    path_mom += "detection_%s_quantification_%s" % (detection, quantification)
    if "Spectra" in model_cls:
        path_mom += "_average_spectra_%s/" % avg_spectra
    path_mom += "/"
    subset = [v for v in os.listdir(path_mom) if "version_%d_" % version in v]
    subset = [v for v in subset if  "learning_rate_%.4f" % lr in v] # or "learning_rate_%.3f" % lr in v]
    if model_cls != "VIT":
    # if model_cls == "VIT":
    #     # subset = [v for v in subset if "patch_height_%d_patch_width_%d" % (patch_height, patch_width) in v]
    #     pass
    # else:
        subset = [v for v in subset if "selection_method_%s_select_percentage_%.3f" % (top_selection_method,
                                                                                           percentage) in v]
    # subset = [v for v in subset if "quantification_%s" % quantification in v]
    if quantification:
        # subset = [v for v in subset if "_cast_quantification_to_cls_%s" % cast_quantification_to_classification in v]    
        subset = [v for v in subset if "concentration_float_%.4f" % concentration_float in v]
    
    if model_cls == "VIT":
        if dataset == "TOMAS" or dataset == "DNP" or dataset == "PA":
            if quantification:
                subset = [v for v in subset if "quantification_loss_%s" % quantification_loss in v]
                # if dataset == "PA":
                subset = [v for v in subset if "%s" % model_init in v]
            if detection == True and dataset == "PA":
                subset = [v for v in subset if "%s" % model_init in v and "%s" % lr_schedule in v]
        # if model_cls == "VIT":
            subset = [v for v in subset if "normalization_%s_" % normalization in v and "target_shape_%d" % target_shape[0] in v]
    
    if len(subset) == 1:
        model_dir = path_mom + subset[0] + "/"
        if dataset == "TOMAS" or dataset == "DNP" or dataset == "PA":
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
    data_path = exp_dir + "/%s/detection_%s_quantification_%s/" % (dataset, detection, quantification)
    model_dir = [v for v in sorted(os.listdir(data_path)) if "version_" in v and "learning_rate_%.4f" % const["lr"] in v]
    print("There are %d models for ensemble calculation" % len(model_dir))
    if dataset in ["TOMAS", "DNP", "PA"]:
        model_dir = [v + "/%s_repeat_%d/" % (const["leave_method"], const["leave_index"]) for v in model_dir]
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
                 model_type="VIT",
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
        self.model_type = model_type
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
            # quan_exp = np.maximum(quan_exp, np.log(self.model_use.hparams.concentration_float))
            # quan_exp = np.exp(quan_exp) #- self.model_use.hparams.concentration_float
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
    

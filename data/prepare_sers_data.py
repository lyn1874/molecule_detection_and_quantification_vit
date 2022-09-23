"""
Created on 15:54 at 24/11/2021
@author: bo
"""
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Dataset
from torch.utils.data import random_split
import torch
import data.utils as data_utils 
import data.read_tomas as read_tomas 
import data.read_dnp as read_dnp 
import data.read_pa as read_pa 


log_base=2.3
if log_base==2:
    log_fun = np.log2 
elif log_base==10:
    log_fun=np.log10 
elif log_base==2.3:
    log_fun = np.log 


class SERSMap(Dataset):
    def __init__(self, maps, labels, concentration, transform):
        super().__init__()
        self.maps = maps
        self.labels = labels
        self.concentration = concentration
        self.transform = transform

    def __len__(self):
        return len(self.maps)

    def __getitem__(self, index):
        s_map = self.maps[index]
        s_label = self.labels[index]
        if self.transform is not None:
            s_map = self.transform(s_map)
        return s_map, s_label, self.concentration[index]


class ReadSERSData(object):
    def __init__(self, dataset, target_shape=[45, 75],
                 percentage=0.0, top_selection_method="sers_maps",
                 avg=True,
                 concentration_float=0.0,
                 quantification=False, 
                 detection=True,
                 leave_index=0,
                 leave_method="leave_one_chip",
                 path_mom="../rs_dataset/"):
        self.dataset = dataset
        self.target_shape = target_shape
        self.path_mom = path_mom
        self.percentage = percentage
        self.top_selection_method = top_selection_method
        self.concentration_float = concentration_float
        self.quantification=quantification
        self.detection = detection
        self.normalization = "none"
        self.avg = avg
        self.leave_index = leave_index
        self.skip_value = 1 
        self.leave_method = leave_method
        self.train_transforms, self.test_transforms = get_sersmap_transforms()
        num_repeat = {}
        num_repeat["TOMAS"] = [[10, 8, 1, 0], [10, 8, 1, 0]]
        num_repeat["DNP"] =[[5, 8, 1, 0], [1, 8, 1, 0]]
        num_repeat["PA"] = [[5, 6, 1, 0], [1, 6, 1, 0]]
        self.repeat_aug = num_repeat[dataset]
            
    def get_tomas(self):
        tr_out, tt_out, wavenumber = read_tomas.prepare_tomas_data(self.target_shape, 
                                                                   self.skip_value, 
                                                                   leave_index=self.leave_index,
                                                                   leave_method=self.leave_method,
                                                                   path=self.path_mom+"/Tomas_obj/",
                                                                   quantification=self.quantification)
        return tr_out, tt_out, wavenumber
    
    def get_dnp(self):
        tr_out, tt_out, wavenumber = read_dnp.prepare_dnp_data(self.target_shape, self.skip_value,
                                                               leave_index=self.leave_index, leave_method=self.leave_method,
                                                               path=self.path_mom + "/DNP_Ag_obj/",
                                                               quantification=self.quantification)
        return tr_out, tt_out, wavenumber
    
    def get_pa(self):
        tr_out, tt_out, wavenumber = read_pa.prepare_pa_data(self.target_shape, skip_value=self.skip_value, 
                                                             leave_index=self.leave_index, leave_method=self.leave_method,
                                                             path=self.path_mom + "/PA_obj/",
                                                             quantification=self.quantification)
        return tr_out, tt_out, wavenumber

    def feed_data_into_dataloader(self, sers_maps, label, concentration, use_transforms):
        print("Log my concentration level")
        zero_index = np.where(concentration == 0.0)[0]
        concentration_update = concentration.copy()
        concentration_update[zero_index] = self.concentration_float
        if self.quantification:
            concentration_use = log_fun(concentration_update)
        elif self.detection:
            concentration_use = concentration_update                
        s_dataloader = SERSMap(sers_maps, label, concentration_use,
                                use_transforms)
        return s_dataloader

    def select_top_spectra(self, sers_maps, label, concentration, 
                           percentage=None, top_selection_method=None, 
                           avg=False, wavenumber=[]):
        """Select the top percentage of the spectra
        Args:
            sers_maps: [num_measurements, imh, imw, wavenumbers]
            label: [num_measurements]
            concentration: [num_measurements]
            peak: [num_measurements, num_peaks]
        """
        if not top_selection_method:
            top_selection_method = self.top_selection_method
            percentage = self.percentage
            avg = self.avg
        imh, imw, = self.target_shape
        num_measurements = len(sers_maps)
        wave = np.shape(sers_maps)[-1]
        map_reshape = np.reshape(sers_maps, [num_measurements, imh * imw, wave])
        print("---------------------------------------------------------------")
        print("                 Use top %.2f spectra with %s" % (percentage * imh * imw,
                                                                 top_selection_method))
        print("---------------------------------------------------------------")
        if top_selection_method == "top_peak" and self.dataset == "TOMAS":
            print("For the 4-NBT dataset, if the selection criterior is peak intensity, get intensity from map regions")
            crit = get_tomas_region(sers_maps, wavenumber)
        elif top_selection_method == "top_peak" and self.dataset == "DNP":
            print("For the DNP dataset, if the selection criterior is peak intensity, get intensity from map regions")
            crit = get_dnp_region(sers_maps, wavenumber)
        elif top_selection_method == "top_peak" and self.dataset == "PA":
            print("For the pricid acid, if the selection criterior is peak intensity, get intensity from map regions")
            crit = get_pa_region(sers_maps, wavenumber)
        else:
            crit = []       
        spectra, label, \
            concentration, num_select_per_map = select_baseon_percentage(map_reshape, label,
                                                                         concentration, crit, 
                                                                         percentage, avg)        
        return spectra, label, concentration, num_select_per_map
    
    def aug_signal(self, tr_stat, val_stat):
        tr_maps, tr_label, tr_conc, tr_wave = tr_stat 
        val_maps, val_label, val_conc = val_stat 
        
        tr_maps, tr_label, tr_conc, tr_peak, _ = data_utils.aug_signal(tr_maps, tr_label, tr_conc, tr_wave, 
                                                                       repeat=self.repeat_aug[0][0], 
                                                                       k=self.repeat_aug[0][1], 
                                                                       norm_std=self.repeat_aug[0][2],
                                                                       norm_mean=self.repeat_aug[0][3], 
                                                                       detection=self.detection, 
                                                                       quantification=self.quantification, 
                                                                       val=False)
        val_maps, val_label, val_conc, val_peak, _ = data_utils.aug_signal(val_maps, val_label, val_conc, tr_wave, 
                                                                           repeat=self.repeat_aug[1][0], 
                                                                           k=self.repeat_aug[1][1], 
                                                                           norm_std=self.repeat_aug[1][2],
                                                                           norm_mean=self.repeat_aug[1][3], 
                                                                           detection=self.detection, quantification=self.quantification, 
                                                                           val=False)
        return [tr_maps, tr_label, tr_conc, tr_peak], [val_maps, val_label, val_conc, val_peak] 

    def forward(self):
        if self.dataset == "TOMAS":
            [tr_maps, tr_label, tr_conc, tr_peak], [tt_maps, tt_label, tt_conc, tt_peak], tr_wave = self.get_tomas()
        elif self.dataset == "DNP":
            [tr_maps, tr_label, tr_conc, tr_peak], [tt_maps, tt_label, tt_conc, tt_peak], tr_wave = self.get_dnp()
        elif self.dataset == "PA":
            [tr_maps, tr_label, tr_conc, tr_peak], [tt_maps, tt_label, tt_conc, tt_peak], tr_wave = self.get_pa()
        num_val=1
        if self.dataset != "PA":        
            tr_update, val_update = data_utils.split_tr_to_tr_val_data(tr_maps, tr_label, tr_conc, tr_peak, 
                                                                           quantification=self.quantification, 
                                                                           num_val=1)             
            tr_maps, tr_label, tr_conc, tr_peak = tr_update
            val_maps, val_label, val_conc, val_peak = val_update
        else:
            tr_update, val_update = data_utils.split_tr_to_tr_val_data(tr_maps, tr_label, tr_conc, tr_peak, 
                                                                           quantification=self.quantification, 
                                                                           num_val=1,
                                                                           dataset=self.dataset, tt_conc=np.unique(tt_conc)[0],
                                                                           detection=self.detection)            
            tr_maps, tr_label, tr_conc, tr_peak = tr_update
            val_maps, val_label, val_conc, val_peak = val_update
            
        [tr_maps, tr_label, tr_conc, tr_peak], \
            [val_maps, val_label, val_conc, val_peak] = self.aug_signal([tr_maps, tr_label, tr_conc, tr_wave],
                                                                        [val_maps, val_label, val_conc])
            
        num_channel = np.shape(tr_maps)[-1]
        print("The shape of the training data", np.shape(tr_maps), np.shape(tr_label))
        print("The shape of the validation dataset", np.shape(val_maps), np.shape(val_label))
        print("The shape of the testing data", np.shape(tt_maps), np.shape(tt_label))
        if self.percentage != 0 or self.top_selection_method != "sers_maps":
            tr_maps, tr_label, tr_conc, _ = self.select_top_spectra(tr_maps, 
                                                                    tr_label, 
                                                                    tr_conc, 
                                                                    wavenumber=tr_wave)
            val_maps, val_label, val_conc, _ = self.select_top_spectra(val_maps, val_label, val_conc, 
                                                                       wavenumber=tr_wave)
            tt_maps, tt_label, tt_conc, _ = self.select_top_spectra(tt_maps, tt_label, tt_conc, 
                                                                    wavenumber=tr_wave)
            data_input_channel = np.shape(tr_maps)[1]
        else:
            data_input_channel = num_channel
            
        tr_conc_array, val_conc_array, tt_conc_array = tr_conc.astype(np.float32), val_conc.astype(np.float32), tt_conc.astype(np.float32)        
            
        print("Top selection method", self.top_selection_method)
        print("The shape of the training data", np.shape(tr_maps), np.shape(tr_label))
        print("The shape of the validation data", np.shape(val_maps), np.shape(val_label))
        print("The shape of the testing data", np.shape(tt_maps), np.shape(tt_label))
        print("The max and min of the training data", np.max(tr_maps), np.min(tr_maps))
        print("The max and min of the test data", np.max(tt_maps), np.min(tt_maps))
        print("---------------------------------------------------------------------------------")
        print("                                 Unique label")
        print("---------------------------------------------------------------------------------")
        
        str_use = ["training", "validation", "testing"]
        for s_str, s_label in zip(str_use, [tr_label, val_label, tt_label]):
            print("==============================%s========================================" % s_str)
            print(np.unique(s_label, return_counts=True))
            
        print("---------------------------------------------------------------------------------")
        print("                                 Unique concentration")
        print("---------------------------------------------------------------------------------")
        for s_str, s_conc in zip(str_use, [tr_conc_array, val_conc_array, tt_conc_array]):
            print("==============================%s========================================" % s_str)
            print(np.unique(s_conc, return_counts=True))
            
        tr_loader = self.feed_data_into_dataloader(tr_maps.astype(np.float32), tr_label.astype(np.int32),
                                                   tr_conc_array, self.train_transforms)
        val_loader = self.feed_data_into_dataloader(val_maps.astype(np.float32), val_label.astype(np.int32),
                                                    val_conc_array, self.test_transforms)
        tt_loader = self.feed_data_into_dataloader(tt_maps.astype(np.float32), tt_label.astype(np.int32),
                                                   tt_conc_array, self.test_transforms)
        num_class = len(np.unique(tr_label))
        num_samples = len(tr_maps)
        imshape = list(self.target_shape) + [num_channel]
        print("There are %d classes" % num_class)
        return tr_loader, val_loader, tt_loader, num_samples, imshape, num_class, data_input_channel

    def forward_test(self):
        if self.dataset == "TOMAS":
            [tr_maps, tr_label, tr_conc, tr_peak], [tt_maps, tt_label, tt_conc, tt_peak], tr_wave = self.get_tomas()
            tt_wave = tr_wave.copy()
        elif self.dataset == "DNP":
            [tr_maps, tr_label, tr_conc, tr_peak], [tt_maps, tt_label, tt_conc, tt_peak], tr_wave = self.get_dnp()
            tt_wave = tr_wave.copy()
        elif self.dataset == "PA":
            [tr_maps, tr_label, tr_conc, tr_peak], [tt_maps, tt_label, tt_conc, tt_peak], tr_wave = self.get_pa()
            tt_wave = tr_wave.copy()
        num_channel = np.shape(tr_maps)[-1]

        
        if self.dataset != "PA":        
            tr_update, val_update = data_utils.split_tr_to_tr_val_data(tr_maps, tr_label, tr_conc, tr_peak, 
                                                                       quantification=self.quantification, 
                                                                       num_val=1)             
            tr_maps, tr_label, tr_conc, tr_peak = tr_update
            val_maps, val_label, val_conc, val_peak = val_update
        else:
            tr_update, val_update = data_utils.split_tr_to_tr_val_data(tr_maps, tr_label, tr_conc, tr_peak, 
                                                                        quantification=self.quantification, 
                                                                        num_val=1,
                                                                        dataset=self.dataset, tt_conc=np.unique(tt_conc)[0],
                                                                        detection=self.detection)            
            tr_maps, tr_label, tr_conc, tr_peak = tr_update
            val_maps, val_label, val_conc, val_peak = val_update

        [tr_maps, tr_label, tr_conc, tr_peak], \
            [val_maps, val_label, val_conc, val_peak] = self.aug_signal([tr_maps, tr_label, tr_conc, tr_wave],
                                                                        [val_maps, val_label, val_conc])
        
        if self.percentage != 0 or self.top_selection_method != "sers_maps":
            tr_maps, tr_label, tr_conc = self.select_top_spectra(tr_maps, tr_label, tr_conc)
            val_maps, val_label, val_conc = self.select_top_spectra(val_maps, val_label, val_conc)
            tt_maps, tt_label, tt_conc = self.select_top_spectra(tt_maps, tt_label, tt_conc)
        imshape = list(self.target_shape) + [num_channel]
        num_class = len(np.unique(tr_label))
        out_group = [[tr_maps, tr_label, tr_conc], [tt_maps, tt_label, tt_conc]]
        use_str = ["Training", "Testing"]
        for i, s_str in enumerate(use_str):
            print("=============================================%s===================================================" % s_str)
            print("Map shape:", np.shape(out_group[i][0]))
            u_l, u_l_c = np.unique(out_group[i][1], return_counts=True)
            print("Label distribution:", ["%d: %d" % (v, q) for v, q in zip(u_l, u_l_c)])
            u_c, u_c_c = np.unique(out_group[i][2], return_counts=True)
            print("Concentration distribution:", ["%.4f: %d" % (v, q) for v, q in zip(u_c, u_c_c)])
        
        return [tr_maps, tr_label, tr_conc, tr_peak, tr_wave], [val_maps, val_label, val_conc, val_peak], \
                [tt_maps, tt_label, tt_conc, tt_peak, tt_wave], imshape, num_class


def get_sersmap_transforms(norm=False):
    tr_transform = transforms.ToTensor()
    tt_transform = transforms.ToTensor()
    return tr_transform, tt_transform


def get_dataloader(tr_data, val_data, tt_data, batch_size, workers):
    train_loader = DataLoader(tr_data, batch_size=batch_size, shuffle=True,
                              drop_last=True, pin_memory=True, num_workers=workers)
    val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=False,
                            drop_last=False, pin_memory=True, num_workers=workers)
    test_loader = DataLoader(tt_data, batch_size=len(tt_data), shuffle=False,
                             drop_last=False, pin_memory=True, num_workers=workers)
    return train_loader, val_loader, test_loader



def get_test_sers_tensor(norm, maps, model):
    maps_tensor = torch.from_numpy(maps).to(torch.float32)
    if len(maps_tensor.shape) == 4:
        maps_tensor = maps_tensor.permute(0, 3, 1, 2).to(model.device)
    else:
        maps_tensor = maps_tensor.to(model.device)
    return maps_tensor, maps


def simple_selection(sers_maps, wavenumber, percentage, method, dataset):
    """Select the top percentage of the spectra
    Args:
        sers_maps: [num_measurements, imh, imw, wavenumbers]
        label: [num_measurements]
        concentration: [num_measurements]
        peak: [num_measurements, num_peaks]
    """
    num_measurements, imh, imw, wave = np.shape(sers_maps)
    if method == "avg_map_dim" or percentage == 1.0:
        spectra = np.mean(sers_maps, axis=(1, 2))
        sort_index = np.concatenate([np.arange(imh * imw) for _ in range(num_measurements)], axis=0)
        crit = np.ones([num_measurements, imh * imw])
    else:
        map_reshape = np.reshape(sers_maps, [num_measurements, imh * imw, wave])
        if method == "top_mean":
            crit = np.mean(map_reshape, axis=-1)
        elif method == "top_std":
            crit = np.std(map_reshape, axis=-1)
        elif method == "top_diff":
            crit = np.sum(abs(np.diff(map_reshape, axis=-1)), axis=-1)
        elif method == "top_peak":
            if dataset == "TOMAS":
                crit = get_tomas_region(sers_maps, wavenumber)
            elif dataset == "DNP":
                crit = get_dnp_region(sers_maps, wavenumber)
            elif dataset == "PA":
                crit = get_pa_region(sers_maps, wavenumber)
        num_select = int(np.shape(map_reshape)[1] * percentage)
        sort_index = np.argsort(crit, axis=-1)[:, -num_select:]
        selected_sersmap = np.concatenate([map_reshape[i:i+1, sort_index[i], :] for i in range(len(map_reshape))], axis=0)
        spectra = np.mean(selected_sersmap, axis=1)
    return spectra, [sort_index, np.reshape(crit, [num_measurements, imh, imw])] 


def select_baseon_percentage(sers_maps, label, concentration, crit, percentage, avg=True):
    """Args:
    sers_maps: [num_measurements, imh * imw, wavenumber]
    label: [num_measurements]
    concentration: [num_measurements]
    crit: [num_measurements, imh * imw]
    percentage: float, 0 - 1    
    avg: whether to average the Top # spectra
    """
    num_select = int(np.shape(sers_maps)[1] * percentage)
    sort_index = np.argsort(crit, axis=-1)[:, -num_select:]
    selected_sersmap = np.concatenate([sers_maps[i:i+1, sort_index[i], :] for i in range(len(sers_maps))], axis=0)
    if avg:
        selected_sersmap = np.mean(selected_sersmap, axis=1, keepdims=True)
        selected_label = label 
        selected_conc = concentration 
    else:
        selected_sersmap = np.reshape(selected_sersmap, [len(sers_maps) * num_select, 1, -1])
        selected_label = np.concatenate([np.repeat(i, num_select) for i in label], axis=0)
        selected_conc = np.concatenate([np.repeat(i, num_select) for i in concentration], axis=0)
    return selected_sersmap, selected_label, selected_conc, [num_select for i in range(len(sers_maps))]


def get_tomas_region(map_use, wavenumber):
    """Args:
    map_use: [num_maps, imh, imw, num_wavenumber]
    wavenumber: [num_wavenumber], this is the real wavenumber rather than np.arange(num_wavenumber)
    """
    _, imh, imw, num_wave = np.shape(map_use)
    assert num_wave == len(wavenumber)
    map_region = [[1071, 1083], [1324, 1336], [1566, 1575]]
    map_select = read_tomas.get_map_at_specified_regions(map_use, map_region, wavenumber)
    crit = np.reshape(map_select, [-1, imh * imw])
    return crit


def get_dnp_region(map_use, wavenumber):
    _, imh, imw, num_wave = np.shape(map_use)
    assert num_wave == len(wavenumber)
    map_region = [[811.98, 850.55], [1272.89, 1340.38]]
    map_select = read_tomas.get_map_at_specified_regions(map_use, map_region, wavenumber)
    crit = np.reshape(map_select, [-1, imh * imw])
    return crit 


def get_pa_region(map_use, wavenumber):
    _, imh, imw, num_wave = np.shape(map_use)
    assert num_wave == len(wavenumber)
    map_region = [[802.34, 840.91], [1311.46, 1350.03]]
    map_select = read_tomas.get_map_at_specified_regions(map_use, map_region, wavenumber)
    crit = np.reshape(map_select, [-1, imh * imw])
    return crit 

    


    





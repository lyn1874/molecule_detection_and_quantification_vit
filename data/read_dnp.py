#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   read_dnp.py
@Time    :   2022/07/13 10:52:56
@Author  :   Bo 
'''
from fileinput import filename
import os 
import numpy as np 
import pickle 
import matplotlib.pyplot as plt
import data.read_tomas as read_tomas 


def get_dnp_data(path="../rs_dataset/DNP_Ag_obj/"):
    sub_file = np.array(sorted([v for v in os.listdir(path) if ".obj" in v]))
    concentration = np.array([float(v.split("M")[0][:-1]) if "Blank" not in v else 0 for v in sub_file]) 
    concentration = np.array([v * 1000 if "nM" not in q and "Blank" not in q else v for v, q in zip(concentration, sub_file)])
    unique_concentration = np.unique(concentration)
    maps_g, wave_g, mapsize_g, filename_g = {}, {}, {}, {}
    for i, s_conc in enumerate(unique_concentration):
        index = np.where(concentration == s_conc)[0]
        all_spectra, all_wave, all_mapsize, all_subfiles = [], [], [], []
        for j, s_c in enumerate(index):
            sers_maps = pickle.load(open(path + sub_file[s_c], "rb"))
            spectra, wavenumber, mapsize = sers_maps["spectrum"], sers_maps["wavenumber"], sers_maps["mapsize"]
            all_spectra.append(spectra)
            all_wave.append(wavenumber)
            all_mapsize.append(mapsize)
            all_subfiles.append(sub_file[s_c])
        # print(s_conc, sub_file[index])
        maps_g["%.1fuM" % s_conc] = all_spectra
        wave_g["%.1fuM" % s_conc] = all_wave
        mapsize_g["%.1fuM" % s_conc] = all_mapsize        
        filename_g["%.1fuM" % s_conc] = all_subfiles
    return maps_g, wave_g, mapsize_g, filename_g



class GetDNPData(object):
    def __init__(self, maps_g, wave_g, mapsize_g, filename_g, padding_approach):
        super(GetDNPData, self).__init__()
        keys = list(maps_g.keys())
        wavenumber = wave_g[keys[-1]][0]
        
        self.wavenumber = wavenumber 
        self.concentration = [float(v.split("uM")[0]) for v in keys]
        self.maps = maps_g 
        self.mapsize = mapsize_g
        self.filename_g = filename_g        
        self.keys = keys        
        self.padding_approach = padding_approach
    
    def cut_maps_wavenumber(self, start_wave, end_wave):
        maps_update = {}
        for k in self.keys:
            _maps_cut = [self._cut_maps_wavenumber(v, q, start_wave, end_wave) for v, q in zip(self.maps[k], self.mapsize[k])]
            maps_update[k] = [v[0] for v in _maps_cut]
        return maps_update, _maps_cut[0][1]
    
    def reduce_map_size(self, maps, skip_size):
        maps_update = {}
        for k in self.keys:
            _map_reduce = [read_tomas.reduce_map_size(v, skip_size) for v in maps[k]]
            maps_update[k] = _map_reduce
        return maps_update 
    
    def cut_maps_imhimw(self, maps_update, target_shape):
        maps_update_update = {}
        mapsize_update = {}
        for k in self.keys:
            _maps_cut = [read_tomas.CutAppendMaps(v, np.shape(v)[:-1], target_shape, self.padding_approach).forward() for v, q in zip(maps_update[k], self.mapsize[k])]
            maps_update_update[k] = _maps_cut 
            mapsize_update[k] = np.array([target_shape for _ in self.mapsize[k]])   
        return maps_update_update, mapsize_update
    
    def _cut_maps_wavenumber(self, sers_maps, mapsize, start=1000, end=1600):
        """Explore the sers maps
        sers_maps: [imh * imw, wavenumber]
        start: int, starting point for cutting the spectra
        end: int, ending point for cutting the spectra
        """
        start_i, end_i = np.where(self.wavenumber >= start)[0][0], np.where(self.wavenumber >= end)[0][0]
        sers_maps_cut = sers_maps[:, start_i:end_i]
        sers_maps_cut = np.reshape(sers_maps_cut, list(mapsize) + [len(self.wavenumber[start_i:end_i])])
        return sers_maps_cut, self.wavenumber[start_i:end_i]   
            
    def forward(self, target_shape, skip_value=1, start_wave=1000, end_wave=1600):
        maps, wavenumber = self.cut_maps_wavenumber(start_wave, end_wave)
        if skip_value > 1:
            maps = self.reduce_map_size(maps, skip_value)
        maps, mapsize = self.cut_maps_imhimw(maps, target_shape)
        data_stat = [[] for _ in range(4)]
        for i, s_k in enumerate(self.keys):
            s_map = maps[s_k]
            data_stat[0].append(s_map)
            data_stat[1].append([float(s_k.split("uM")[0]) for _ in s_map])
            label = [0 if float(s_k.split("uM")[0]) == 0.0 else 1.0][0]
            data_stat[2].append([label for _ in s_map])
            data_stat[3].append(self.filename_g[s_k])
        for i, s_data in enumerate(data_stat):
            data_stat[i] = np.array([v for q in s_data for v in q])
        return data_stat[0], data_stat[1], data_stat[2], data_stat[3], wavenumber, mapsize
    
    
def prepare_dnp_data(target_shape, skip_value=1, 
                       padding_approach="zero", leave_index=-1, 
                       leave_method="leave_one_chip", path="../rs_dataset/DNP_Ag_obj/", 
                       check_filename=False, testing=False, quantification=False):    
    maps_g, wave_g, mapsize_g, filename_g = get_dnp_data(path)
    dnp_obj = GetDNPData(maps_g, wave_g, mapsize_g, filename_g, padding_approach=padding_approach)
    sers_maps, concentration, label, filename_prepare, wavenumber, mapsize = dnp_obj.forward(target_shape, 
                                                                                               skip_value,
                                                                                               start_wave=750, 
                                                                                               end_wave=1500)
    if quantification:
        use_index = np.where(concentration != 0)[0]
        sers_maps = sers_maps[use_index]
        concentration = concentration[use_index]
        label = label[use_index]
        filename_prepare = filename_prepare[use_index]
        
    concentration = concentration / np.max(concentration)
    tr_out, tt_out = split_tr_tt(sers_maps, concentration, label, filename_prepare, wavenumber, leave_index, leave_method)
    tr_chip_index = [int(v.split("_")[1].split(".obj")[0]) for v in tr_out[-1]]
    tt_chip_index = [int(v.split("_")[1].split(".obj")[0]) for v in tt_out[-1]]
    print("Training chip index:", np.unique(tr_chip_index, return_counts=True))
    print("Testing chip index:", np.unique(tt_chip_index, return_counts=True))    
    if leave_method == "leave_one_chip_per_conc":
        assert np.sum([v for v in tt_chip_index if v in tr_chip_index]) == 0
    
    if not testing:
        if check_filename:
            return tr_out[:-1], tt_out[:-1], wavenumber, tr_chip_index, tt_chip_index
        else:
            return tr_out[:-1], tt_out[:-1], wavenumber
    else:
        return sers_maps, label, concentration, wavenumber, mapsize
       
            
def split_tr_tt(sers_maps, concentration, label, filename, wavenumber, leave_index, leave_method):
    tr_index, tt_index = read_tomas.get_tr_tt_index(concentration, leave_index, leave_method)
    tr_map, tr_conc, tr_label = sers_maps[tr_index], concentration[tr_index], label[tr_index]
    tt_map, tt_conc, tt_label = sers_maps[tt_index], concentration[tt_index], label[tt_index]
    tr_filename, tt_filename = filename[tr_index], filename[tt_index]
    tr_peak = np.zeros([len(tr_map), 1]) + np.where(wavenumber >= 1320)[0][0]
    tt_peak = np.zeros([len(tt_map), 1]) + np.where(wavenumber >= 1320)[0][0]
    return [tr_map, tr_label, tr_conc, tr_peak, tr_filename], [tt_map, tt_label, tt_conc, tt_peak, tt_filename]
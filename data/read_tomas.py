#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   read_tomas.py
@Time    :   2022/03/14 12:29:15
@Author  :   Bo 
'''
from fileinput import filename
import os 
import numpy as np 
import pickle 
import matplotlib.pyplot as plt


def get_tomas_data(path="../rs_dataset/Tomas_obj/"):
    sub_file = np.array(sorted([v for v in os.listdir(path) if ".obj" in v]))
    concentration = np.array([float(v.split("PT93_")[1].split("nM")[0]) if "nM" in v else 0.0 for v in sub_file])
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


class CutAppendMaps(object):
    def __init__(self, sers_maps, mapsize, target_size, padding_approach="zero"):
        super(CutAppendMaps, self).__init__()
        imh, imw = mapsize 
        sers_map_reshape = np.reshape(sers_maps, [imh, imw, np.shape(sers_maps)[-1]])
        self.sers_map_reshape = sers_map_reshape 
        self.mapsize = mapsize 
        self.target_size = target_size        
        self.padding_approach = padding_approach
        
    def calc_h_size(self):
        h_diff = self.mapsize[0] - self.target_size[0]
        h_diff_abs = abs(h_diff)
        
        if h_diff_abs % 2 == 0:
            up, bottom = h_diff_abs // 2, h_diff_abs // 2 
        else:
            up = h_diff_abs // 2 
            bottom = h_diff_abs - up 
        
        if h_diff > 0:
            im_update = self.sers_map_reshape[up:-bottom]
        elif h_diff < 0:
            if self.padding_approach == "zero":
                first, last = np.zeros_like(self.sers_map_reshape[:1]), np.zeros_like(self.sers_map_reshape[-1:])
            else:
                first, last = self.sers_map_reshape[:1], self.sers_map_reshape[-1:]
            im_update = np.vstack([np.repeat(first, up, axis=0), 
                                   self.sers_map_reshape, 
                                   np.repeat(last, bottom, axis=0)])
        else:
            im_update = self.sers_map_reshape
        return im_update 
    
    def calc_w_size(self, im_update):
        w_diff = self.mapsize[1] - self.target_size[1]
        w_diff_abs = abs(w_diff)
        if w_diff_abs % 2 == 0:
            left, right = w_diff_abs // 2, w_diff_abs // 2 
        else:
            left = w_diff_abs // 2 
            right = w_diff_abs - left
        if w_diff > 0:
            im_update_w = im_update[:, left:-right]
        elif w_diff < 0:
            if self.padding_approach == "zero":
                first, last = np.zeros_like(im_update[:, :1]), np.zeros_like(im_update[:, -1:])
            else:
                first, last = im_update[:, :1], im_update[:, -1:]
            im_update_w = np.hstack([np.repeat(first, left, axis=1), 
                                     im_update, 
                                     np.repeat(last, right, axis=1)])
        else:
            im_update_w = im_update 
        return im_update_w    
    
    def forward(self):
        im_update_h = self.calc_h_size()
        im_update_w = self.calc_w_size(im_update_h)
        assert np.shape(im_update_w)[0] == self.target_size[0]
        assert np.shape(im_update_w)[1] == self.target_size[1]
        return im_update_w
        


class GetTomasData(object):
    def __init__(self, maps_g, wave_g, mapsize_g, filename_g, padding_approach):
        super(GetTomasData, self).__init__()
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
            _map_reduce = [reduce_map_size(v, skip_size) for v in maps[k]]
            maps_update[k] = _map_reduce
        return maps_update 
    
    def cut_maps_imhimw(self, maps_update, target_shape):
        maps_update_update = {}
        mapsize_update = {}
        for k in self.keys:
            _maps_cut = [CutAppendMaps(v, np.shape(v)[:-1], target_shape, self.padding_approach).forward() for v, q in zip(maps_update[k], self.mapsize[k])]
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
            
    def forward(self, target_shape, skip_value):
        maps, wavenumber = self.cut_maps_wavenumber(1000, 1600)
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
        
        
def prepare_tomas_data(target_shape, skip_value=1, 
                       padding_approach="zero", leave_index=-1, 
                       leave_method="leave_one_chip", path="../rs_dataset/Tomas_obj/", 
                       check_filename=False, testing=False, quantification=False):    
    maps_g, wave_g, mapsize_g, filename_g = get_tomas_data(path)
    tomas_obj = GetTomasData(maps_g, wave_g, mapsize_g, filename_g, padding_approach=padding_approach)
    sers_maps, concentration, label, filename_prepare, wavenumber, mapsize = tomas_obj.forward(target_shape, 
                                                                                               skip_value)
    if quantification:
        use_index = np.where(concentration != 0)[0]
        sers_maps = sers_maps[use_index]
        concentration = concentration[use_index]
        label = label[use_index]
        filename_prepare = filename_prepare[use_index]
    concentration = concentration / np.max(concentration)
    tr_out, tt_out = split_tr_tt(sers_maps, concentration, label, filename_prepare, wavenumber, leave_index, leave_method)
    tr_chip_index = [int(v.split("chip")[1].split(".obj")[0]) for v in tr_out[-1]]
    tt_chip_index = [int(v.split("chip")[1].split(".obj")[0]) for v in tt_out[-1]]
    print("Training chip index:", np.unique(tr_chip_index, return_counts=True))
    print("Testing chip index:", np.unique(tt_chip_index, return_counts=True))    
    if not testing:
        if check_filename:
            return tr_out[:-1], tt_out[:-1], wavenumber, tr_chip_index, tt_chip_index
        else:
            return tr_out[:-1], tt_out[:-1], wavenumber
    else:
        return sers_maps, label, concentration, wavenumber, mapsize


def get_tr_tt_index(concentration, leave_index=0, leave_method="leave_one_chip_per_conc"):
    unique_conc = np.unique(concentration)
    tr_index, tt_index = [], []
    if leave_method == "leave_one_chip_per_conc":
        print("**************Leaving index %d as the test data**************" % leave_index)
        for i, s_conc in enumerate(unique_conc):
            index = np.where(concentration == s_conc)[0]
            tt_index.append(index[leave_index])
            tr_index.append(index[np.delete(np.arange(len(index)), leave_index)])
        tr_index = np.array([v for q in tr_index for v in q])
        tt_index = np.array(tt_index) 
    elif leave_method == "leave_one_chip":
        print("**************I will only leave one measurement out************")
        tr_index = np.delete(np.arange(len(concentration)), leave_index).astype(np.int32)
        tt_index = np.array([leave_index]).astype(np.int32)
    repeat = [v for v in tr_index if v in tt_index]
    assert len(repeat) == 0 
    if leave_method == "leave_one_chip_per_conc":
        assert len(np.unique(concentration[tt_index])) == len(unique_conc)
        assert np.unique(np.unique(concentration[tt_index], return_counts=True)[1]) == 1 
    else:
        assert len(tt_index) == 1 
    return tr_index, tt_index
       
            
def split_tr_tt(sers_maps, concentration, label, filename, wavenumber, leave_index, leave_method):
    tr_index, tt_index = get_tr_tt_index(concentration, leave_index, leave_method)
    tr_map, tr_conc, tr_label = sers_maps[tr_index], concentration[tr_index], label[tr_index]
    tt_map, tt_conc, tt_label = sers_maps[tt_index], concentration[tt_index], label[tt_index]
    tr_filename, tt_filename = filename[tr_index], filename[tt_index]
    tr_peak = np.zeros([len(tr_map), 1]) + np.where(wavenumber >= 1334)[0][0]
    tt_peak = np.zeros([len(tt_map), 1]) + np.where(wavenumber >= 1334)[0][0]
    return [tr_map, tr_label, tr_conc, tr_peak, tr_filename], [tt_map, tt_label, tt_conc, tt_peak, tt_filename]


def reduce_map_size(sers_maps, skip_value):
    """Reduce the size of the map by skipping i.e., every kth row/column
    Args:
        sers_maps: [num_maps, imh, imw, num_wavenumber]
        update_size: [imh_update, imw_update]
    """
    imh, imw, num_wave = np.shape(sers_maps)
    return sers_maps[::skip_value, ::skip_value, :]


def split_tr_val(tr_maps, tr_label, tr_conc, tr_wave):
    """Args:
    tr_maps: [num_tr_maps, imh, imw, num_wave]
    tr_conc: [num_tr_maps]
    tr_label: [num_tr_maps]
    tr_wave: [num_tr_maps, num_wave]    
    """
    tr_index, val_index = [], []
    unique_conc = np.unique(tr_conc)
    for i, s_conc in enumerate(unique_conc):
        index = np.where(tr_conc == s_conc)[0]
        tr_index.append(index[:-1])
        val_index.append(index[-1:])      
    tr_index = np.array([v for q in tr_index for v in q]).astype(np.int32)
    val_index = np.array([v for q in val_index for v in q]).astype(np.int32)
    print("Tr index", tr_index)
    print("Val index", val_index)
 
    tr_stat = [tr_maps[tr_index], tr_label[tr_index], tr_conc[tr_index], tr_wave[tr_index]]
    val_stat = [tr_maps[val_index], tr_label[val_index], tr_conc[val_index], tr_wave[val_index]]
    print("Tr shape", [np.shape(v) for v in tr_stat])
    print("Val shape", [np.shape(v) for v in val_stat])
    return tr_stat, val_stat 


def get_map_at_specified_regions(sers_map, regions, wavenumber):
    """Args:
    sers_map: [number of sample, imh, imw, wavenumber]
    regions: [num_of_peaks, [start_wave, end_wave]]
    """
    sers_map_select = np.zeros_like(sers_map[:, :, :, 0])
    num_divide = 0.0
    for i, s_region in enumerate(regions):
        start, end = np.where(wavenumber >= s_region[0])[0][0], np.where(wavenumber >= s_region[1])[0][0]
        sers_map_select += np.sum(sers_map[:, :, :, start:end], axis=-1)
        # print("selected wavenumber", wavenumber[start], wavenumber[end])
        num_divide += (end - start)
    return sers_map_select / num_divide


def get_intensity_at_specified_regions(spectra, regions, wavenumber):
    """Args:
    spectra: [num of sample, 1, wavenumber]
    regions: [num_of_peaks, [start_wave, end_wave]]
    """
    intensity = np.zeros([len(spectra)])
    for i, s_region in enumerate(regions):
        start, end = np.where(wavenumber >= s_region[0])[0][0], np.where(wavenumber >= s_region[1])[0][0]
        subset_spectra = spectra[:, 0, start:end]
        intensity += np.sum(subset_spectra, axis=-1)
    return intensity

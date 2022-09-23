import h5py
import numpy as np
import logging
import pickle
import os
import matplotlib.pyplot as plt

    
def run_convert_mapx_to_pickle_tomas(save=False):
    path2read = "../rs_dataset/Tomas/"
    path2save = "../rs_dataset/Tomas_obj/"
    if not os.path.exists(path2save):
        os.makedirs(path2save)
    RamanExperimentData(path2read, path2save, save)   
    
    
def run_convert_mapx_to_pickle_dnp_ag(save=False):
    path2read = "../rs_dataset/DNP_Ag/"
    path2save = "../rs_dataset/DNP_Ag_obj/"
    if not os.path.exists(path2save):
        os.makedirs(path2save)
    RamanExperimentData(path2read, path2save, save)


def run_convert_mapx_to_pickle_pa(save=False):
    path2read = "../rs_dataset/PA/"
    path2save = "../rs_dataset/PA_obj/"
    if not os.path.exists(path2save):
        os.makedirs(path2save)
    RamanExperimentData(path2read, path2save, save)

    
class RamanExperimentData(object):
    """
    Class to hold Raman maps saved in mapx files - a flavour of HDF5 files.
    """

    def __init__(self, filenames, path2save, save):
        """
        Save mapx files in obj format that can be opened via Pickle
        """
        self.path2save = path2save
        self.save = save
        if type(filenames) == str:
            if filenames.split(".")[-1] == "mapx":
                filenames = [filenames]
            else:
                filenames = sorted([filenames + '/' + v for v in os.listdir(filenames) if '.mapx' in v])
        if len(filenames) > 0:
            for f in filenames:
                self.s_data = {}
                print("Dealing file.....................", f)
                self.s_data = self.read_mapx(f)
            print("Convert %d files in total" % len(filenames))

    def read_mapx(self, filename):
        # read HDF5 file
        file = h5py.File(filename, 'r')
        # more than one acquired region.. sigh
        if len(file['Regions']) > 1:
            logging.warning(f"File {filename} has more than one region, only largest acquired region is imported")

        # find largest map and extract spectra
        N = 0
        info = file['Regions']
        for group in info.keys():
            dataset = info[group]['Dataset']
            N_cur = np.prod(dataset.shape[:2])
            if N_cur > N:
                spectra = dataset[:]
                N = N_cur
        # get mapsize
        mapsize = (spectra.shape[1], spectra.shape[0])

        # extract wavenumbers
        metadata = file['FileInfo'].attrs
        wl_start = metadata['SpectralRangeStart']
        wl_end = metadata['SpectralRangeEnd']
        w = np.linspace(wl_start, wl_end, spectra.shape[2])

        # reshape spectra into regular size (N x W) - super weird, but it works!
        spectra = spectra.T.reshape((spectra.shape[2], np.prod(spectra.shape[:2])), order='C').T
        # spectra = spectra.T.reshape((spectra.shape[2], np.prod(spectra.shape[:2])), order='F').T

        # close file
        file.close()
        if self.save:
            self.s_data["spectrum"] = spectra
            self.s_data["wavenumber"] = w
            self.s_data["mapsize"] = mapsize

        if self.save:
            save_name = filename.split("/")[-1].split(".mapx")[0] + ".obj"
            if os.path.isfile(self.path2save + "/" + save_name):
                print("File ", self.path2save + "/" + save_name, " already exists, PASS")
            else:
                print("-----save file to obj", self.path2save+'/' + save_name)
                pickle.dump(self.s_data, open(self.path2save + '/' + save_name, 'wb'))
            return []
        else:
            return [spectra, w, mapsize]

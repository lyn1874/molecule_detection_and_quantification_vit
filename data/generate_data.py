import numpy as np
import matplotlib.pyplot as plt
import data.generate_sers_bg as gsb
import data.utils as utils
import os 
import pickle

"""
 Copyright 2021 Technical University of Denmark
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 
     http://www.apache.org/licenses/LICENSE-2.0
 
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.

Author: Tommy Sonne Alstrøm <tsal@dtu.dk> 
(translated to Python and extended functionality by David Frich Hansen <dfha@dtu.dk>)
 $Date: 2021/03/04$

 generate simulated raman data
"""

class SERSGeneratorMore(object):
    def __init__(self, mapsize, num_hotspot, Nw, sbr, concentration, 
                 noise_sigma,
                 seed=None, seed_intensity=None, 
                 c=None, gamma=None, eta=None, fix_hotspot_intensity=False,
                 hotspot_size=3):
        """Args:
        mapsize: [imh, imw]
        num_hotspot: int
        Nw: int, number of wavenumber 
        sbr: float, signal-to-background ratio
        concentration: float, the ratio of the peak intensities of the target anlayte
        noise_sigma: float, the standard deviation for the noise 
        seed: int, seed for hotspots locations
        seed_intensity: int, seed for generating the peaks 
        seed_containminate_intensity: int, seed for generating the peaks for the containminate
        c: peak locations [K]
        gamma (np.array): Full-width-at-half-maximum (FWHM) of Voigt curves (K)
        eta (np.array): Mixture coefficients of Lorentzian and Gaussian (0 <= eta <= 1) (K)
        """
        super(SERSGeneratorMore, self).__init__()
        
        self.mapsize = mapsize 
        self.num_hotspot = num_hotspot
        self.Nw = Nw
        self.sbr = sbr 
        self.concentration = concentration
        self.noise_sigma = noise_sigma
        self.seed = seed 
        self.seed_intensity = seed_intensity
        self.K = len(c)
        self.c = c 
        self.gamma = gamma 
        self.eta = eta
        self.fix_hotspot_intensity=fix_hotspot_intensity
        self.hotspot_size=hotspot_size
        if self.fix_hotspot_intensity:
            print("============================================================================")
            print("ATTENTION: FIX HOTSPOT INTENSITY", self.fix_hotspot_intensity)
            print("============================================================================")

        assert len(self.gamma) == self.K and len(self.eta) == self.K

    def simulate_pseudo_voigt(self, w, c, gamma, eta):
        """
        Args:
            w: wavenumbers, [w, 1]
        Return:
            peak curve: K x W
        """
        num_peak = len(c)
        xdata = np.tile(w, (num_peak, 1))  # K x Nw 
        c_arr = np.tile(c, (self.Nw, 1)).T  # K x Nw 
        gamma_arr = np.tile(gamma, (self.Nw, 1)).T  # K x Nw 
        eta_arr = np.tile(eta, (self.Nw, 1)).T  # K x Nw 
        
        diff = xdata - c_arr 
        kern = diff / gamma_arr 
        diffsq = kern * kern 
        # Lorentizian 
        L = 1 / (1 + diffsq)
        
        # Gaussian
        G_kern = - np.log(2) * diffsq 
        G = np.exp(G_kern)
        
        # vectorized vogits 
        Vo = eta_arr * L + (1 - eta_arr) * G 
        return Vo / np.max(Vo, axis=-1, keepdims=True)
    
    def simulate_background(self, w, background_method, background_parameters):
        """Simulate the SERS background
        Args:
            w: wavenumbers [Nw]
            background_method: str, "ar", "bspline"
            background_parameters: a list of parameters. 
                "ar": initial value, increasing value 
                "bspline: number of knots:
        """
        np.random.seed(self.seed)
        w = np.arange(self.Nw)
        if background_method == "ar":
            background = gsb.AR_process_background(w, background_parameters[0],
                                                   background_parameters[1], smooth=False)
        elif background_method == "bspline":
            background = gsb.B_spline_background(w, np.linspace(0, self.Nw, 
                                                                background_parameters[0]), np.random.randn(background_parameters[0])) 
        elif background_method == "None":
            background = np.zeros([self.Nw])
        else:
            print("The required background does not exist")
        background = background / np.max(background)
        background = np.repeat([background], np.prod(self.mapsize), axis=0) 
        b = np.random.beta(100, 100, size=(np.prod(self.mapsize), 1)) 

        b = np.repeat(b, self.Nw, axis=-1)
        BB = b * background
        return BB, background, b
    
    def simulate_noise(self):
        return self.noise_sigma ** 2 * np.random.randn(np.prod(self.mapsize), self.Nw)
    
    def simulate_hotspot(self):
        map_size_array = np.reshape(self.mapsize, [1, len(self.mapsize)])
        mu = np.repeat(map_size_array, self.num_hotspot, 0) * np.random.rand(self.num_hotspot, 2) # [num_hotspots, 2], where the hotspot is 

        if self.num_hotspot <= 2:
            r = np.random.normal(self.hotspot_size, 1, [self.num_hotspot]) + 2
        else:
            r = abs(np.random.normal(self.hotspot_size-1, 1, [self.num_hotspot]) + 2)
        np.random.seed(self.seed_intensity)

        if self.sbr == 0:
            A = np.random.uniform(0, 0.05, [self.num_hotspot, 1])
        else:
            A = np.random.uniform(0.5, 0.6, [self.num_hotspot, 1]) #/ self.num_hotspot * 2 #* 1.8
        if self.fix_hotspot_intensity:
            if self.sbr == 0:
                A = np.zeros([self.num_hotspot, 1]) + 0.02
            else:
                A = np.zeros([self.num_hotspot, 1]) + 0.55
        X = np.arange(self.mapsize[0])
        Y = np.arange(self.mapsize[1])
        XX, YY = np.meshgrid(X, Y)

        P = np.array([XX.reshape((-1)), YY.reshape(-1)]).T  # this is the map

        D = np.zeros(np.prod(self.mapsize))

        for h in range(self.num_hotspot):
            inner_f = np.repeat(mu[h:(h+1), :], np.prod(self.mapsize), 0) 
            inner = (inner_f - P) ** 2
            # D = D + A[h].item() * np.exp(-np.sum(inner, axis=1) / (r[h] * r[h]))
            D = np.maximum(D, A[h].item() * np.exp(-np.sum(inner, axis=1) / (r[h] * r[h])))
        np.random.seed(self.seed)
        return D, mu, r, A 
    
    def simulate_contaminate_hotspot(self, D):
        """Add hotspots from the contaminate
        Args:
            D: [imh * imw]. The map that contains the hotspot
        Ops:
            Add one hotspot on top of D which correspond to the contaminate
        """
        if len(D) == 0:
            D = np.zeros(np.prod(self.mapsize))
        map_size_array = np.reshape(self.mapsize, [1, len(self.mapsize)])
        num_hotspot = 1
        mu = np.repeat(map_size_array, num_hotspot, 0) * np.random.rand(num_hotspot, 2) # [num_hotspots, 2], where the hotspot is 
        # The size of the hotspots
        # r = 5 * np.random.rand(self.num_hotspot, 1) + 2 
        r = self.hotspot_size * np.random.rand(num_hotspot, 1) + 2.5
        
        # A = np.zeros([num_hotspot, 1]) + 0.4 #np.random.uniform(0.05, 0.1, [num_hotspot, 1]) # what is this though? The scale of the hotspots
        A = np.random.uniform(0.5, 0.6, [num_hotspot, 1])
        X = np.arange(self.mapsize[0])
        Y = np.arange(self.mapsize[1])
        XX, YY = np.meshgrid(X, Y)

        P = np.array([XX.reshape((-1)), YY.reshape(-1)]).T  # this is the map
        for h in range(num_hotspot):
            inner_f = np.repeat(mu[h:(h+1), :], np.prod(self.mapsize), 0) 
            inner = (inner_f - P) ** 2
            D = np.maximum(D, A[h].item() * np.exp(-np.sum(inner, axis=1) / (r[h] * r[h])))
        np.random.seed(self.seed)
        return D, mu, r, A 
    
    def test_hotspot(self):
        out_signal = self.simulate_hotspot()
        out_contain = self.simulate_contaminate_hotspot(out_signal[0])
        title_group = ["signal", "contaminate"]
        out_g = [out_signal, out_contain]
        name_use = ["Hotspots location", "Size", "Scale"]
        for j in range(3):
            print("--------------------------------")
            print(name_use[j])
            for i, s_title in enumerate(title_group):
                print(s_title, out_g[i][j+1])
            print("--------------------------------")
        fig = plt.figure(figsize=(7, 4))
        for i, s_out in enumerate([out_signal[0], out_contain[0]]):
            ax = fig.add_subplot(1, 2, i+1)
            ax.imshow(np.reshape(s_out, self.mapsize))     
            ax.set_title(title_group[i])
               
        return out_signal, out_contain 
        
    def simulate_signal(self, D, BB):
        """
        Args:
            D: [imh * imw], the SERS map that defines the hotspot
        """
        Vp = self.simulate_pseudo_voigt(np.arange(self.Nw), self.c, self.gamma, self.eta)  # [num_peaks, wavenumber]
        amplitude_act = calculate_act_amplitude(self.sbr, Vp, D, BB)
        if self.hotspot_size >= 10:
            amplitude_act = amplitude_act * 4
        A = np.repeat([np.ones([self.K]) * amplitude_act], np.prod(self.mapsize), axis=0) * np.repeat(D[:, np.newaxis], self.K, axis=-1)
        DD = A @ Vp  # [imh * imw, wavenumber]
        # print("The amplitude", alpha, "The max for the spectra", np.max(Vp), np.min(Vp), "The signal", np.max(DD), np.min(DD))
        np.random.seed(self.seed)
        return DD, Vp, amplitude_act
    
    def add_containminate(self, D, BB, c, gamma, eta, amplitude):
        V_con = self.simulate_pseudo_voigt(np.arange(self.Nw), c, gamma, eta)
        amplitude_act = calculate_act_amplitude(amplitude, V_con, D, BB)
        A = np.repeat([np.ones([1]) * amplitude_act], np.prod(self.mapsize), axis=0) * np.repeat(D[:, np.newaxis], 1, axis=-1)
        DD_con = A @ V_con
        return DD_con, V_con, A
    
    def test_contaminate_spectra(self, c, gamma, eta, amplitude, tds_dir, save=False):
        D_sig, mu_sig, r_sig, s_sig = self.simulate_hotspot()
        DD_sig, Vp_sig, A_sig = self.simulate_signal(D_sig)
        
        D_con, mu_con, r_con, s_con = self.simulate_contaminate_hotspot([])
        DD_con, Vp_con, A_con = self.add_containminate(D_con, c, gamma, eta, amplitude)
        
        utils.show_sers_map_and_spectra(D_sig, mu_sig, r_sig, DD_sig, self.mapsize, "Pure signal", tds_dir=tds_dir, save=save)
        utils.show_sers_map_and_spectra(D_con, mu_con, r_con, DD_con, self.mapsize, "Contaminate", tds_dir=tds_dir, save=save)
        utils.show_sers_map_and_spectra(D_sig + D_con, np.concatenate([mu_sig, mu_con], axis=0), 
                                  np.concatenate([np.expand_dims(r_sig, 1), r_con], axis=0), DD_sig + DD_con, self.mapsize, 
                                  "Combined", tds_dir=tds_dir, save=save)
        utils.compare_signal_with_contaminate(DD_sig, DD_con, self.mapsize, tds_dir=tds_dir, save=save) 
        
    def forward(self, background_method, background_parameters, contaminate_parameter=[]):
        """Generate spectra
        Args:
            contaminate_parameter: [c, gamma, eta, amplitude]
        Ops:
            1. simulate background
            2. simulate noise 
            3. simulate signal spectra 
            4. if the length of the contaminate_parameter > 0, simulate contaminate spectra 
            5. add everything together
        """
        np.random.seed(self.seed)
        BB, _, _ = self.simulate_background(np.arange(self.Nw), background_method, background_parameters)
        noise = self.simulate_noise()
        
        D_signal, mu_signal, r_signal, s_signal = self.simulate_hotspot()
        DD_signal, Vp, amplitude_act = self.simulate_signal(D_signal, BB)
        act_sbr = np.sum(DD_signal ** 2) / np.sum(BB ** 2)
        if len(contaminate_parameter) > 0:
            D_contaminate, mu_contaminate, r_contaminate, s_contaminate = self.simulate_contaminate_hotspot([])
            DD_contaminate, _, _ = self.add_containminate(D_contaminate, BB, contaminate_parameter[0], contaminate_parameter[1], contaminate_parameter[2], contaminate_parameter[3])
        else:
            DD_contaminate = np.zeros_like(DD_signal)
            mu_contaminate, r_contaminate, s_contaminate = [[] for _ in range(3)]
        DD = DD_signal + DD_contaminate
        X = DD + BB + noise        
        return X, DD, BB, [mu_signal, r_signal, s_signal, D_signal, act_sbr, amplitude_act], [mu_contaminate, r_contaminate, s_contaminate], Vp

    def forward_test(self, background_method, background_parameters, contaminate_parameter=[]):
        np.random.seed(self.seed)
        BB, background, background_ratio = self.simulate_background(np.arange(self.Nw), background_method, background_parameters)
        noise = self.simulate_noise()
        
        D_signal, mu_signal, r_signal, s_signal = self.simulate_hotspot()
        DD_signal, Vp_signal, A_signal = self.simulate_signal(D_signal)
        
        if len(contaminate_parameter) > 0:
            D_contaminate, mu_contaminate, r_contaminate, s_contaminate = self.simulate_contaminate_hotspot()
            DD_contaminate, _, _ = self.add_containminate(D_contaminate, contaminate_parameter[0], contaminate_parameter[1], contaminate_parameter[2], 3)
        else:
            DD_contaminate = np.zeros_like(DD_signal)
        DD = DD_signal + DD_contaminate
        X = DD + BB + noise
        background_group = [BB, background, background_ratio]
        signal_group = [D_signal, mu_signal, r_signal]
        contaminate_group = [D_contaminate, mu_contaminate, r_contaminate]
        return X, background_group, signal_group, contaminate_group, noise
    
    

def calculate_act_amplitude(sbr, vp, hotspot_map, background):
    """Caculate the actual amplitude used for generating the signal
    sbr: single value 
    vp: [num_peaks, Nw]
    hotspot_map: [h * w]
    background: [h * w, Nw]    
    """
    vp = np.mean(vp, axis=0, keepdims=True)
    hotspot_map = np.expand_dims(hotspot_map, axis=1)
    bottom = np.sum(background ** 2)
    upper_wo_amplitude = np.sum((hotspot_map * vp) ** 2)
    amplitude = sbr * bottom / upper_wo_amplitude
    return np.sqrt(amplitude * 0.25)
    
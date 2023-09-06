
### Vision Transformer on SERS maps

This repository provides the implementation for our paper [Nitroaromatic explosives detection and quantification using attention-based transformer on surface-enhanced Raman spectroscopy maps](http://mikkelschmidt.dk/papers/li2023analyst.pdf). We experimentally demonstrated that we significantly outperform or are on par with the existing approaches for explosives detection and quantification using raw SERS maps as input. We also open-source three SERS datasets measured down to very low concentrations: 4-NBT, 4-dinitrophenyl hydrazine (DNP), and Picric acid (PA) at https://data.dtu.dk/articles/dataset/Datasets_for_replicating_paper_Nitroaromatic_Explosives_detection_and_quantification_using_Attention-based_Transformer_on_surface-enhanced_Raman_spectroscopy_maps_/21602928.

  

#### Requirement
```bash
git clone https://github.com/lyn1874/molecule_detection_quantification.vit
cd molecule_detection_quantification
conda env create -f molecule_env.yml 
conda activate torch_dl
```

#### Data preparation
Download the dataset into folder `datasets/`

#### Model preparation

Download the model checkpoints into `exp_data/VIT/`

#### Model testing 

Please look at the file `test_experiment.ipynb` for evaluating the experiment

#### Model training 

###### ViT-based explosives detection and quantification
```python
./run_multiple_vit.sh detection quantification dataset gpu_index 
Args:
	detection: bool variable, true/false 
	quantification: bool variable, true/false
	dataset: str variable, TOMAS/DNP/PA
	gpu_index: int, which gpu to use
```

For example, to run a detection experiment on dataset DNP, you can simply run:
`./run_multiple_vit.sh true false DNP 0` 

###### Spectra-based explosive detection and quantification
```python
./run_multiple_spectra.sh dataset model_group detection quantification version percentage gpu_index 
Args:
	dataset: str varialbe, TOMAS/DNP/PA 
	model_group: str variable, xception/unified_cnn/resnet 
	detection: bool variable, true/false 
	quantification: bool variable, true/false 
	version: int 
	percentage: float, 0.002/0.005/0.01/0.02/0.05/0.1/0.2/0.5/1.0 
	gpu_index: int 
```

For example, to run a detection experiment on dataset PA with model xception, and you choose to average each SERS map with the Top-2% of the spectra that have the highest peak intensity:
	`./run_multiple_spectra.sh PA xception true false 0 0.002 0`


#### SERS map generation

The processes for generating SERS maps are shown in the jupyter file `SERS_maps_generation.ipynb`

#### Citation
If you use this code, please cite:
```
@Article{D3AN00446E,
author ="Li, Bo and Zappalá, Giulia and Dumont, Elodie and Boisen, Anja and Rindzevicius, Tomas and Schmidt, Mikkel N. and Alstrøm, Tommy S.",
title  ="Nitroaromatic explosives’ detection and quantification using an attention-based transformer on surface-enhanced Raman spectroscopy maps",
journal  ="Analyst",
year  ="2023",
publisher  ="The Royal Society of Chemistry",
url  ="http://dx.doi.org/10.1039/D3AN00446E",
}
```



# Data/A3D3_Hackathon_230919
Two-photon calcium imaging dataset
https://drive.google.com/drive/folders/1MVDSZs-6K-tGb79wU9wHuQ2Veoho-LiP?usp=sharing

## Neural signals
### How it is made
Raw two-photon calcium images are processed using suite2p. It extracts fluorescence traces of all the segmented neurons.\
    + https://suite2p.readthedocs.io/en/latest/outputs.html
    + https://www.biorxiv.org/content/10.1101/061507v2.abstract
### Files
+ ffneu_z_sel.npy
    + fluoresence traces. (num_neuron, len_seq_neural)
    + f-fneu*0.7. f:fluorescence traces, fneu:neuropil fluorescence traces. 
    + Preprocess: Z-scored (z) and neurons with high possibility are selected (sel).
+ spks_z_sel.npy
    + deconvolved traces. Spikes. (num_neuron, len_seq_neural)
    + Deconvolution means what?: https://suite2p.readthedocs.io/en/latest/FAQ.html
    + Preprocess: Z-scored (z) and neurons with high possibility are selected (sel).
### Which one to use?
+ The stream of data processing can be represented as: Raw two-photon calcium images -> ffneu -> spks.
+ In Dadarlatlab's preliminary research, only spks worked for decoding (mapping spks->limb coordinates). So I would recommend to start with spks first.
It seems that ffneu is too noisy to relate to behavior, or we don't have enough big size of data.
+ But our future plan includes to use raw calcium images for real-time application.
+ Raw calcium images are too large so not included here. Ask Seungbin if you want files.

## Behavior: Limb coordinates
### How it is made
Limb coordinates were predicted from video frames using Deeplabcut.\
+ http://www.mackenziemathislab.org/deeplabcut
### Files
+ behav_coord_likeli.npy
    + Continuous limb coordinates of a running mouse. (8, len_seq_behav). 8 = x and y coordinate for each four limb (right front, right hind, left front, left hind)
    + Preprocess: Coordinates with low likelihood from Deeplabcut (reference Deeplabcut pages) were replaced to interpolated values with neighboring coordinates (likeli).
+ behav_coord_likeli_norm.npy
    + Preprocess: Min-max scaled (norm).

## idx_coord_neural - Matching problem



# MODELS in progress
1) GNN for Classification
2) Transformer for reconstruction and subsequent classification
3) Encoder for limb ---> neuron

Neuroscience group codebase for A3D3 2023 Hackathon
![data_neuro_1](https://github.com/a3d3-hackathon/neuro_hack_2023/assets/102822547/3cfd43e5-f496-475b-bc67-91aa14c44ee7)
![data_neuro_2](https://github.com/a3d3-hackathon/neuro_hack_2023/assets/102822547/6268770b-af83-49f9-b096-7ff3f6716358)







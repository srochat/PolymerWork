# Polymer Data Extraction
This repository contains the data, code and models used in the article titled "Named-Entity Recognition for Polymer Scientific Literature Using Hybrid Deep Learning Architectures" by B. Joseph and S. Rochat.
## Data
The Data folder contains the two data sets used in the investigation, namely the Polymer Abstracts from Shetty et al.(https://github.com/Ramprasad-Group/polymer_information_extraction/tree/main/data/PolymerAbstracts) and the PolyIE dataset by Cheung et al. (https://github.com/jerry3027/PolyIE).
## Code
The other folders contain the models investigated. Where a BERT model is involved, the presented code uses "bert-base-uncased", which can simply be switched with other BERT models such as SciBERT ("allenai/scibert_scivocab_uncased"), MatSciBERT ("m3rg-iitd/matscibert"), or MaterialsBERT ("pranav-s/MaterialsBERT").
Some of the more complex implementations contain a debug-mode version that can be enabled by modifying the code; similarly, some implementations contain a toggle to enable or disable CRF.

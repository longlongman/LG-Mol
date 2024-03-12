# LG-Mol: Drug Property Prediction by Integrating Local and Global Information
We introduce LG-Mol, a novel framework that seamlessly integrates local and global information.
It employs a tailored Transformer architecture for atomic feature extraction and a graph neural network with equivariance properties to analyze the electrostatic potential surface.

![image](https://github.com/longlongman/LG-Mol/assets/18597120/fabd4def-ee9b-462e-9991-557913e853e9)

## Install
LG-Mol is built upon the foundation of Uni-Mol.
Therefore, you need to install Uni-Mol first.
For instructions on installing Uni-Mol, please refer to the [Uni-Mol installation guide](https://github.com/dptech-corp/Uni-Mol/tree/main/unimol).
After installing Uni-Mol, move the files from the unimol folder (in this repository) to the corresponding folder within the installed Uni-Mol directory.

## Data
We provide the code for data preprocessing in the data folder.
The preprocessed data is also available [here](https://drive.google.com/drive/folders/1Vvf6giqU929PaQ7y1AT-N6UwrGlrWmK5?usp=drive_link).

## Usage
The scripts required to run LG-Mol on the MoleculeNet benchmark are provided in the scripts folder. 
Here is an example:
```bash
bash molecular_property_prediction_bbbp_h.sh (fill up you own data path first)
```

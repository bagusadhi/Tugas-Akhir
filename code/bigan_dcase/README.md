# dcase2022_task2_bigan

In this DCASE 2022 Task 2 challenge, the BiGAN approach is based on this paper [Efficient GAN-Based Anomaly Detection](https://arxiv.org/pdf/1802.06222.pdf)

## Description
This system consists of two main scripts:
- `00_train.py`
  - "Development" mode: 
    - This script trains a model for each machine type by using the directory `dev_data/<machine_type>/train/`.
  - "Evaluation" mode: 
    - This script trains a model for each machine type by using the directory `eval_data/<machine_type>/train/`. (This directory will be from the "additional training dataset".)
- `01_test.py`
  - "Development" mode:
    - This script makes a csv file for each section including the anomaly scores for each wav file in the directories `dev_data/<machine_type>/source_test/` and `dev_data/<machine_type>/target_test/`.
    - The csv files are stored in the directory `result/`.
    - It also makes a csv file including AUC, pAUC, precision, recall, and F1-score for each section.
  - "Evaluation" mode: 
    - This script makes a csv file for each section including the anomaly scores for each wav file in the directories `eval_data/<machine_type>/source_test/` and `eval_data/<machine_type>/target_test/`. (These directories will be from the "evaluation dataset".)
    - The csv files are stored in the directory `result/`.

## Usage

### 1. Clone repository
Clone this repository from Github.

### 2. Download datasets
We will launch the datasets in three stages. 
So, please download the datasets in each stage:
- "Development dataset"
  - Download `dev_data_<machine_type>.zip` from https://zenodo.org/record/4562016. 
- "Additional training dataset", i.e. the evaluation dataset for training
  - After April. 1, 2021, download `eval_data_train_<machine_type>.zip` from https://zenodo.org/record/4660992.
- "Evaluation dataset", i.e. the evaluation dataset for test
  - After June. 1, 2021, download `eval_data_test_<machine_type>.zip` from https://zenodo.org/record/4884786.

### 3. Unzip dataset
Unzip the downloaded files and make the directory structure as follows:
- /dcase2021_task2_bigan
    - bigan
    - data
    - gan
    - utils
    - /00_train.py
    - /01_test.py
    - /common.py
    - /baseline.yaml
    - /readme.md
- /dev_data
    - /fan
        - /train (Normal data in the **source** and **target** domains for all sections are included.)
            - /section_00_source_train_normal_0000_<attribute>.wav
            - ...
            - /section_00_source_train_normal_0999_<attribute>.wav
            - /section_00_target_train_normal_0000_<attribute>.wav
            - /section_00_target_train_normal_0001_<attribute>.wav
            - /section_00_target_train_normal_0002_<attribute>.wav
            - /section_01_source_train_normal_0000_<attribute>.wav
            - ...
            - /section_02_target_train_normal_0002_<attribute>.wav
        - /test (Normal and anomaly data in the **source** and **target** domains for all sections are included.)
            - /section_00_source_test_normal_0000.wav
            - ...
            - /section_00_source_test_normal_0099.wav
            - /section_00_source_test_anomaly_0000.wav
            - ...
            - /section_00_source_test_anomaly_0099.wav
            - /section_01_source_test_normal_0000.wav
            - ...
            - /section_00_target_test_anomaly_0099.wav
            - /section_01_target_test_normal_0000.wav
            - ...
            - /section_02_target_test_anomaly_0099.wav
    - /gearbox (The other machine types have the same directory structure as fan.)
    - /bearing
    - /slider
    - /valve
    - /ToyCar
    - /ToyTrain
- /eval_data (Add this directory after launch)
    - /fan
        - /train (Unzipped additional training dataset. Normal data in the **source** and **target** domains for all sections are included.)
            - /section_03_source_train_normal_0000_<attribute>.wav
            - ...
            - /section_03_source_train_normal_0999_<attribute>.wav
            - /section_03_target_train_normal_0000_<attribute>.wav
            - /section_03_target_train_normal_0001_<attribute>.wav
            - /section_03_target_train_normal_0002_<attribute>.wav
            - /section_04_source_train_normal_0000_<attribute>.wav
            - ...
            - /section_05_target_train_normal_0002_<attribute>.wav
        - /test (Unzipped additional training dataset. Normal and anomaly data in the **source** and **target** domains for all sections are included.)
            - /section_03_source_test_0000.wav
            - ...
            - /section_03_source_test_0199.wav
            - /section_04_source_test_0000.wav
            - ...
            - /section_03_target_test_0000.wav
            - ...
            - /section_03_target_test_0199.wav
            - /section_04_target_test_0000.wav
            - ...
            - /section_05_target_test_0199.wav
    - /gearbox (The other machine types have the same directory structure as fan.)
    - /pump
    - /slider
    - /valve
    - /ToyCar
    - /ToyTrain

### 4. Change parameters
You can change parameters for feature extraction and model definition by editing `baseline.yaml`.

### 5. Feature extraction for training data (for the development dataset)
Run the training script `00_train.py`. 
Use the option `-d` for the development dataset `dev_data/<machine_type>/train/`. This step will generate an npy file for training data.

Example :

data : x_train_{machine_type}_section_{section_idx}.npy

label : y_train_{machine_type}_section_{section_idx}.npy
```
$ python3.6 00_train.py -d
```

### 6. Feature extraction for testing data (for the development dataset)
Run the test script `01_test.py`.
Use the option `-d` for the development dataset `dev_data/<machine_type>/test/`. This step will generate an npy file for testing data.

Example :

data : x_test_{machine_type}_section_{section_idx}.npy

label : y_test_{machine_type}_section_{section_idx}.npy
```
$ python3.6 01_test.py -d
```
Options:

| Argument                    |                                   | Description                                                  | 
| --------------------------- | --------------------------------- | ------------------------------------------------------------ | 
| `-h`                        | `--help`                          | Application help.                                            | 
| `-v`                        | `--version`                       | Show application version.                                    | 
| `-d`                        | `--dev`                           | Mode for the development dataset                             |  
| `-e`                        | `--eval`                          | Mode for the additional training and evaluation datasets     | 

`00_train.py` trains a model for each machine type and store the trained models in the directory `model/`.



### 7. Start Training and Testing BiGAN
For start training, run this code below:
```
$python3 main.py <gan, bigan> <mnist, kdd> run --nb_epochs=<number_epochs> --w=<float between 0 and 1> --m=<'cross-e','fm'> --d=<int> --rd=<int>
```

To reproduce the results of BiGAN, please use w=0.1 (as in the original BiGAN paper which gives a weight of 0.1 to the discriminator loss), d=1 for the feature matching loss.

### 8. Check results
The AUC and pAUC value of the BiGAN training and testing results can be seen in the results folder


## Dependency
We develop the source code on Ubuntu 16.04 LTS and 18.04 LTS.
In addition, we checked performing on **Ubuntu 16.04 LTS**, **18.04 LTS**, **CentOS 7**, and **Windows 10**.

### Software packages
- p7zip-full
- Python == 3.6.5
- FFmpeg

### Python packages
- Keras                         == 2.3.0
- Keras-Applications            == 1.0.8
- Keras-Preprocessing           == 1.0.5
- matplotlib                    == 3.0.3
- numpy                         == 1.18.1
- PyYAML                        == 5.1
- scikit-learn                  == 0.22.2.post1
- scipy                         == 1.1.0
- librosa                       == 0.6.0
- audioread                     == 2.1.5 (more)
- setuptools                    == 41.0.0
- tensorflow                    == 1.15.0
- tqdm                          == 4.43.0

## Citation
- Houssam Zenati, Chuan-Sheng Foo, Bruno Lecouat, Gaurav Manek, Vijay Ramaseshan Chandrasekhar, "Efficient GAN-Based Anomaly Detection." [URL](https://arxiv.org/pdf/1802.06222.pdf)
- Yohei Kawaguchi, Keisuke Imoto, Yuma Koizumi, Noboru Harada, Daisuke Niizumi, Kota Dohi, Ryo Tanabe, Harsh Purohit, and Takashi Endo, "Description and Discussion on DCASE 2021 Challenge Task 2: Unsupervised Anomalous Sound Detection for Machine Condition Monitoring under Domain Shifted Conditions," in arXiv e-prints: 2106.04492, 2021. [URL](https://arxiv.org/abs/2106.04492)
- Noboru Harada, Daisuke Niizumi, Daiki Takeuchi, Yasunori Ohishi, Masahiro Yasuda, Shoichiro Saito, "ToyADMOS2: Another Dataset of Miniature-Machine Operating Sounds for Anomalous Sound Detection under Domain Shift Conditions," in arXiv e-prints: 2106.02369, 2021. [URL](https://arxiv.org/abs/2106.02369)
- Ryo Tanabe, Harsh Purohit, Kota Dohi, Takashi Endo, Yuki Nikaido, Toshiki Nakamura, and Yohei Kawaguchi, "MIMII DUE: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection with Domain Shifts due to Changes in Operational and Environmental Conditions," in arXiv e-prints: 2105.02702, 2021. [URL](https://arxiv.org/abs/2105.02702)

# Conv-SNN
### Convolutional spiking neural networks (SNN) for spatio-temporal feature extraction
This paper highlights potentials of Convolutional spiking neural networks and introduces a new architecture to tackle training deep convolutional SNN problems.

## Prerequisites
The Following Setup is tested and it is working:
- Python>=3.5
- Pytorch>=0.4.1
- Cuda>=9.0
- opencv>=3.4.2

## Data preparation
- Download CIFAR10-DVS dataset
    + Extract the dataset under DVS-CIFAR10/dvs-cifar10 folder
    + Use test_dvs.m in matlab to convert events into matrix of ```t, x, y, p``` (make sure to adjust the test_dvs.m folder addresses inside the code) 
    + Run ```python3 dvscifar_dataloader.py``` to prepare the dataset (make sure to have files like dvs-cifar10/airplane/0.mat inside main.py directory)

## Training & Testing
- DVS-CIFAR10 model
    + Run ```python3 main.py```


- Spatio-temporal feature extraction tests
    + For each architecture simply run main file with python3


- Note: There are problems with training SNNs, such as extreme importance of initialization; Therefore, you may not reach the highest accuracy as mentioned in the paper. 
The solution is to try other torch versions and parameters or contact me / make an issue if you truly need the highest accuracy.

## Citing
Please adequately refer to the papers any time this Work is being used. If you do publish a paper where this Work helped your research, Please cite the following papers in your publications.

	@misc{samadzadeh2020convolutional,
            title={Convolutional Spiking Neural Networks for Spatio-Temporal Feature Extraction},
            author={Ali Samadzadeh and Fatemeh Sadat Tabatabaei Far and Ali Javadi and Ahmad Nickabadi and Morteza Haghir Chehreghani},
            year={2020},
            eprint={2003.12346},
            archivePrefix={arXiv},
            primaryClass={cs.CV}
        }

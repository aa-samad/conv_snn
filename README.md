# Conv-SNN
### Convolutional spiking neural networks (SNN) for spatio-temporal feature extraction
This paper highlights potentials of Convolutional spiking neural networks and introduces a new architecture to tackle training deep convolutional SNNs

## Prerequisites
The Following Setup is tested and working:
- Python>=3.5
- Pytorch>=0.4.1
- Cuda>=9.0
- opencv>=3.4.2

## Testing & Training
- DVS-CIFAR10 model
    + download DVS-cifar10 dataset
    + extract the dataset in the DVS-CIFAR10 under DVS-CIFAR10 folder
    + run ```python3 dvscifar_dataloader.py``` to prepare the dataset
    + run ```python3 main.py```

- Spatio-temporal feature extraction tests
    - for each architecture simply run main file with python3

## Citing
Please adequately refer to the papers any time this Work is being used. If you do publish a paper where this Work helped your research, Please cite the following papers in your publications.

	@inproceedings{...,
	  title={...},
	  author={...},
	  year={...},
	  organization={...}}

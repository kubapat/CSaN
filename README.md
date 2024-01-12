# CSaN

## Description
This is a simple runner demonstrating the Neural Network Hardware Acceleration using the CUDA technology by training a model on the both CPU and GPU and comparing the performance results.
## Requirements
* Python 3.9, other versions were not tested
* Pip
* Anaconda, all necessary packages can be installed using conda
* CUDA 11.3 (compatable with tensorflow 2.10, that is the last version for native Windows)
* CUDNN - should be compatable with your CUDA

## Installation
To install the required packages, you need to run the following command in the terminal:
```
conda env create -f environment.yml
```

## Usage
To run the program, you need to run the following command in the terminal:
```
python CSaN.py
```
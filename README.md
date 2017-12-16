# Point Cloud Segmentation
Project for Shapenet Segmentation Challenge, 2017

## Description
This is an implementation of the KD-Tree network method used for the [challenge](https://shapenet.cs.stanford.edu/iccv17/) in Keras.

Uses inspiration from [Escape from Cells: Deep Kd-Networks for the Recognition of 3D Point Cloud Models](https://arxiv.org/abs/1704.01222) for pre-processing the data.
The architecture is based on the famous [U-Net](https://arxiv.org/abs/1505.04597)

Official report available [here](https://arxiv.org/pdf/1710.06104.pdf)

## Usage

The data provided by the organizers must be extracted into a folder 'data'.
* prepare_data.py - processes data and packages them into numpy arrays
* model.py - defines and trains the model
* generate_segs.py - to generate labels for test/validation models post training

# Concrete Crack Image Dataset using Transfer Learning

Creating a prediction for concrete crack image dataset using transfer learning method in deep learning

### Description
Objective: Create a prediction using transfer learning method
for concrete crack image dataset

* Model training - Deep learning
* Method: Sequential, MobileNetV2, GlobalAveragePooling2D, Dense, Dropout
* Module: Sklearn & Tensorflow

In this analysis, dataset used from https://data.mendeley.com/datasets/5y9wdsg2zt/2

### About The Dataset:
The dataset used in this analysis is Concrete Crack Image for Classification dataset that contain 2 classes and 40000.

To do the classification using tranfer learning method, the dataset used will be split into 2 that is train and validation dataset which the validation will contain 20% of the image dataset.

### Deep learning model with Transfer Learning Method
A sequential model was created with 1 MobileNetV2 layer, 1 GlobalAveragePooling2D layer, 1 Dropout layer, 1 Dense Layer:
<p align="center">
  <img src="https://github.com/Ghost0705/Concrete_Crack_Using_Transfer_Learning/blob/main/Image/model_architecture_flow.png">
</p>

Model Archictecture when the trainable parameter is freeze
<p align="center">
  <img src="https://github.com/Ghost0705/Concrete_Crack_Using_Transfer_Learning/blob/main/Image/model_architecture_freezing.png">
</p>
The model archictecture above shows that the trainable parameter is freeze and the number that are trainable are 2,562

Model Archictecture when the trainable parameter is unfreeze
<p align="center">
  <img src="https://github.com/Ghost0705/Concrete_Crack_Using_Transfer_Learning/blob/main/Image/model_architecture_unfreeze.png">
</p>
When the trainable is unfreeze the trainable parameter value is change to 1,861,440 that is more trainable than the freezing model archictecture

The reason why we do this is because freezing and unfreezing these trainable parameters is a technique used in transfer learning, a process where a pre-trained model is used as a starting point for a new task.

Data were trained with 5 epoch:
When the Trainable Parameter is freeze
<p align="center">
  <img src="https://github.com/Ghost0705/Concrete_Crack_Using_Transfer_Learning/blob/main/Image/model_performance_freeze.png">
</p>

When the Trainable Parameter is unfreeze
<p align="center">
  <img src="https://github.com/Ghost0705/Concrete_Crack_Using_Transfer_Learning/blob/main/Image/model_performance_unfreeze.png">
</p>

After the deployment of both model we can see that the accuracy able to achieve 99% with a val_accuracy of 99%. The model is good enough to be used to predict the concrete crack image classification to see whether the concrete is in negative or positive. 

### How to run the pythons file:
1. Download the dataset
2. Run the concrete_crack.py 

Enjoy!


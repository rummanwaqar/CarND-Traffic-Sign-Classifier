# Project: Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

In this project, I used convolutional neural networks to classify traffic signs using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). 

---

The goals / steps of this project are the following:
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the results


[//]: # (Image References)

[dataset]: ./output_images/dataset.png "Dataset"
[counts]: ./output_images/counts.png "Counts"
[preprocessed]: ./output_images/preprocessed.png "Preprocessing"
[nn]: ./output_images/nn.svg "Neural Network"
[confusion]: ./output_images/confusion.png "Confusion Matrix"
[test_images]:./output_images/test_images.png "Test images"
[report]: ./output_images/report.png "Report"

---

### Data Set Summary & Exploration

Summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

![dataset]

The bar chart shows the distribution of training and validation images in each class. The dataset is unbalanced with 150 to 2000 images per class.

![counts]

### Design and Test a Model Architecture

#### Preprocessing dataset

1. Normalize the data-set to centre the data around *zero* mean. This helps the network to learn faster since gradients act uniformly.

2. Convert images from RGB to grayscale since the classification is not sign classification is not dependent on colour. This improved our accuracy.

Here is an example of a traffic sign image before and after preprocessing.

![preprocessed]

#### Model Architecture
![nn]

My final model consisted of the following layers:

| Layer           | Description                                 |
|-----------------|---------------------------------------------|
| Input           | 32x32x1 image                               |
| Convolution 5x5 | 1x1 stride, valid padding, outputs 32x32x30 |
| RELU            |                                             |
| Dropout         |                                             |
| Convolution 5x5 | 2x2 stride, same padding, outputs 14x14x30  |
| RELU            |                                             |
| Dropout         |                                             |
| Convolution 5x5 | 1x1 stride, valid padding, outputs 14x14x64 |
| RELU            |                                             |
| Max pooling     | 2x2 stride, outputs 5x5x64                  |
| Dropout         |                                             |
| Flatten         | outputs 1600                                |
| Fully connected | 1600-400                                    |
| RELU            |                                             |
| Dropout         |                                             |
| Fully connected | 400-84                                      |
| RELU            |                                             |
| Dropout         |                                             |
| Fully connected | 84-43                                       |

Model parameters:
* Adam optimizer
* Learning rate: 0.001
* Epochs: 20
* Batch size: 128

I started with the lenet-5 model. The table below shows my incremental steps towards the final model.

| Changes                                                | Accuracy |
|--------------------------------------------------------|----------|
| Generic Lenet                                          | 0.891    |
| Added normalization                                    | 0.902    |
| Added grayscale                                        | 0.909    |
| Changed conv layer depth to 30-64                      | 0.934    |
| Added dropout                                          | 0.959    |
| **Replaced 1st max pooling with conv layer with stride=2** | **0.963**    |

My final model results were:
* Validation set accuracy of 96.3%
* Test set accuracy of 94.7%


### Test a Model on New Images

The following test images were used:

![test_images]

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Bumpy road      		| Bumpy road   									|
| Road work     			| Road work 										|
| Right-of-way at next intersection | Right-of-way at next intersection|
| Children crossing	      		| Children crossing |
| Turn right ahead	| Turn right ahead	|

We got a 100% accuracy on the test images. For each of the five images, the model is very sure about the classification and the first class probability is close to 100%. This means that the model is very certain about the classification results.

### Test on test set

I tested the network on the provided test dataset. It contains 12630 images. We got an accuracy of 94.7%.

The following image shows the confusion matrix:

![confusion]

The following table shows a more detailed analysis:

![report]

---
layout: post
title: "Convolutional Neural Networks for Predicting Poverty in Sparse Data Settings"
subtitle: "Predicting household consumption in Malawi from satellite images with convolutional neural networks"
background: '/img/posts/cnn-poverty/sidebyside.png'
---

This project was my final coursework project at the University of Bonn in the winter semester of 2022/23, for the course ***Effective Programming for Economists***, which focused on writing efficient, reproducible code in Python. The goal was an application of machine learning to a real-world problem that could be built and documented within the four-month semester. At the time I was also a research assistant at the United Nations University Institute for Environment and Human Security in Bonn, which drew me towards a topic in human welfare.

The project builds on Jean et al. (2016), who combine deep learning with high-resolution daytime satellite imagery and nightlight data to estimate poverty in African countries. A key obstacle to tracking progress on poverty reduction is the lack of survey data in many of the regions most affected by poverty, because large-scale household surveys are slow and expensive to run. Progress on poverty reduction nevertheless depends on measuring it in a timely and reliable way. Jean et al. use Convolutional Neural Networks (CNNs) to fill this gap with cheap, abundant data as an alternative to household surveys.

## Setup

Satellite imagery does not directly capture differences in wealth between regions, so the approach uses transfer learning to extract economically relevant features from thousands of unstructured satellite images. Brighter nightlights typically indicate that a region is more developed. Nightlight data can therefore be used to train a model to recognize features in daytime imagery that are correlated with economic outcomes, such as roads, farmland, waterways, roofs and urban areas. In the final step, the trained model predicts village-level wealth as measured by traditional surveys.

![Two satellite images side by side: (a) an area with low nightlight intensity and (b) an area with high nightlight intensity](/img/posts/cnn-poverty/hl_ll.png)

My contribution was a reproducible workflow for the original study, showing how unconventional data can be integrated into economists' research and programming pipelines. The original replication kit and related GitHub repositories required considerable work to obtain and manage the data and to run several scripts in a particular order. That complexity makes it harder for new users to change model parameters and build on the authors' work. I addressed this by integrating all scripts and tasks into a single `pytask` workflow. Within it, I used the CNN method to estimate consumption expenditure in Malawi and compared the R² with the authors' model, documenting each programming step from data collection to prediction.

## Overview

The model training followed these steps, using nightlight data to predict consumption:

1. Use a Gaussian mixture model to separate satellite images into three nightlight classes (no, low and high).
2. Train a CNN to predict the nightlight classes in the training set (80%) and compute model accuracy in the validation set (20%).
3. Use the trained model to extract a feature vector for each image in the validation set, by taking the model output just before the final classification step.
4. Train a ridge regression on the principal components of the feature matrix and generate predicted consumption in the validation set.

In brief, transfer learning works as follows here: nightlights serve as a proxy for economic activity, and the CNN takes thousands of satellite images as input and learns to predict variation in nightlight intensity.

The model receives no human input indicating that these images are useful for predicting economic activity or consumption; it only learns to identify the features in satellite images that are relevant for predicting nightlights. The output of this step is a trained model that reduces high-dimensional image inputs to low-dimensional features relevant for predicting nightlights.

## Model Training

The training phase used a Gaussian mixture model to group the satellite images by nightlight intensity, and the CNN was trained to predict these classes. After training, the CNN transforms high-dimensional image data into feature vectors.

![Density of nightlight intensity for the low, medium and high classes found by the Gaussian mixture model](/img/posts/cnn-poverty/gmm.png)

The implementation uses PyTorch, which offers a range of CNN architectures, and the fine-tuning functions written by Inkawhich (2022). I modified the code so that the first two epochs perform fine-tuning, updating all model parameters, and later epochs perform feature extraction, updating only the output layer weights while keeping the pre-trained parameters fixed.

## Prediction and Results

The trained CNN then extracted a feature vector for each image in the validation set. These vectors are a nonlinear mapping of the input images, taken just before the classification step. The result is an n × p matrix, where n = 2,296 validation images and p = 4,096 features per image. The images were grouped by household-cluster coordinates and averaged within each cluster, which reduces the data to a 780 × 4,096 feature matrix X and an outcome vector Y of consumption per capita. The goal is to predict consumption expenditure from these image features rather than directly from nightlights.

Training models on such a high-dimensional dataset is slow and caused convergence problems for some models, so I first reduced the dimensionality with principal component analysis.

![Scatter plot of actual against predicted consumption from the CNN-ridge model, with an r-squared of 0.28](/img/posts/cnn-poverty/prediction.png)

Under both 5-fold and 10-fold cross-validation, models trained on CNN features generally have a lower R² than models trained on nightlight data. Some models, such as XGBoost, perform poorly on both nightlight data and CNN features, while random forests improve with CNN features. Ridge regression is among the best-performing models. The chart above plots actual consumption against values predicted by a cross-validated ridge regression using the first 30 principal components of the CNN features.

## Conclusion

The project shows how CNNs and satellite imagery can be used to predict poverty and how the workflow can be made reproducible. The final model explains about 28% of the variation in consumption in Malawi (R² = 0.28). The `pytask` workflow can be reused or adapted for other regions.

Satellite imagery offers a lower-cost complement to household surveys for measuring poverty, particularly where survey data are scarce.

## References

Inkawhich, Nathan (2022). "Finetuning Torchvision Models". PyTorch Tutorials. [https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html](https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html)

Jean, Neal, Marshall Burke, Michael Xie, W. Matthew Davis, David B. Lobell, and Stefano Ermon (2016). "Combining satellite imagery and machine learning to predict poverty". *Science* 353(6301), pp. 790–794.

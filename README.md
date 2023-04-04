# **Unstructured Data course**

## **Art Classifier**

This project was developed for the Unstructured Data course in the Big Data Master's Degree at Comillas ICAI University.

This project has been done by:

|Name                    |Email                              |
|------------------------|-----------------------------------|
|Jorge Ayuso Martínez    |jorgeayusomartinez@alu.comillas.edu|
|Carlota Monedero Herranz|carlotamoh@alu.comillas.edu        |
|José Manuel Vega Gradit |josemanuel.vega@alu.comillas.edu   |

The main goal of this project is building an Art Classifier leveraging Deep Learning techniques. The data we used belongs to the [WikiArt](https://www.wikiart.org/) project. The exact dataset used is available at [kaggle](https://www.kaggle.com/datasets/sivarazadi/wikiart-art-movementsstyles). This dataset is specifically tailored for building Deep Learning models that are able to classify different art-styles. For the sake of simplicity and disk space limitations, we will only be classifying 4 of the available movement styles: 

- #### Romanticism

<p align="left">
    <img src="docs/romanticism_example.jpg" width="500" height="400" />
<p>

- #### Realism

<p align="left">
    <img src="docs/realism_example.jpg" width="500" height="400" />
<p>

- #### Renaissance

<p align="left">
    <img src="docs/renaissance_example.jpg" width="300" height="450" />
<p>

- #### Baroque

<p align="left">
    <img src="docs/baroque_example.jpg" width="300" height="400" />
<p>

## Overview

We start by building a simple CNN architecture from scratch. This netwrok will serve as our base model. The main issue with this network is the significant amount of *overfitting* to the trainig data, which we try to reduce by using different techniques:

+ Dropout
+ Batch normalization
+ Data augmentation

Once our custom-built model is fully explored, we then shift our focus towards using transfer learning by combining the convolutional part of an off-the-shelf pre-trained network, which we will then combine with a fully connected classifier (DNN). For this purpose, we will be evaluating three different models:

+ ResNet50
+ VGG16 & VGG19
+ MobileNet

Finally, we will explore a different network architecture by using Huggingface's Transformers. 
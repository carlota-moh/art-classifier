# **Unstructured Data course**

## **Art Classifier**


The main goal of this project is to create an art classifier. It was developed for the Unstructured Data course in the Big Data Master's Degree at Comillas ICAI University.

This project has been done by:

|Name                    |Email                              |
|------------------------|-----------------------------------|
|Jorge Ayuso Martínez    |jorgeayusomartinez@alu.comillas.edu|
|Carlota Monedero Herranz|carlotamoh@alu.comillas.edu        |
|José Manuel Vega Gradit |josemanuel.vega@alu.comillas.edu   |

The main objective of this project is building an Art Classifier leveraging Deep Learning techniques. The data we used belongs to the [WikiArt](https://www.wikiart.org/) project. The exact dataset used is available at [kaggle](https://www.kaggle.com/datasets/sivarazadi/wikiart-art-movementsstyles). For the sake of simplicity and disk space limitations in our computers, we will only be classifying 4 of the available movement styles: Romanticism, Realism, Renaissance and Baroque. 

We will start by building a simple CNN architecture, which will serve as our base model for reference and comparision. We will then try to extend its capabilities by using a two-step approach:

+ First, we will increase complexity by increasing the number of hyperparameters in the network until accuracy in training set is high enough and clear signs of overfitting are observed (Notebook 1)

+ Secondly, we will reduce overfitting of the complex model using the following techniques: Data Augmentation, Batch-normalization and Dropout (Notebooks 2-4).

Once our custom.built model is fully explored, we will then shift our focus towards using transfer learning based on the convolutional part of an off-the-shelf pre-trained network, which we will then combine with a fully connected classifier (DNN). For this purpose, we will be evaluating two models:

+ ResNet50
+ VGG19
+ AlexNet

Finally, we will evaluate a different model architecture based on transformers.
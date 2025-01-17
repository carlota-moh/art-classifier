# Python base libraries
import os
import glob

# Data Science libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning and Deep Learning libraries
from sklearn.metrics import classification_report
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from transformers import ViTImageProcessor
from torchvision.transforms import (
    RandomResizedCrop,
    RandomHorizontalFlip,
    CenterCrop, 
    Compose, 
    Normalize, 
    Resize, 
    ToTensor
)

# Other libraries
from tqdm import tqdm


# Constants
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIGHTNING_LOGS_DIR = os.path.join(ROOT_DIR, 'lightning_logs')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
VAL_DIR = os.path.join(DATA_DIR, 'validation')

def drop_checkpoints(base_dir):
    """
    Drops the .ipynb_checkpoints hidden files from the specified dataset.

    Args:
    -------
    base_dir: str
        Path to the dataset directory.
    
    Returns:
    -------
    None
    """
    for path in os.walk(base_dir):
        for folder in path[1]:
            if ".ipynb_checkpoints" in folder:
                os.rmdir(os.path.join(path[0], folder))


# Create function to see dataset statistics, such as number of images per class
def dataset_stats(train_dir, validation_dir, test_dir):
    """
    Gets the number of images in each class and the total number of 
    images in the dataset, for both training, validation and test sets.

    Args:
    -------
    train_dir: str
        Path to the training set directory.
    validation_dir: str
        Path to the validation set directory.
    test_dir: str
        Path to the test set directory.
    
    Returns:
    -------
    None
    """
    # Number of classes
    n_classes = len(os.listdir(train_dir))
    print(f"Number of classes: {n_classes}")

    # Get existing classes
    classes = os.listdir(train_dir)
    print(f"Existing classes: {classes}\n")

    # Create dataframe to store number of classes per style and dataset
    df_stats = pd.DataFrame(columns=["Style", "Train", "Validation", "Test"])
    # Define empty lists to store number of classes per style
    styles = []
    train = []
    validation = []
    test = []

    # Loop through each class
    for cl in classes:
        styles.append(cl)
        train.append(len(os.listdir(os.path.join(train_dir, cl))))
        validation.append(len(os.listdir(os.path.join(validation_dir, cl))))
        test.append(len(os.listdir(os.path.join(test_dir, cl))))

    # Add data to dataframe
    df_stats["Style"] = styles
    df_stats["Train"] = train
    df_stats["Validation"] = validation
    df_stats["Test"] = test

    # Set index to style
    df_stats.set_index("Style", inplace=True)

    # Print dataframe
    print("-"*40)
    print("Number of images per class and dataset:")
    print("-"*40)
    print(df_stats)


def plot_metric_curves(epochs, train_curve, val_curve, train_color, val_color, metric):
    """
    Plots the training and validation curves for a specific metric.

    Args:
    -------
    epochs: list
        List of epochs.
    train_curve: list
        List of training metric values.
    val_curve: list
        List of validation metric values.
    train_color: str
        Color of the training curve.
    val_color: str
        Color of the validation curve.
    metric: str
        Metric to plot.
    
    Returns:
    -------
    None
    """
    sns.set_theme()
    plt.figure(figsize=(15,10), dpi=200)
    plt.plot(epochs, train_curve, color=train_color, linewidth=2, label=f'Training {metric.lower()}')
    plt.plot(epochs, val_curve, color=val_color, linewidth=2, label=f'Validation {metric.lower()}')
    plt.title(f'Training and validation {metric.lower()}', fontsize=20)
    plt.xlabel("Epochs", fontsize=15)
    plt.ylabel(metric.capitalize(), fontsize=15)
    plt.legend(frameon=False, fontsize=15)
    plt.show()
    return


def load_latest_checkpoint(model_class, logs_dir=LIGHTNING_LOGS_DIR):
    """
    Loads the latest checkpoint from the lightning_logs directory.

    Args:
    -----
    model_class: PyTorch Lightning model class
        The model class to use for loading the checkpoint.
    
    Returns:
    --------
    model: PyTorch Lightning model
        The model loaded from the latest checkpoint.
    """
    version_dirs = glob.glob(os.path.join(logs_dir, 'version_*'))
    latest_version_dir = max(version_dirs, key=os.path.getmtime)
    ckpt_files = glob.glob(os.path.join(latest_version_dir, 'checkpoints', '*.ckpt'))
    latest_ckpt_file = max(ckpt_files, key=os.path.getmtime)
    
    # Load the checkpoint into a new instance of the model class
    model = model_class.load_from_checkpoint(latest_ckpt_file)
    
    return model


# Create function to plot images to show data augmentation technique in a visual way
def data_augmentation_plot(data_generator, images):
    """
    Plots 2 images augmented with the specified data generator.

    Args:
    -------
    data_generator: Keras ImageDataGenerator
        Generator to use for data augmentation.
    images: numpy array
        Images to augment.

    Returns:
    -------
    None
    """
    data_generator.fit(images)
    image_iterator = data_generator.flow(images)
    
    # Plot the images given by the iterator
    fig, rows = plt.subplots(nrows=1, ncols=2, figsize=(25, 25))
    for row in rows:
        row.imshow(image_iterator.next()[0].astype('int'))
        row.axis('off')
    plt.show()


def extract_features(sample_count, dnn_input_shape, conv_base, generator, batch_size, n_classes=4):
    """
    Extracts features from the convolutional base of a pre-trained model.

    Args:
    -------
    sample_count: int
        Number of samples in the dataset.
    dnn_input_shape: tuple
        Shape of the input images.
    conv_base: Keras model
        Convolutional base of the pre-trained model.
    generator: Keras ImageDataGenerator
        Generator to use for extracting the features.
    batch_size: int
        Batch size to use for extracting the features.
    n_classes: int
        Number of classes in the dataset.

    Returns:
    -------
    features: numpy array
        Features extracted from the convolutional base.
    """
    features = np.zeros(shape=(sample_count, dnn_input_shape[0], dnn_input_shape[1], dnn_input_shape[2]))
    labels = np.zeros(shape=(sample_count, n_classes))
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            # Note that since generators yield data indefinitely in a loop,
            # we must `break` after every image has been seen once.
            break
    return features, labels


def get_pytorch_predictions_from_dataloader(model, dataloader):
    """
    Get predictions from a Pytorch model on a given dataloader.

    Args:
    -----
    model: PyTorch model
        The model to use for predictions.
    dataloader: PyTorch dataloader
        The dataloader to use for predictions.

    Returns:
    --------
    all_predictions: list
        List of predictions.
    all_targets: list
        List of targets.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    # Move model to device
    model.to(device)

    # Set model to evaluation mode and freeze it
    model.eval()
    model.freeze()

    # Lists to store predictions and targets
    all_predictions = []
    all_targets = []

    # Use a progress bar to show the progress of the predictions
    for batch in tqdm(dataloader):
        images, targets = batch
        images = images.to(device) # Move inputs to the same device as the model
        predictions = model(images)
        # Convert ImageClassifierOutput to tensor
        predictions = predictions.logits
        all_predictions.append(predictions.cpu())
        all_targets.append(targets.cpu())

    # Concatenate all predictions and targets
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    return all_predictions, all_targets


def get_vit_metrics(model, train=False):
    """
    Gets the metrics for the ViT model.

    Args:
    -----
    model: PyTorch model
        The ViT model to use for predictions.
    train: bool, optional (default=False)
        Whether to get the metrics for the training set.

    Returns:
    --------
    None
        The classification report for the ViT model is printed
        to the console for both the validation and test sets.
    """
    # Get the image processor and its parameters
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    img_size = processor.size
    img_mean = processor.image_mean
    img_std = processor.image_std

    transform = Compose([
        Resize(img_size['height']),
        CenterCrop(img_size['height']),
        ToTensor(),
        Normalize(mean=img_mean, std=img_std)
    ])

    # Get the classes from the model
    classes = model.id2label.values()

    # Get the dataloaders for the test and validation sets
    test_dataset = ImageFolder(TEST_DIR, transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    val_dataset = ImageFolder(VAL_DIR, transform=transform)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    if train:
        transform_train = Compose([
            RandomResizedCrop(img_size['height']),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(mean=img_mean, std=img_std)
        ])
        # Get the dataloader for the training set
        train_dataset = ImageFolder(TRAIN_DIR, transform=transform_train)
        train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=False)

        # Get the classification report for the training set
        predictions, targets = get_pytorch_predictions_from_dataloader(model, train_dataloader)
        print('Training set classification report:')
        print(classification_report(targets, predictions.argmax(dim=1), target_names=classes))

    # Get the classifcation report for the test set
    predictions, targets = get_pytorch_predictions_from_dataloader(model, test_dataloader)
    print('Test set classification report:')
    print(classification_report(targets, predictions.argmax(dim=1), target_names=classes))

    # Get the classification report for the validation set
    predictions, targets = get_pytorch_predictions_from_dataloader(model, val_dataloader)
    print('Validation set classification report:')
    print(classification_report(targets, predictions.argmax(dim=1), target_names=classes))

    return
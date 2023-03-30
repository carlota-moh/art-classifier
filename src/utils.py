import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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
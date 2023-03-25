import os
import shutil
import numpy as np

def get_n_images(directory):
    """
    Returns the number of images present in a directory.

    Args:
    ----------
    directory(str): 
        The directory path to count the number of images.

    Returns:
    ----------
    n_images(int): 
        The number of images present in the specified directory.
    """
    assert isinstance(directory, str), "Input folder path must be a string."
    assert os.path.exists(directory), f"No such file or directory: '{directory}'"
    
    n_images = 0
    for path in os.walk(directory):
        for file in path[2]:
            n_images += 1
    return n_images

def remove_images(directory, n_images, n_keep):
    """
    Removes a specified number of images from a directory at random.

    Args:
    ----------
    directory(str): 
        The directory path to remove images from.
    n_images(int): 
        The total number of images in the directory.
    n_keep(int): 
        The number of images to keep in the directory.
    """
    assert isinstance(directory, str), "Input directory path must be a string."
    assert isinstance(n_images, int) and isinstance(n_keep, int), "n_images and n_keep must be integers."
    assert n_images >= n_keep, "n_images must be greater than or equal to n_keep."
    assert n_images >= 0 and n_keep >= 0, "n_images and n_keep must be non-negative."
    assert os.path.exists(directory), f"No such file or directory: '{directory}'"
    
    # Get images to remove randomly
    images_to_remove = np.random.choice(np.arange(0, n_images), size=(n_images-n_keep), replace=False)
    n_image = 0
    for path in os.walk(directory):
        for file in path[2]:
            # Remove specified images
            if n_image in images_to_remove:
                image = os.path.join(path[0], file)
                os.remove(image)
            n_image += 1
            
def move_images(src_path, dest_path, n_images, n_move):
    """
    Moves a specified number of images from one directory to another at random.

    Args:
    ----------
    src_path(str): 
        The source directory path to move images from.
    dest_path(str): 
        The destination directory path to move images to.
    n_images(int): 
        The total number of images in the source directory.
    n_move(int): 
        The number of images to move from the source directory to the destination directory.
    """
    assert isinstance(src_path, str) and isinstance(dest_path, str), "Input directory paths must be strings."
    assert isinstance(n_images, int) and isinstance(n_move, int), "n_images and n_move must be integers."
    assert n_images >= n_move, "n_move must be less than or equal to n_images."
    assert n_images >= 0 and n_move >= 0, "n_images and n_move must be non-negative."
    assert os.path.exists(src_path), f"No such file or directory: '{src_path}'"
    
    os.makedirs(dest_path, exist_ok=True)
    
    # Get images to move randomly
    images_to_move = np.random.choice(np.arange(0, n_images), size=n_move, replace=False)
    n_image = 0
    for path in os.walk(src_path):
        for file in path[2]:
            if n_image in images_to_move:
                # Get original path and destination path where to move the image
                old_image = os.path.join(path[0], file)
                new_image = os.path.join(dest_path, file)
                # Move image from source dir to dest dir
                shutil.move(old_image, new_image)
            n_image += 1


if __name__=='__main__':
    # Define styles (folder names for each of them)
    styles_dir = [
        "data/Romanticism", 
        "data/Baroque", 
        "data/Realism", 
        "data/Renaissance"
    ]
    # Number of images to keep (therefore, we have balanced classes)
    n_keep = 5000
    # Split between train, validation and test
    sets = {
        "train": 4000, 
        "validation": 500, 
        "test": 500
        }

    root_dir = "../data"

    # Iterate for each style 
    for style in styles_dir:
        # Number of images we have initially in the source dir
        n = n_keep
        # Get number of images in the source dir
        n_images = get_n_images(style)
        # Remove images to have n_keep images in source dir
        remove_images(style, n_images, n_keep)
        for key, value in sets.items():
            # Get source and destination dirs
            src_path = style
            dest_path = os.path.join(root_dir, key, style)
            # Move "value" images from source to destination dir
            move_images(src_path, dest_path, n, value)
            # Now source dir has "value" images less
            n -= value
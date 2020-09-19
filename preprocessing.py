# import some common libraries
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import random


def get_image_ids(image_folder=None):
    """
    Explores a folder of images and gets their ID from their file name.
    Returns a list of all image ID's in image_folder.
    
    Params
    ------
    image_folder (str): path to folder of images, e.g. "../validation/"
    """
    return [
        os.path.splitext(img_name)[0]
        for img_name in os.listdir(image_folder)
        if img_name.endswith(".jpg")
    ]


# Make a function which formats a specific annotations csv based on what we're dealing with


# Make a function which formats a specific annotations csv based on what we're dealing with
def format_annotations(image_folder, annotation_file, target_classes=None):
    """
    Formats annotation_file based on images contained in image_folder.
    Will get all unique image IDs and make sure annotation_file
    only contains those (the target images).
    Adds meta-data to annotation_file such as class names and categories.
    If target_classes isn't None, the returned annotations will be filtered by this list.
    Note: image_folder and annotation_file should both be validation if working on
    validation set or both be training if working on training set.
    
    Params
    ------
    image_folder (str): path to folder of target images.
    annotation_file (str): path to annotation file of target images.
    target_classes (list), optional: a list of target classes you'd like to filter labels.
    """

    # Get all image ids from target directory
    image_ids = get_image_ids(image_folder)

    # Setup annotation file and classnames
    annot_file = pd.read_csv(annotation_file)
    classes = pd.read_csv(
        "class-descriptions-boxable.csv", names=["LabelName", "ClassName"]
    )

    # Create classname column on annotations which converts label codes to string labels
    annot_file["ClassName"] = annot_file["LabelName"].map(
        classes.set_index("LabelName")["ClassName"]
    )

    # Sort annot_file by "ClassName" for alphabetical labels (used with target_classes)
    annot_file.sort_values(by=["ClassName"], inplace=True)

    for s in tqdm(image_ids):
        # Make sure we only get the images we're concerned about
        if target_classes:
            annot_file = annot_file[
                annot_file["ImageID"].isin(image_ids)
                & annot_file["ClassName"].isin(target_classes)
            ]
        else:
            annot_file = annot_file[annot_file["ImageID"].isin(image_ids)]

        assert len(annot_file.ImageID.unique()) == len(
            image_ids
        ), "Label unique ImageIDs doesn't match target folder."

        # Add ClassID column, e.g. "Bathtub, Toilet" -> 1, 2
        annot_file["ClassName"] = pd.Categorical(annot_file["ClassName"])
        annot_file["ClassID"] = annot_file["ClassName"].cat.codes

    return annot_file

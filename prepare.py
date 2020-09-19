import os
import tqdm
from preprocessing import format_annotations, get_image_ids
from mungedata import yolo_box_data

subset_classes = ['Bathtub', 'Bed', 'Billiard table', 'Ceiling fan', 'Coffeemaker', 'Couch', 'Countertop', 'Dishwasher', 'Fireplace', 'Fountain',
                'Gas stove', 'Jacuzzi', 'Kitchen & dining room table', 'Microwave oven', 'Mirror', 'Oven', 'Pillow', 'Porch', 'Refrigerator', 'Shower',
                'Sink', 'Sofa bed', 'Stairs',  'Swimming pool', 'Television', 'Toilet', 'Towel', 'Tree house', 'Washing machine', 'Wine rack'] 

DATA_PATH = "/home/newlearningsrujan_gmail_com/dl-projects/airbnb_object_detection/dataset/"

# Define the two target classes we're working with
images_train_path = os.path.join(DATA_PATH, "images/train/")
images_valid_path = os.path.join(DATA_PATH, "images/valid/")
images_test_path = os.path.join(DATA_PATH, "images/test/")

label_test_path = os.path.join(DATA_PATH, "labels/test")
label_valid_path = os.path.join(DATA_PATH, "labels/valid")
label_train_path = os.path.join(DATA_PATH, "labels/train")


print("Forming Annotation dataframe for training images:")
train_annots_formatted = format_annotations(
    image_folder=images_train_path,
    annotation_file="train-annotations-bbox.csv",
    target_classes=subset_classes,
)

print("Forming Annotation dataframe for test images:")
test_annots_formatted = format_annotations(
    image_folder=images_test_path,
    annotation_file="test-annotations-bbox.csv",
    target_classes=subset_classes,
)


print("Forming Annotation dataframe for validation images:")
val_annots_formatted = format_annotations(
    image_folder=images_valid_path,
    annotation_file="validation-annotations-bbox.csv",
    target_classes=subset_classes,
)

print("Creating object boxes txt files for training images:")
yolo_box_data(train_annots_formatted, label_train_path)

print("Creating object boxes txt files for validation images:")
yolo_box_data(val_annots_formatted, label_valid_path)

print("Creating object boxes txt files for test images:")
yolo_box_data(test_annots_formatted, label_test_path)

#print("Creating image data details:")
#image_box_data(train_annots_formatted, label_train_path, images_train_path)
#image_box_data(train_annots_formatted, label_valid_path, images_valid_path)

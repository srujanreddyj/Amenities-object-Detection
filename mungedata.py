# import some common libraries
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
from preprocessing import format_annotations, get_image_ids


def yolo_box_data(df, label_path):
    for _, row in tqdm(df.iterrows(), total=len(df)):
        yolo_data = []
        image_id = row["ImageID"]
        class_id = row["ClassID"]
        x_center = (row["XMax"] + row["XMin"]) / 2
        y_center = (row["YMax"] + row["YMin"]) / 2
        box_width = row["XMax"] - row["XMin"]
        box_height = row["YMax"] - row["YMin"]
        yolo_data.append([class_id, x_center, y_center, box_width, box_height])
        yolo_data = np.array(yolo_data)

        try:
            # do the needed
            with open(os.path.join(label_path, image_id + ".txt"), "ab") as f:
                np.savetxt(f, yolo_data, fmt=["%d", "%f", "%f", "%f", "%f"])
                f.close()

        except:
            # generate the file
            np.savetxt(
                os.path.join(label_path, image_id + ".txt"),
                yolo_data,
                fmt=["%d", "%f", "%f", "%f", "%f"],
            )


def image_box_data(df, label_path, image_path):
    total_df = pd.DataFrame(
        columns=[
            "class_id",
            "x_center",
            "y_center",
            "box_width",
            "box_height",
            "height",
            "width",
        ]
    )
    for _, row in tqdm(df.iterrows(), total=len(df)):
        image_id = row["ImageID"]
        height, width = cv2.imread(image_path + image_id + ".jpg").shape[:2]
        class_id = row["ClassID"]
        x_center = (row["XMax"] + row["XMin"]) / 2
        y_center = (row["YMax"] + row["YMin"]) / 2
        box_width = row["XMax"] - row["XMin"]
        box_height = row["YMax"] - row["YMin"]
        total_df = total_df.append(
            pd.Series(
                [class_id, x_center, y_center, box_width, box_height, height, width],
                index=total_df.columns,
            ),
            ignore_index=True,
        )
    return total_df

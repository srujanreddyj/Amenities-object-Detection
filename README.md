# AMENITY OBJECT DETECTION
This is one of my favorite project. After doing pyimagesearch gurus course, I wanted something more. 
I came across this [medium article by Airbnb](https://medium.com/airbnb-engineering/amenity-detection-and-beyond-new-frontiers-of-computer-vision-at-airbnb-144a4441b72e). 

### DESCRIPTION
This is a an end-to-end Pytorch based Object Detection that is hosted on Flask, allowing users to boot up a local web application and upload their own photos and see how many amenities are detected in their homes or photos. This project was built following needs and directions mentioned in this [Airbnb article](https://medium.com/airbnb-engineering/amenity-detection-and-beyond-new-frontiers-of-computer-vision-at-airbnb-144a4441b72e). 


### Project Goal
Develop an end to end machine learning product useful for businesses. 

#### REQUIREMENT
Python 3.8 or later with all requirements.txt dependencies installed, including torch>=1.6. To install run:

```$ pip install -r requirements.txt```


#### DATASET
- - - - - - -
Downloaded images from [OPENIMAGES DATASET](https://storage.googleapis.com/openimages/web/index.html) using ```downloadIO.py```
  * Collect the data according to needed amenities from openimages dataset
  * Download the respective .csv files from openimages dataset

#### TRAINING 
- - - - - - -
##### PREPARE DATA
```
## download data, get the needed images ids and the bounding boxes information
python main.py
```
##### TRAIN THE DATA
```
python train.py --img 640 
                --batch 8 
                --epochs 75 
                --data data/airbnb.yaml 
                --cfg models/yolov5x.yaml 
                --weights '' 
                --name yolov5x_airbnb_results 
                --cache
```
Here, I trained the model with random initialized weights and Yolo5x trained weights on COCO. The weights can be downloaded from google drive.

##### TESTING
```
python test.py --weights runs/exp4_yolov5x_airbnb_results/weights/best.pt 
               --data airbnb.yaml 
               --img 672
```

##### INFERENCE
```
python detect.py --weights runs/exp6_yolov5x_airbnb_results/weights/best.pt 
                 --source dataset/images/test/ 
                 --output output/inference_new 
                 --img 640 
                 --save-txt

## ENSEMBLE INFERENCE               
python detect.py --weights runs/exp5_yolov5x_airbnb_results/weights/best.pt runs/exp6_yolov5x_airbnb_results/weights/best.pt 
                --source dataset/images/test/ 
                --output output/inference_right_weights 
                --img 640 
                --save-txt

```

##### WEB APP
FLASK API
```
python detect_flask.py
```


I want to thank from botoom of my heart to the Abhisekh Thakur, Daniel Bourke, Eugene Yan and most important YOLOv5 Ultra Analytics. I learned a lot from them for through this project.
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3983579.svg)](https://doi.org/10.5281/zenodo.3983579)

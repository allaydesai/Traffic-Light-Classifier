# Traffic-Light-Classifier

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Self-Driving Car Engineer Nanodegree Program

---

[//]: # (Image References)

[video]: ./images/result.gif "Video showing segmentation in action"
[sim-image-1]: ./images/sim_g_0002.jpg "Sim Training Image"
[sim-image-2]: ./images/sim_r_0030.jpg "Sim Training Image"
[sim-image-3]: ./images/sim_r_0103.jpg "Sim Training Image"
[sim-image-4]: ./images/sim_y_0001.jpg "Sim Training Image"
[loop-image-1]: ./images/uda_g_0005.jpg "Loop Training Image"
[loop-image-2]: ./images/uda_r_0066.jpg "Loop Training Image"
[loop-image-3]: ./images/uda_u_0337.jpg "Loop Training Image"
[loop-image-4]: ./images/uda_y_0160.jpg "Loop Training Image"
[result1]: ./images/sim_result_image.png "Sim Result"
[result2]: ./images/loop_result_image.png "Loop Result"

### OVERVIEW

Fine tuning SSD_mobilenet using Tensorflow Object Detection API for custom detection

### DATASET

![alt text][sim-image-1] | ![alt text][sim-image-2] 
:-------------------------:|:-------------------------:
![alt text][sim-image-3] | ![alt text][sim-image-4] 

### TRAINING PROCESS 

**STEP-1:** Create anaconda environment and install dependencies 

	conda create -n object_detection python=3

	activate object_detection

	pip install tensorflow
	pip install pillow
	pip install lxml
	pip install jupyter
	pip install matplotlib


**STEP-2:** Navigate to project directory and clone repo

	git clone https://github.com/tensorflow/models.git

Upon completing this you should be able to navigate  the following path:
Path_to_project_folder/models/research/object_detection

**STEP-3:** Extract dependent python programs 

Download and unzip Protocol Buffers V3.4.0 https://github.com/protocolbuffers/protobuf/releases

From : `models/research/`

Windows:
	protoc object_detection/protos/*.proto --python_out=.

Linux:
	"C:/Program Files/protoc/bin/protoc" object_detection/protos/*.proto --python_out=.

**STEP-4:** Add system path 

From: `models/research/`

Windows:
	SET PYTHONPATH=%cd%;%cd%\slim

Linux:
	export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

**STEP-5:** Move the new finetune dataset in project folder 

Move all the tf-records created to data folder
	- Train.tfrecord…

To: models/research/object_detection/data/

Move the label.lbtxt file to data folder

To: models/research/object_detection/data/

**STEP-6:** Download the model and config file for fine tuning 

From: models/research/object_detection/

Model of choice: `ssd_mobilenet_v1_coco`

	`wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz`

From: models/research/object_detection/training/

Config file: `ssd_mobilenet_v1_coco.config`

	`wget https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/samples/configs/ssd_mobilenet_v1_coco.config`

**STEP-7:** Edit the config file

Edit the following parameters 
`
	Model.ssd
		- .num_classes: 3
		- .type: 'ssd_mobilenet_v1'
	Train_config
		- .batch_size: 10-28
		- fine_tune_checkpoint: "ssd_mobilenet_v1_coco_2018_01_28/model.ckpt"

	Train_input_reader
		- .tf_record_input_reader.input_path: "data/train.record"
		- label_map_path: "data/label_map.pbtxt"

	eval_input_reader 
		- tf_record_input_reader.input_path: "data/test.record"
		- label_map_path: "data/label_map.pbtxt"
`
**STEP-8:** Create a label File

	
	item {
	  id: 1
	  name: 'Green'
	}

	item {
	  id: 2
	  name: 'Yellow'
	}

	item {
	  id: 3
	  name: 'Red'
	}
	

**STEP-9:** Train model

From: `models/research/object_detection`

	python3 legacy/train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v1_coco.config

**STEP-10:** Visualize model training

From: `models/research/object_detection`

	tensorboard --logdir='training'

Visit in browser: 127.0.0.1:6006


### RESULTS

![alt text][result1] | ![alt text][result2] 
:-------------------------:|:-------------------------:

### REFERENCE LINKS:

- Tensorflow Models
https://github.com/tensorflow/models

- Tensorflow API Object Detection Installation https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md

- Tensorflow Object Detection API Tutorial https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html

- Tensorflow detection model zoo https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

- Tensorflow detection model config https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs

- Introduction and Use - Tensorflow Object Detection API Tutorial https://pythonprogramming.net/introduction-use-tensorflow-object-detection-api-tutorial/


### REFERENCE REPOSITORIES:

- https://github.com/SiliconCar/traffic_light_detection_one_shot_transfer_learning

- https://github.com/Az4z3l/CarND-Traffic-Light-Detection

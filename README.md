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

This project is part of Udacity self driving car nano-degree [Capstone Project](https://github.com/allaydesai/SDCND_system_integration) and plays the role of perception module within it. Goal of this project is to fine tune a SSD_mobilenet model using tensorflow object detection API and pretrained weights on coco dataset for customized traffic light detection. The trained model detects traffic lights from the images sent by the onboard camera and classify's the detections into 3 available categories: green, yellow and red. Based on the classification the car can decide on how to behave in proximity of traffic lights.

The process begins with data collection followed by labeling of collected data. Next the data is divided into training and testing sets. The newly divided sets of data are then converted to TF Records to be processed by tensorflow API training function. 

Upon completion of the newly created dataset, the next step is to create an environment with required dependecies. Thereafter, the tensorflow/models repository is cloned. Now we can download the chosen pre trained model and its respective config file. Finally, the config file is edited and the model is set to train. 

### DATASET

Steps for creating custom data:

**STEP1-Data Collection:** Collect a few hundred images of traffic lights in udacity simulator and from udacity test loop found in rosbag.

**STEP2-Label Data:** Label the images using LabelImg: This process involves drawing boxes around your object's in an image. The label program generates an XML file that describes the object's in the pictures.

	https://github.com/tzutalin/labelImg

**STEP3-Split Data:** Split the data into train/test samples, images and respective XML files

**STEP4-Convert XML to CSV:** Using the helper library convert the csv files to XML

	def main():
	    for directory in ['train','test']:
		image_path = os.path.join(os.getcwd(), 'images/{}'.format(directory))
		xml_df = xml_to_csv(image_path)
		xml_df.to_csv('data/{}_labels.csv'.format(directory), index=None)
		print('Successfully converted xml to csv.')
	    
**STEP5-Generate TF Records:** Generate TF Records from these split data

  Make the following changes:
	
	def class_text_to_int(row_label):
	    if row_label == 'Green':
		return 1
	    else if row_label == 'Yellow':
		return 2
	    else if row_label == 'Red':
		return 3
	    else:
		None

  Execute:
	
	python3 generate_tfrecord.py --csv_input=data/train_labels.csv --output_path=data/train.record

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

Upon completing this you should be able to navigate to the following path:

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

To: `models/research/object_detection/data/`

Move the label.lbtxt file to data folder

To: `models/research/object_detection/data/`

**STEP-6:** Download the model and config file for fine tuning 

From: `models/research/object_detection/`

Model of choice: `ssd_mobilenet_v1_coco`

	wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz

From: `models/research/object_detection/training/`

Config file: `ssd_mobilenet_v1_coco.config`

	wget https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/samples/configs/ssd_mobilenet_v1_coco.config

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

Visit in browser: `127.0.0.1:6006`

**STEP-11:** Create inference graph

Finally, Due to system requirements on Carla, udacity's Self Driving. The model must load and predict using tensorflow V1.3.0. Now the object detection API can only be rolled back to compatibility with tensorflow V1.4.0. Hence, we have to create the inference graph using the training checkpoints in tensorflow V1.4.0 virtual environment. The graph created from this is compatible with tensorflow V1.3.0.

Create a temporary repository of tensorflow models/object_detection

	git clone https://github.com/tensorflow/models.git temp
	
Roll it back to a version compatible with tensorflow V1.4.0: 

	cd temp
	git checkout d135ed9c04bc9c60ea58f493559e60bc7673beb7
	
Export the inference graph:

	python3 export_inference_graph.py \
	    --input_type image_tensor \
	    --pipeline_config_path training/ssd_mobilenet_v1_pets.config \
	    --trained_checkpoint_prefix training/model.ckpt-10856 \
	    --output_directory mac_n_cheese_inference_graph

### RESULTS

![alt text][result1] | ![alt text][result2] 
:-------------------------:|:-------------------------:

### REFERENCE LINKS:

- Tensorflow Models

	https://github.com/tensorflow/models

- Tensorflow API Object Detection Installation 

	https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md

- Tensorflow Object Detection API Tutorial 

	https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html

- Tensorflow detection model zoo 

	https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

- Tensorflow detection model config 

	https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs

- Introduction and Use - Tensorflow Object Detection API Tutorial 

	https://pythonprogramming.net/introduction-use-tensorflow-object-detection-api-tutorial/


### REFERENCE REPOSITORIES:

- https://github.com/SiliconCar/traffic_light_detection_one_shot_transfer_learning

- https://github.com/Az4z3l/CarND-Traffic-Light-Detection

- https://github.com/datitran/raccoon_dataset

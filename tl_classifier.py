from styx_msgs.msg import TrafficLight
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import rospy

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt

import label_map_util as label_util

class TLClassifier(object):


		def __init__(self):
				#TODO load classifier
			MODEL_NAME = 'simulator_inference_graph'
			#MODEL_FILE = MODEL_NAME + '.tar.gz'
			#DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
			BASE_PATH = './light_classification/'
			# Path to frozen detection graph. This is the actual model that is used for the object detection.
			rospy.loginfo("path :: %s", os.getcwd())
			PATH_TO_FROZEN_GRAPH = BASE_PATH + 'models/' + MODEL_NAME + '/frozen_inference_graph.pb'

			# List of the strings that is used to add correct label for each box.
			PATH_TO_LABELS = BASE_PATH + 'label_map.pbtxt'


			# Loading tensorflow model
			self.detection_graph = tf.Graph()
			with self.detection_graph.as_default():
				self.od_graph_def = tf.GraphDef()
				with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
					self.serialized_graph = fid.read()
					self.od_graph_def.ParseFromString(self.serialized_graph)

					tf.import_graph_def(self.od_graph_def, name='')


			#Loading label map
			self.category_index = label_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

			#

		def run_inference_for_single_image(self, image):
			with self.detection_graph.as_default():
				with tf.Session() as sess:
					# Get handles to input and output tensors
					ops = tf.get_default_graph().get_operations()
					all_tensor_names = {output.name for op in ops for output in op.outputs}
					tensor_dict = {}
					for key in [
							'num_detections', 'detection_boxes', 'detection_scores',
							'detection_classes', 'detection_masks'
					]:
						tensor_name = key + ':0'
						if tensor_name in all_tensor_names:
							tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
									tensor_name)
					if 'detection_masks' in tensor_dict:
						# The following processing is only for single image
						detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
						detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
						# Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
						real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
						detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
						detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
						detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
								detection_masks, detection_boxes, image.shape[0], image.shape[1])
						detection_masks_reframed = tf.cast(
								tf.greater(detection_masks_reframed, 0.5), tf.uint8)
						# Follow the convention by adding back the batch dimension
						tensor_dict['detection_masks'] = tf.expand_dims(
								detection_masks_reframed, 0)
					image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

					# Run inference
					output_dict = sess.run(tensor_dict,
																 feed_dict={image_tensor: np.expand_dims(image, 0)})

					# all outputs are float32 numpy arrays, so convert types as appropriate
					output_dict['num_detections'] = int(output_dict['num_detections'][0])
					output_dict['detection_classes'] = output_dict[
							'detection_classes'][0].astype(np.uint8)
					output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
					output_dict['detection_scores'] = output_dict['detection_scores'][0]
					if 'detection_masks' in output_dict:
						output_dict['detection_masks'] = output_dict['detection_masks'][0]
			return output_dict

		def get_classification(self, image):
			"""Determines the color of the traffic light in the image
			Args:
					image (cv::Mat): image containing the traffic light
			Returns:
					int: ID of traffic light color (specified in styx_msgs/TrafficLight)
			"""

			#image: uint8 numpy array with shape (img_height, img_width, 3)
		    #boxes: a numpy array of shape [N, 4]
		    #classes: a numpy array of shape [N]. Note that class indices are 1-based,
		    #  and match the keys in the label map.
		    #scores: a numpy array of shape [N] or None.  If scores=None, then
		    #  this function assumes that the boxes to be plotted are groundtruth
		    #  boxes and plot all boxes as black with no classes or scores.
		    #category_index: a dict containing category dictionaries (each holding
		    #  category index `id` and category name `name`) keyed by category indices.
		    #instance_masks: a numpy array of shape [N, image_height, image_width] with
		    #  values ranging between 0 and 1, can be None.


			#input_image = np.expand_dims(image,axis=0)

			#TODO implement light color prediction
			output_dict = self.run_inference_for_single_image(image)
			#print("ran inference")

			boxes = output_dict['detection_boxes']
			classes = output_dict['detection_classes']
			scores = output_dict['detection_scores']
			instance_masks=output_dict.get('detection_masks')

			min_score_thresh = 0.5
			max_boxes_to_draw=20

			#print('no of boxes: ', boxes.shape[0])
			print('')
			# assigining a default value
			class_name = 'Unknown'

			for i in range(min(max_boxes_to_draw, boxes.shape[0])):
				if scores is None or scores[i] > min_score_thresh:
					box = tuple(boxes[i].tolist())

					if classes[i] in self.category_index.keys():
						class_name = self.category_index[classes[i]]['name']
						print(i,class_name, '{}%'.format(int(100*scores[i])))

			#class_name = category_index[classes[0]]['name']
			#print(class_name)

			# uint8 UNKNOWN=4
			# uint8 GREEN=2
			# uint8 YELLOW=1
			# uint8 RED=0
			
			if class_name == 'Red':
				return TrafficLight.RED
			elif class_name == 'Green':
				return TrafficLight.GREEN
			elif class_name == 'Yellow':
				return TrafficLight.YELLOW
			else:
				return TrafficLight.UNKNOWN

			#return class_name

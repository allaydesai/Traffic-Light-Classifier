import tl_classifier as cl
from PIL import Image
import numpy as np



print("imported library")

Classifier = cl.TLClassifier()

print("created object")

def load_image_into_numpy_array(image):
			(im_width, im_height) = image.size
			return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


image_path = "/home/student/traffic_light_dataset/carla/carla/testing/uda_2_r_0082.jpg"

image = Image.open(image_path)
# the array based representation of the image will be used later in order to prepare the
# result image with boxes and labels on it.
image_np = load_image_into_numpy_array(image)
# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
image_np_expanded = np.expand_dims(image_np, axis=0)

print('running classification')
#output = Classifier.run_inference_for_single_image(image_np)
result = Classifier.get_classification(image_np)

print('result_received: ', result)

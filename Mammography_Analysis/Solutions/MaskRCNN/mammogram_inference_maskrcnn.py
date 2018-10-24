
# coding: utf-8

# In[ ]:


## Specify the test data path here 
data_path = '/home/ddh/Projects/DDH/Caries/Mask_RCNN/samples/Mammogram/Boston Meditech Group/Test_data/'


# In[1]:




# Set matplotlib backend
# This has to be done before other importa that might
# set it, but only if we're running in script mode
# rather than being imported.
# get_ipython().run_line_magic('matplotlib', 'inline')
import random
import os
import sys
import json
import datetime
import numpy as np
import skimage.io
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("./")
print(ROOT_DIR)
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize
from mrcnn.model import log
import scipy.io as sio

from keras.applications.imagenet_utils import preprocess_input


# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
# DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR,"caries", "logs")
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
# Results directory
# Save submission files here
RESULTS_DIR = os.path.join(ROOT_DIR, "results/")
# print(COCO_WEIGHTS_PATH,DEFAULT_LOGS_DIR,RESULTS_DIR)
# exit(0)


# The dataset doesn't have a standard train/val split, so I picked
# a variety of images to surve as a validation set.
VAL_IMAGE_IDS = [
    "mdb058",
    "mdb111",
    "mdb179",
    "mdb213",
    "mdb241",
    "mdb274",
    "mdb125",
    "mdb010",
    "mdb030",
    "mdb097",
    "mdb145",
    "mdb188",
    "mdb222",
    "mdb314"
]



# In[2]:


class MamoConfig(Config):
    """Configuration for training on the nucleus segmentation dataset."""
    # Give the configuration a recognizable name
    NAME = "mamo"
    GPU_COUNT = 2

    # Adjust depending on your GPU memory
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # Background + Benign + Malignant

    # Number of training and validation steps per epoch
    STEPS_PER_EPOCH = (112 - len(VAL_IMAGE_IDS)) // IMAGES_PER_GPU
    VALIDATION_STEPS = max(1, len(VAL_IMAGE_IDS) // IMAGES_PER_GPU)

    # Don't exclude based on confidence. Since we have two classes
    # then 0.5 is the minimum anyway as it picks between nucleus and BG
    DETECTION_MIN_CONFIDENCE = 0.35

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet101"

    # Input image resizing
    # Random crops of size 512x512
    IMAGE_RESIZE_MODE = "none"
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024
    IMAGE_MIN_SCALE = 2.0

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = ( 16, 32, 64, 128,256)
    #RPN_ANCHOR_SCALES = (2, 4, 8, 16)

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 1000

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.9

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256

    # Image mean (RGB)
    # MEAN_PIXEL = np.array([43.53, 39.56, 48.22])
    MEAN_PIXEL = np.array([120, 120, 120])

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 512

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 3

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 3


config = MamoConfig()
config.display()


# In[3]:


model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=DEFAULT_LOGS_DIR)


# In[4]:


# Which weights to start with?
init_with = "last"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_WEIGHTS_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last(), by_name=True)


# In[5]:


class InferenceConfig(MamoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()
# Device to load the neural network on.
# Useful if you're training a model
# on the same
# machine, in which case use CPU and leave the
# GPU for training.
DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# Only inference mode is supported right now
TEST_MODE = "inference"


# In[6]:


# Create model in inference mode
LOGS_DIR = os.path.join(ROOT_DIR, "logs")
import tensorflow as tf

with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference",
                              model_dir=LOGS_DIR,
                              config=config)


# In[7]:


# Or, load the last model you trained
weights_path = model.find_last()
# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)


# In[8]:


test_image = os.listdir(data_path)


# In[9]:


class_names = ['BG', 'Benign', 'Malignant']
for img in test_image:
    image1 = skimage.io.imread(data_path+img)

    image1 = skimage.color.gray2rgb(image1)
    result = model.detect([image1],verbose = 1)
    r = result[0]
    print(r)
    visualize.display_instances(image1,r['rois'], r['masks'], r['class_ids'],class_names, r['scores'])


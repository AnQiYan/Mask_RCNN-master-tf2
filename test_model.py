# -*- coding: utf-8 -*-
import os
import sys
import random
import skimage.io
from mrcnn.config import Config
from datetime import datetime
from PIL import Image
# size = 600, 450
# im = Image.open("img.png")
# im_resized = im.resize(size, Image.ANTIALIAS)
# im_resized.save("my_image_resized.png", "PNG")
#




ROOT_DIR = os.getcwd()
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
import mrcnn.model as modellib
from mrcnn import visualize

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
MODEL_WEIGHT = './logs/shapes20220326T1031/mask_rcnn_shapes_0010.h5'

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images\\neuro")

class ShapesConfig(Config):
    # Give the configuration a recognizable name
    NAME = "shapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 320
    IMAGE_MAX_DIM = 384

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8 * 6, 16 * 6, 32 * 6, 64 * 6, 128 * 6)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE =100

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50

#import train_tongue
#class InferenceConfig(coco.CocoConfig):
class InferenceConfig(ShapesConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(MODEL_WEIGHT, by_name=True)

# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'neuro']
# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]
print(file_names)
image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
a=datetime.now()
# Run detection
results = model.detect([image], verbose=1)
b=datetime.now() 
# Visualize results
print("time:",(b-a).seconds)
r = results[0]
print(r)
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            class_names, r['scores'], show_mask=True, colors=[(0, 0, 0)])

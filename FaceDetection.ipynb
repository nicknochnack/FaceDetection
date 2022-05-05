{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 1. Setup and Get Data"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 1.1 Install Dependencies and Setup"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "!pip install labelme tensorflow tensorflow-gpu opencv-python matplotlib albumentations"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 1.2 Collect Images Using OpenCV"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "import os\n",
    "import time\n",
    "import uuid\n",
    "import cv2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "IMAGES_PATH = os.path.join('data','images')\n",
    "number_images = 30"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "cap = cv2.VideoCapture(1)\n",
    "for imgnum in range(number_images):\n",
    "    print('Collecting image {}'.format(imgnum))\n",
    "    ret, frame = cap.read()\n",
    "    imgname = os.path.join(IMAGES_PATH,f'{str(uuid.uuid1())}.jpg')\n",
    "    cv2.imwrite(imgname, frame)\n",
    "    cv2.imshow('frame', frame)\n",
    "    time.sleep(0.5)\n",
    "\n",
    "    if cv2.waitKey(1) & 0xFF == ord('q'):\n",
    "        break\n",
    "cap.release()\n",
    "cv2.destroyAllWindows()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 1.3 Annotate Images with LabelMe"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "!labelme"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 2. Review Dataset and Build Image Loading Function"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2.1 Import TF and Deps"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "import tensorflow as tf\n",
    "import json\n",
    "import numpy as np\n",
    "from matplotlib import pyplot as plt"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2.2 Limit GPU Memory Growth"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Avoid OOM errors by setting GPU Memory Consumption Growth\n",
    "gpus = tf.config.experimental.list_physical_devices('GPU')\n",
    "for gpu in gpus: \n",
    "    tf.config.experimental.set_memory_growth(gpu, True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "tf.config.list_physical_devices('GPU')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2.3 Load Image into TF Data Pipeline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "images = tf.data.Dataset.list_files('data\\\\images\\\\*.jpg')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "images.as_numpy_iterator().next()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "def load_image(x): \n",
    "    byte_img = tf.io.read_file(x)\n",
    "    img = tf.io.decode_jpeg(byte_img)\n",
    "    return img"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "images = images.map(load_image)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": true,
    "tags": []
   },
   "outputs": [],
   "source": [
    "images.as_numpy_iterator().next()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "type(images)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2.4 View Raw Images with Matplotlib"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "image_generator = images.batch(4).as_numpy_iterator()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "plot_images = image_generator.next()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "fig, ax = plt.subplots(ncols=4, figsize=(20,20))\n",
    "for idx, image in enumerate(plot_images):\n",
    "    ax[idx].imshow(image) \n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 3. Partition Unaugmented Data"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3.1 MANUALLY SPLT DATA INTO TRAIN TEST AND VAL"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "90*.7 # 63 to train"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "90*.15 # 14 and 13 to test and val"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3.2 Move the Matching Labels"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "for folder in ['train','test','val']:\n",
    "    for file in os.listdir(os.path.join('data', folder, 'images')):\n",
    "        \n",
    "        filename = file.split('.')[0]+'.json'\n",
    "        existing_filepath = os.path.join('data','labels', filename)\n",
    "        if os.path.exists(existing_filepath): \n",
    "            new_filepath = os.path.join('data',folder,'labels',filename)\n",
    "            os.replace(existing_filepath, new_filepath)      "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 4. Apply Image Augmentation on Images and Labels using Albumentations"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 4.1 Setup Albumentations Transform Pipeline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "import albumentations as alb"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "augmentor = alb.Compose([alb.RandomCrop(width=450, height=450), \n",
    "                         alb.HorizontalFlip(p=0.5), \n",
    "                         alb.RandomBrightnessContrast(p=0.2),\n",
    "                         alb.RandomGamma(p=0.2), \n",
    "                         alb.RGBShift(p=0.2), \n",
    "                         alb.VerticalFlip(p=0.5)], \n",
    "                       bbox_params=alb.BboxParams(format='albumentations', \n",
    "                                                  label_fields=['class_labels']))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 4.2 Load a Test Image and Annotation with OpenCV and JSON"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "img = cv2.imread(os.path.join('data','train', 'images','ffd85fc5-cc1a-11ec-bfb8-a0cec8d2d278.jpg'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "with open(os.path.join('data', 'train', 'labels', 'ffd85fc5-cc1a-11ec-bfb8-a0cec8d2d278.json'), 'r') as f:\n",
    "    label = json.load(f)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "label['shapes'][0]['points']"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 4.3 Extract Coordinates and Rescale to Match Image Resolution"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "coords = [0,0,0,0]\n",
    "coords[0] = label['shapes'][0]['points'][0][0]\n",
    "coords[1] = label['shapes'][0]['points'][0][1]\n",
    "coords[2] = label['shapes'][0]['points'][1][0]\n",
    "coords[3] = label['shapes'][0]['points'][1][1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "coords"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "coords = list(np.divide(coords, [640,480,640,480]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "coords"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 4.4 Apply Augmentations and View Results"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "augmented = augmentor(image=img, bboxes=[coords], class_labels=['face'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "augmented['bboxes'][0][2:]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "augmented['bboxes']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "cv2.rectangle(augmented['image'], \n",
    "              tuple(np.multiply(augmented['bboxes'][0][:2], [450,450]).astype(int)),\n",
    "              tuple(np.multiply(augmented['bboxes'][0][2:], [450,450]).astype(int)), \n",
    "                    (255,0,0), 2)\n",
    "\n",
    "plt.imshow(augmented['image'])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 5. Build and Run Augmentation Pipeline"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 5.1 Run Augmentation Pipeline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "for partition in ['train','test','val']: \n",
    "    for image in os.listdir(os.path.join('data', partition, 'images')):\n",
    "        img = cv2.imread(os.path.join('data', partition, 'images', image))\n",
    "\n",
    "        coords = [0,0,0.00001,0.00001]\n",
    "        label_path = os.path.join('data', partition, 'labels', f'{image.split(\".\")[0]}.json')\n",
    "        if os.path.exists(label_path):\n",
    "            with open(label_path, 'r') as f:\n",
    "                label = json.load(f)\n",
    "\n",
    "            coords[0] = label['shapes'][0]['points'][0][0]\n",
    "            coords[1] = label['shapes'][0]['points'][0][1]\n",
    "            coords[2] = label['shapes'][0]['points'][1][0]\n",
    "            coords[3] = label['shapes'][0]['points'][1][1]\n",
    "            coords = list(np.divide(coords, [640,480,640,480]))\n",
    "\n",
    "        try: \n",
    "            for x in range(60):\n",
    "                augmented = augmentor(image=img, bboxes=[coords], class_labels=['face'])\n",
    "                cv2.imwrite(os.path.join('aug_data', partition, 'images', f'{image.split(\".\")[0]}.{x}.jpg'), augmented['image'])\n",
    "\n",
    "                annotation = {}\n",
    "                annotation['image'] = image\n",
    "\n",
    "                if os.path.exists(label_path):\n",
    "                    if len(augmented['bboxes']) == 0: \n",
    "                        annotation['bbox'] = [0,0,0,0]\n",
    "                        annotation['class'] = 0 \n",
    "                    else: \n",
    "                        annotation['bbox'] = augmented['bboxes'][0]\n",
    "                        annotation['class'] = 1\n",
    "                else: \n",
    "                    annotation['bbox'] = [0,0,0,0]\n",
    "                    annotation['class'] = 0 \n",
    "\n",
    "\n",
    "                with open(os.path.join('aug_data', partition, 'labels', f'{image.split(\".\")[0]}.{x}.json'), 'w') as f:\n",
    "                    json.dump(annotation, f)\n",
    "\n",
    "        except Exception as e:\n",
    "            print(e)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 5.2 Load Augmented Images to Tensorflow Dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "train_images = tf.data.Dataset.list_files('aug_data\\\\train\\\\images\\\\*.jpg', shuffle=False)\n",
    "train_images = train_images.map(load_image)\n",
    "train_images = train_images.map(lambda x: tf.image.resize(x, (120,120)))\n",
    "train_images = train_images.map(lambda x: x/255)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "test_images = tf.data.Dataset.list_files('aug_data\\\\test\\\\images\\\\*.jpg', shuffle=False)\n",
    "test_images = test_images.map(load_image)\n",
    "test_images = test_images.map(lambda x: tf.image.resize(x, (120,120)))\n",
    "test_images = test_images.map(lambda x: x/255)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "val_images = tf.data.Dataset.list_files('aug_data\\\\val\\\\images\\\\*.jpg', shuffle=False)\n",
    "val_images = val_images.map(load_image)\n",
    "val_images = val_images.map(lambda x: tf.image.resize(x, (120,120)))\n",
    "val_images = val_images.map(lambda x: x/255)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "train_images.as_numpy_iterator().next()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 6. Prepare Labels"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 6.1 Build Label Loading Function"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "def load_labels(label_path):\n",
    "    with open(label_path.numpy(), 'r', encoding = \"utf-8\") as f:\n",
    "        label = json.load(f)\n",
    "        \n",
    "    return [label['class']], label['bbox']"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 6.2 Load Labels to Tensorflow Dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "train_labels = tf.data.Dataset.list_files('aug_data\\\\train\\\\labels\\\\*.json', shuffle=False)\n",
    "train_labels = train_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "test_labels = tf.data.Dataset.list_files('aug_data\\\\test\\\\labels\\\\*.json', shuffle=False)\n",
    "test_labels = test_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "val_labels = tf.data.Dataset.list_files('aug_data\\\\val\\\\labels\\\\*.json', shuffle=False)\n",
    "val_labels = val_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_labels.as_numpy_iterator().next()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 7. Combine Label and Image Samples"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 7.1 Check Partition Lengths"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "len(train_images), len(train_labels), len(test_images), len(test_labels), len(val_images), len(val_labels)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 7.2 Create Final Datasets (Images/Labels)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "train = tf.data.Dataset.zip((train_images, train_labels))\n",
    "train = train.shuffle(5000)\n",
    "train = train.batch(8)\n",
    "train = train.prefetch(4)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "test = tf.data.Dataset.zip((test_images, test_labels))\n",
    "test = test.shuffle(1300)\n",
    "test = test.batch(8)\n",
    "test = test.prefetch(4)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "val = tf.data.Dataset.zip((val_images, val_labels))\n",
    "val = val.shuffle(1000)\n",
    "val = val.batch(8)\n",
    "val = val.prefetch(4)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "train.as_numpy_iterator().next()[1]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 7.3 View Images and Annotations"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": true,
    "tags": []
   },
   "outputs": [],
   "source": [
    "data_samples = train.as_numpy_iterator()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "res = data_samples.next()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "fig, ax = plt.subplots(ncols=4, figsize=(20,20))\n",
    "for idx in range(4): \n",
    "    sample_image = res[0][idx]\n",
    "    sample_coords = res[1][1][idx]\n",
    "    \n",
    "    cv2.rectangle(sample_image, \n",
    "                  tuple(np.multiply(sample_coords[:2], [120,120]).astype(int)),\n",
    "                  tuple(np.multiply(sample_coords[2:], [120,120]).astype(int)), \n",
    "                        (255,0,0), 2)\n",
    "\n",
    "    ax[idx].imshow(sample_image)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 8. Build Deep Learning using the Functional API"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 8.1 Import Layers and Base Network"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "from tensorflow.keras.models import Model\n",
    "from tensorflow.keras.layers import Input, Conv2D, Dense, GlobalMaxPooling2D\n",
    "from tensorflow.keras.applications import VGG16"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 8.2 Download VGG16"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "vgg = VGG16(include_top=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "vgg.summary()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 8.3 Build instance of Network"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "def build_model(): \n",
    "    input_layer = Input(shape=(120,120,3))\n",
    "    \n",
    "    vgg = VGG16(include_top=False)(input_layer)\n",
    "\n",
    "    # Classification Model  \n",
    "    f1 = GlobalMaxPooling2D()(vgg)\n",
    "    class1 = Dense(2048, activation='relu')(f1)\n",
    "    class2 = Dense(1, activation='sigmoid')(class1)\n",
    "    \n",
    "    # Bounding box model\n",
    "    f2 = GlobalMaxPooling2D()(vgg)\n",
    "    regress1 = Dense(2048, activation='relu')(f2)\n",
    "    regress2 = Dense(4, activation='sigmoid')(regress1)\n",
    "    \n",
    "    facetracker = Model(inputs=input_layer, outputs=[class2, regress2])\n",
    "    return facetracker"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 8.4 Test out Neural Network"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "facetracker = build_model()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "facetracker.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "X, y = train.as_numpy_iterator().next()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "X.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "classes, coords = facetracker.predict(X)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "classes, coords"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 9. Define Losses and Optimizers"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 9.1 Define Optimizer and LR"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "batches_per_epoch = len(train)\n",
    "lr_decay = (1./0.75 -1)/batches_per_epoch"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "opt = tf.keras.optimizers.Adam(learning_rate=0.0001, decay=lr_decay)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 9.2 Create Localization Loss and Classification Loss"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "def localization_loss(y_true, yhat):            \n",
    "    delta_coord = tf.reduce_sum(tf.square(y_true[:,:2] - yhat[:,:2]))\n",
    "                  \n",
    "    h_true = y_true[:,3] - y_true[:,1] \n",
    "    w_true = y_true[:,2] - y_true[:,0] \n",
    "\n",
    "    h_pred = yhat[:,3] - yhat[:,1] \n",
    "    w_pred = yhat[:,2] - yhat[:,0] \n",
    "    \n",
    "    delta_size = tf.reduce_sum(tf.square(w_true - w_pred) + tf.square(h_true-h_pred))\n",
    "    \n",
    "    return delta_coord + delta_size"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "classloss = tf.keras.losses.BinaryCrossentropy()\n",
    "regressloss = localization_loss"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 9.3 Test out Loss Metrics"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "localization_loss(y[1], coords)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "classloss(y[0], classes)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "regressloss(y[1], coords)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 10. Train Neural Network"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 10.1 Create Custom Model Class"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "class FaceTracker(Model): \n",
    "    def __init__(self, eyetracker,  **kwargs): \n",
    "        super().__init__(**kwargs)\n",
    "        self.model = eyetracker\n",
    "\n",
    "    def compile(self, opt, classloss, localizationloss, **kwargs):\n",
    "        super().compile(**kwargs)\n",
    "        self.closs = classloss\n",
    "        self.lloss = localizationloss\n",
    "        self.opt = opt\n",
    "    \n",
    "    def train_step(self, batch, **kwargs): \n",
    "        \n",
    "        X, y = batch\n",
    "        \n",
    "        with tf.GradientTape() as tape: \n",
    "            classes, coords = self.model(X, training=True)\n",
    "            \n",
    "            batch_classloss = self.closs(y[0], classes)\n",
    "            batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords)\n",
    "            \n",
    "            total_loss = batch_localizationloss+0.5*batch_classloss\n",
    "            \n",
    "            grad = tape.gradient(total_loss, self.model.trainable_variables)\n",
    "        \n",
    "        opt.apply_gradients(zip(grad, self.model.trainable_variables))\n",
    "        \n",
    "        return {\"total_loss\":total_loss, \"class_loss\":batch_classloss, \"regress_loss\":batch_localizationloss}\n",
    "    \n",
    "    def test_step(self, batch, **kwargs): \n",
    "        X, y = batch\n",
    "        \n",
    "        classes, coords = self.model(X, training=False)\n",
    "        \n",
    "        batch_classloss = self.closs(y[0], classes)\n",
    "        batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords)\n",
    "        total_loss = batch_localizationloss+0.5*batch_classloss\n",
    "        \n",
    "        return {\"total_loss\":total_loss, \"class_loss\":batch_classloss, \"regress_loss\":batch_localizationloss}\n",
    "        \n",
    "    def call(self, X, **kwargs): \n",
    "        return self.model(X, **kwargs)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "model = FaceTracker(facetracker)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "model.compile(opt, classloss, regressloss)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 10.2 Train"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "logdir='logs'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": true,
    "tags": []
   },
   "outputs": [],
   "source": [
    "hist = model.fit(train, epochs=10, validation_data=val, callbacks=[tensorboard_callback])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 10.3 Plot Performance"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": true,
    "tags": []
   },
   "outputs": [],
   "source": [
    "hist.history"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "fig, ax = plt.subplots(ncols=3, figsize=(20,5))\n",
    "\n",
    "ax[0].plot(hist.history['total_loss'], color='teal', label='loss')\n",
    "ax[0].plot(hist.history['val_total_loss'], color='orange', label='val loss')\n",
    "ax[0].title.set_text('Loss')\n",
    "ax[0].legend()\n",
    "\n",
    "ax[1].plot(hist.history['class_loss'], color='teal', label='class loss')\n",
    "ax[1].plot(hist.history['val_class_loss'], color='orange', label='val class loss')\n",
    "ax[1].title.set_text('Classification Loss')\n",
    "ax[1].legend()\n",
    "\n",
    "ax[2].plot(hist.history['regress_loss'], color='teal', label='regress loss')\n",
    "ax[2].plot(hist.history['val_regress_loss'], color='orange', label='val regress loss')\n",
    "ax[2].title.set_text('Regression Loss')\n",
    "ax[2].legend()\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 11. Make Predictions"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 11.1 Make Predictions on Test Set"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "test_data = test.as_numpy_iterator()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "test_sample = test_data.next()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "yhat = facetracker.predict(test_sample[0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "fig, ax = plt.subplots(ncols=4, figsize=(20,20))\n",
    "for idx in range(4): \n",
    "    sample_image = test_sample[0][idx]\n",
    "    sample_coords = yhat[1][idx]\n",
    "    \n",
    "    if yhat[0][idx] > 0.9:\n",
    "        cv2.rectangle(sample_image, \n",
    "                      tuple(np.multiply(sample_coords[:2], [120,120]).astype(int)),\n",
    "                      tuple(np.multiply(sample_coords[2:], [120,120]).astype(int)), \n",
    "                            (255,0,0), 2)\n",
    "    \n",
    "    ax[idx].imshow(sample_image)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 11.2 Save the Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "from tensorflow.keras.models import load_model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "facetracker.save('facetracker.h5')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "facetracker = load_model('facetracker.h5')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 11.3 Real Time Detection"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "cap = cv2.VideoCapture(1)\n",
    "while cap.isOpened():\n",
    "    _ , frame = cap.read()\n",
    "    frame = frame[50:500, 50:500,:]\n",
    "    \n",
    "    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)\n",
    "    resized = tf.image.resize(rgb, (120,120))\n",
    "    \n",
    "    yhat = facetracker.predict(np.expand_dims(resized/255,0))\n",
    "    sample_coords = yhat[1][0]\n",
    "    \n",
    "    if yhat[0] > 0.5: \n",
    "        # Controls the main rectangle\n",
    "        cv2.rectangle(frame, \n",
    "                      tuple(np.multiply(sample_coords[:2], [450,450]).astype(int)),\n",
    "                      tuple(np.multiply(sample_coords[2:], [450,450]).astype(int)), \n",
    "                            (255,0,0), 2)\n",
    "        # Controls the label rectangle\n",
    "        cv2.rectangle(frame, \n",
    "                      tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int), \n",
    "                                    [0,-30])),\n",
    "                      tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int),\n",
    "                                    [80,0])), \n",
    "                            (255,0,0), -1)\n",
    "        \n",
    "        # Controls the text rendered\n",
    "        cv2.putText(frame, 'face', tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int),\n",
    "                                               [0,-5])),\n",
    "                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)\n",
    "    \n",
    "    cv2.imshow('EyeTrack', frame)\n",
    "    \n",
    "    if cv2.waitKey(1) & 0xFF == ord('q'):\n",
    "        break\n",
    "cap.release()\n",
    "cv2.destroyAllWindows()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "facedet",
   "language": "python",
   "name": "facedet"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

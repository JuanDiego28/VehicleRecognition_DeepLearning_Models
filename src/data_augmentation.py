from pathlib import Path
import os
from shutil import copytree
import config
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
# https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator


# create path for augmented data
DATASET_AUGMENTED_ROOT_PATH = os.path.join(os.getcwd(),"dataset_augmented")
if not os.path.exists(DATASET_AUGMENTED_ROOT_PATH):
    copytree(config.DATASET_ROOT_PATH,DATASET_AUGMENTED_ROOT_PATH)

DATASET_AUGMENTED_FOLDER = os.path.join(DATASET_AUGMENTED_ROOT_PATH, "eu-car-dataset_subset")

# data generator
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

j = 0
for subdir, dirs, files in os.walk(os.path.join(DATASET_AUGMENTED_FOLDER,"train")):
    for filename in files:
        j += 1
        # Create the corresponding folder for the augmented images
        augmented_subdir = os.path.join(subdir, "augmented")
        os.makedirs(augmented_subdir, exist_ok=True)

        print(j, " - ", subdir, " - ", filename)

        # data augmentation
        if filename.split(".")[-1] == 'jpg':
            try:
                img = load_img(os.path.join(subdir,filename)) # load the image
                x = img_to_array(img)  # preprocess it
                x = x.reshape((1,) + x.shape)
                # generate batches of randomly transformed images
                # and save the results to the `augmented` directory
                i = 0
                for batch in datagen.flow(x, batch_size=1,
                                        save_to_dir=subdir,
                                        save_prefix= filename + "_" + str(i), save_format='jpeg'):
                    i += 1
                    if i > 3: # generate 4 images for each original one
                        break
                    
            except Exception as e:
                print(f"Error processing {os.path.join(subdir, filename)}: {e}")
                continue

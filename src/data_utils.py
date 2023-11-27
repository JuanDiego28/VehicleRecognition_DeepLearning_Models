import os
import zipfile

import gdown

from src import config
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def download_datasets(output_folder=config.DATASET_ROOT_PATH):
    """
    Download from GDrive all the needed datasets for the project.
    """
    # Create folder if doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    # Download dataset from Drive
    output_filename = os.path.join(output_folder, config.ZIP_DATASET_FILENAME)
    if not os.path.exists(output_filename):
        gdown.download(config.DATASET_URL, output_filename, quiet=False)

    # Unzip dataset
    with zipfile.ZipFile(output_filename, "r") as zip_ref:
        zip_ref.extractall(os.path.dirname(output_filename))

def plot_history(history):
    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

def  evaluate_model(model_evaluated, test_ds, class_names):
    y_pred = model_evaluated.predict(test_ds)
    # Extract ground truth labels from the test dataset
    y_true = np.concatenate([y for x, y in test_ds], axis=0)

    # Convert one-hot encoded labels to class indices
    y_true_indices = np.argmax(y_true, axis=1)

    # Convert predicted probabilities to class indices
    y_pred_indices = np.argmax(y_pred, axis=1)

    # Generate the classification report
    report = classification_report(y_true_indices, y_pred_indices, target_names=class_names)
    print(f'\n Classification Report: \n')
    print(report)

    conf_matrix = confusion_matrix(y_true_indices,y_pred_indices)

    # Display the confusion matrix
    print(f'\n Confusion Matrix: \n')

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()
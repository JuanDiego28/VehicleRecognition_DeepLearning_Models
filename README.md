# Sprint Project 04
> Vehicle classification from images

## The Business problem

Imagine a company that has a large inventory of used cars and wants to create a website or a mobile app to make it easier for customers to search and find the exact car they are looking for. One way to improve the user experience would be to allow customers to search for cars by make and model. However, manually entering this information for every car in the inventory could be time-consuming and error-prone.

Using a machine learning model capable of classifying vehicle make and model from images, the company could automatically extract this information from pictures of the cars. This would make it faster and more accurate to input the information, allowing customers to easily search and filter cars by make and model. This would not only improve the customer experience but also save the company time and resources. Additionally, the model could be used to automatically identify and categorize new cars as they come into the inventory. 

This is a Multi-class Classification task: we want to predict, given a picture of a vehicle, which of the possible 25 classes is the correct vehicle make-model.

As Machine Learning Engineers assigned to this task, our primary objective was to create a proof-of-concept model. The dataset provided included labeled images of 25 different cars for training and testing. The challenge was to train a model with high accuracy (over 80% on the testing dataset) for multi-class classification, predicting the correct vehicle make-model given an input image.

## Data Description

The dataset comprised JPG images organized into folders by label (vehicle make-model), with separate sets for training and testing. The data retrieval process was streamlined and automated within the AnyoneAI - Sprint Project 04.ipynb notebook, eliminating the need for manual data downloads.

## Technical aspects

* Python: Main programming language
* Tensorflow and Keras: Used for building features and training machine learning models
* Matplotlib: Employed for data visualizations
* Jupyter Notebooks: Facilitated an interactive and iterative experimentation process
* Google Colab: The best free option to work with gpu's to train the CNNs

The successful development of this machine learning model not only improved the customer experience by enabling efficient search and filtering of cars by make and model but also contributed to significant time and resource savings for the company. The model also demonstrated potential for automating the identification and categorization of new cars as they entered the inventory.
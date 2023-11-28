from typing import Tuple

from tensorflow.keras.applications import ResNet50, EfficientNetV2S, EfficientNetB1, EfficientNetB5
from tensorflow.keras.layers import (
    AveragePooling2D,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    Rescaling
)
from tensorflow.keras.models import Sequential

from tensorflow.keras import regularizers


def create_mlp_model(input_shape: Tuple[int, int, int], num_classes: int) -> Sequential:
    """Creates a Multi-layer perceptron model using Keras.

    Args:
        input_shape (Tuple[int, int, int]): The shape of the input data.
        num_classes (int): The number of output classes.

    Returns:
        A Sequential model object with the specified MLP architecture.
    """
    # Define the model
    model = Sequential()

    # Add the input layer
    model.add(Input(shape=input_shape))

    # Layer 0: Rescaling operation to map image pixels from [0, 255] to [0, 1] range
    model.add(Rescaling(1.0 / 255, input_shape=input_shape))

    # Layer 1: Flatten layer to convert the 3D input image to a 1D array
    model.add(Flatten())

    # Adding hidden layers to the model
    # Layer 2: Fully connected layer with 512 neurons,
    # followed by a relu activation function
    model.add(Dense(units = 512,activation = 'relu'))

    # Layer 3: Fully connected layer with 1024 neurons,
    # followed by a relu activation function
    model.add(Dense(units = 1024,activation = 'relu'))

    # Layer 4: Fully connected layer with 512 neurons,
    # followed by a relu activation function
    model.add(Dense(units = 512,activation = 'relu'))

    # Layer 5: Classification layer with num_classes output units,
    # followed by a softmax activation function
    model.add(Dense(units = num_classes,activation = 'softmax'))

    # Print a summary of the model architecture
    print(model.summary())

    return model


def create_lenet_model(
    input_shape: Tuple[int, int, int], num_classes: int
) -> Sequential:
    """
    Creates a LeNet convolutional neural network model. For reference see original
    publication: http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf.

    Args:
        input_shape (Tuple[int, int, int]): The shape of the input data.
        num_classes (int): The number of output classes.

    Returns:
        A Sequential model object representing the LeNet architecture.
    """
    # Define the model
    model = Sequential()

    # Add the input layer
    model.add(Input(shape=input_shape))

    # Layer 0: Rescaling operation to map image pixels from [0, 255] to [0, 1] range
    model.add(Rescaling(1.0 / 255, input_shape=input_shape))

    # Layer 1: Convolutional layer with 6 filters, each 3x3 in size,
    # followed by a tanh activation function
    model.add(Conv2D(filters = 6, kernel_size =(3,3),activation = 'tanh'))

    # Layer 2: Average pooling layer with 2x2 pool size
    model.add(AveragePooling2D(pool_size = (2,2))) 

    # Layer 3: Convolutional layer with 16 filters, each 3x3 in size,
    # followed by a tanh activation function
    model.add(Conv2D(filters = 16, kernel_size =(3,3),activation = 'tanh'))

    # Layer 4: Average pooling layer with 2x2 pool size
    model.add(AveragePooling2D(pool_size = (2,2))) 

    # Layer 5: Flatten layer to convert the output of the previous layer to a 1D array
    model.add(Flatten())

    # Layer 6: Fully connected layer with 120 neurons,
    # followed by a tanh activation function
    model.add(Dense(units=120,activation = 'tanh'))

    # Layer 7: Fully connected layer with 84 neurons,
    # followed by a tanh activation function
    model.add(Dense(units=84,activation = 'tanh'))

    # Layer 8: Classification layer with num_classes output units,
    # followed by a softmax activation function
    model.add(Dense(units = num_classes,activation = 'softmax'))

    # Print a summary of the model architecture
    print(model.summary())

    return model


def create_resnet50_model(
    input_shape: Tuple[int, int, int], num_classes: int
) -> Sequential:
    """
    Function to create a convolutional neural network model based on ResNet50
    architecture with transfer learning.

    Args:
        input_shape (Tuple[int, int, int]): The shape of the input data.
        num_classes (int): The number of output classes.

    Returns:
        A Sequential model object.
    """
    # ResNet50 model, transfer learning (fine-tuning).
    # `tensorflow.keras.applications.ResNet50()` with:
    #   1. "imagenet" weights
    #   2. Without the classification layer (include_top=False)
    #   3. input_shape equals to this function input_shape
    resnet = ResNet50(
        weights = 'imagenet',
        input_shape=(input_shape),
        include_top = False
    )

    # Freeze all layers in the ResNet50 model
    for layer in resnet.layers:
        layer.trainable = False

    # Define the model
    model = Sequential()

    # Add the resnet model to the Sequential model.
    model.add(resnet)

    # Add flatten layer to convert the output of the model to a 1D array
    model.add(Flatten())

    # Add a dropout layer with to avoid over-fitting
    model.add(Dropout(0.5))

    # -- > testing: add more layers
    # model.add(Dense(1024,activation = 'relu')) # kernel_regularizer = regularizers.L2(0.01))

    # Add a classification layer with num_classes output units,
    # followed by a softmax activation function
    model.add(Dense(num_classes, activation="softmax"))

    # Print a summary of the model architecture
    print(model.summary())

    return model

    # # to prevent overfitting:
    # add more layers and test
    # add early stopping 
    # add regularization with L1 and L2 parameter <-- test with diferent factors for lambda
    # change batch size 
    # data augmentation?
    # change the patience of the early stopping
    # dynamic learning rate

def create_efficient_model(
        input_shape: Tuple[int, int, int], num_classes: int
) -> Sequential:
    """
    Function that creates a EfficientNetV2S based model, using transfer learning
    
    Args:
        input_shape (Tuple[int, int, int]): The shape of the input data.
        num_classes (int): The number of output classes.

    Returns:
        A Sequential model object.

    """
    # bring the model
    model_efficient = EfficientNetV2S(
    include_top=False,
    weights="imagenet",
    input_shape=(input_shape),
    )
    
    # freeze pretrained layers
    for layer in model_efficient.layers:
        layer.trainable = False

    # create model and add the pretrained one
    model = Sequential()
    model.add(model_efficient)

    # add output layers
    model.add(Flatten())
    # model.add(Dense(1024,activation= 'relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes,activation = 'softmax'))

    print(model.summary())

    return model

def create_other_model(
        input_shape: Tuple[int, int, int], num_classes: int
) -> Sequential:
    """
    Function that creates a EfficientNetV2S based model, using transfer learning
    
    Args:
        input_shape (Tuple[int, int, int]): The shape of the input data.
        num_classes (int): The number of output classes.

    Returns:
        A Sequential model object.

    """
    # bring the model
    model_other = EfficientNetB1(
    include_top=False,
    weights="imagenet",
    input_shape=(input_shape),
    )
    
    # freeze pretrained layers
    for layer in model_other.layers:
        layer.trainable = False

    # create model and add the pretrained one
    model = Sequential()
    model.add(model_other)

    # add output layers
    model.add(Flatten())
    # model.add(Dense(1024,activation= 'relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes,activation = 'softmax'))

    print(model.summary())

    return model


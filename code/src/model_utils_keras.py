"""Keras model utils.

Train model, save model.
"""
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import History
import codecs
import json


def train_model(model: Model, epochs: int):
    """Train vanilla cnn keras model.
    
    Args:
        model: Model, keras model
        epochs: int, training epochs
    Returns:
        model: Model, trained model
        history: History, keras history
    """
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    test_datagen = ImageDataGenerator(
        rescale=1. / 255,
        width_shift_range=0.2,
        height_shift_range=0.2)

    train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        '../../data/AsianSampleCategory/train',
        target_size=(128, 128),
        batch_size=32,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='categorical')
    validation_generator = test_datagen.flow_from_directory(
        '../../data/AsianSampleCategory/val',
        target_size=(128, 128),
        batch_size=20,
        class_mode='categorical')

    # model = vanilla_cnn_keras(input_shape=input_shape, classes=classes)

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=50,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=50)

    return model, history


def save_model(model: Model, history: History, name: str):
    """Save model and history in ../../models folder.
    
    Args:
        model: Model, trained model to save
        history: History, mdoel hisotry to save
        name: str, name for model and history
    Returns:
        None
    """
    model.save('../../models/' + name + '.h5')
    with codecs.open('../../models/' + name + '.json', 'w', 'utf-8') as f:
        json.dump(history.history, f, ensure_ascii=False)
        f.write('\n')
        
        
def load_model_history(name: str):
    """Load model and history from ../../models folder.
    
    Args:
        name: str, model and history name
    Returns:
        model: Model, model from file
        
    """
    model = load_model('../../models/' + name + '.h5')
    with codecs.open('../../models/' + name + '.json', 'r', 'utf-8') as f:
         history = json.load(f)
    return model, history


def load_history(name: str):
    with codecs.open('../../models/' + name + '.json', 'r', 'utf-8') as f:
         history = json.load(f)
    return history
"""Generate image data for algo.'s input."""
from pathlib import Path
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


def gen_category_to_file(data_path: str, tofile_path: str):
    """Generate category from data, write to file.
    
    Args:
        data_path: str, file path to read data
        tofile_path: str, file path to write category
    Returns:
        None
    """
    names = []
    for d in Path(data_path).iterdir():
        names.append(d.name[:6])
    names = '\t'.join(names)
    with open(tofile_path, 'w') as f:
        f.write(names)


def load_data(data_path: str, test_size=0.2, random_state=42, info=False, target_size=(128, 128)):
    """Load data from file.
    
    Args:
        data_path: str, file path to read data
        test_size: float, the ratio of test / all
        random_state: int, random seed to split data
        info: bool, control info print
    Returns:
        train_data: np.array, train set
        test_data: np.array, test set
    """
    imgs = []
    for d in Path(data_path).iterdir():
        if info:
            print('Load image: ' + d.name)
        imgs.append(img_to_array(load_img(d, target_size=target_size)))
    data = np.array(imgs)
    train_data, test_data = train_test_split(
        data, test_size=test_size, random_state=random_state)
    return train_data, test_data


def load_label(label_path: str, test_size=0.2, random_state=42):
    """Load label from file.
    
    Args:
        label_path: str, file path to read label
        test_size: float, the ratio of test / all, 
            this arg should match up test_size in load_data
        random_state: int, random seed to split data
            this arg should match up random_state in load_data
    """
    with open(label_path, 'r') as f:
        labels = f.readline()
    train_label, test_label = train_test_split(
        labels.split('\t'), test_size=test_size, random_state=random_state)
    le = LabelEncoder()
    le.fit(train_label)
    return le.transform(train_label), le.transform(test_label)


def avg_float_asian():
    train_data, test_data = load_data('../../data/Asian/')
    train_label, test_label = load_label('../../data/Asian.txt')
    train_data = train_data.astype('float32') / 255
    test_data = test_data.astype('float32') / 255
    return train_data, test_data, train_label, test_label


def data_augmentation():
    datagen = ImageDataGenerator(
        rotation_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    train_data, _ = load_data('data/Asian/')
    train_label, _ = load_label('data/Asian.txt')
    for _, _ in zip(range(100), datagen.flow(
            train_data, train_label,
            batch_size=50,
            save_to_dir='data/AsianAug',
            save_format='jpg')):
        pass
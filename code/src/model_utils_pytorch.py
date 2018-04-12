"""Pytorch model utils.

Train model, save model.
"""
import codecs
import json
import torch
from torch import save, optim, cuda, load
from torch.nn import Module, CrossEntropyLoss
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os
import time
import copy
from torch.autograd import Variable


def train_model(model: Module, optimizer: optim.Optimizer, epochs: int):
    """Train vanilla cnn pytorch model.
    
    Args:
        model: Module, pytorch model
        epochs: int, training epochs
    Returns:
        model: Module, trained model
        history: dict, history
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(120),
            transforms.Resize((128, 128)),
            transforms.RandomRotation(40),
            transforms.ToTensor(),
        ]),
        'val': transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
        ])
    } 
    data_dir = '../../data/AsianSampleCategory/'
    image_datasets = {x: ImageFolder(os.path.join(data_dir, x), data_transforms[x]) 
                    for x in ['train', 'val']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=32,
                    shuffle=True) for x in ['train', 'val']}
    data_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    use_gpu = cuda.is_available()

    criterion = CrossEntropyLoss()
    # Observe that all parameters are being optimized
#     optimizer = optim.Adam(model.parameters(), lr=1e-4)

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    history = {
        'loss': {'train': [], 'val': []},
        'acc': {'train': [], 'val': []}
    }
    
    for epoch in range(epochs):
        print('Epoch {}/{}:\t'.format(epoch, epochs - 1), end='')

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                    model = model.cuda()
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0] * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / data_sizes[phase]
            epoch_acc = running_corrects / data_sizes[phase]
            
            history['loss'][phase].append(epoch_loss)
            history['acc'][phase].append(epoch_acc)
            print('{} Loss: {:.4f} Acc: {:.4f}\t'.format(
                phase, epoch_loss, epoch_acc), end='')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, history


def save_model(model: Module, history: dict, name: str):
    """Save model and history in ../../models folder.
    
    Args:
        model: Module, trained model to save
        history: dict, mdoel hisotry to save
        name: str, name for model and history
    Returns:
        None
    """
    save(model, '../../models/' + name + '.pkl')
    with codecs.open('../../models/' + name + '.json', 'w', 'utf-8') as f:
        json.dump(history, f, ensure_ascii=False)
        f.write('\n')
        
        
def load_model_history(name: str):
    """Load model and history from ../../models folder.
    
    Args:
        name: str, model and history name
    Returns:
        model: Model, model from file
        
    """
    model = load('../../models/' + name + '.pkl')
    with codecs.open('../../models/' + name + '.json', 'r', 'utf-8') as f:
         history = json.load(f)
    return model, history
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


def get_data_loader(training=True):
    """
    TODO: implement this function.

    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_set = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    test_set = datasets.FashionMNIST('./data', train=False, transform=transform)

    if training:
        return torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=64)
    else:
        return torch.utils.data.DataLoader(test_set, shuffle=False, batch_size=64)


def build_model():
    """
    TODO: implement this function.

    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """
    model = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))
    return model

def train_model(model, train_loader, criterion, T):
    """
    TODO: implement this function.

    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """
    model.train()  # set in training mode
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(T):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            opt.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()

            running_loss += loss.item()
            _, predict = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predict == labels).sum().item()

        print(f'Train Epoch: {epoch} Accuracy:{correct}/{total}({(correct/total)*100:.2f}%) Loss: {running_loss/total:.3f}')
        running_loss=0
        total = 0
        correct = 0



def evaluate_model(model, test_loader, criterion, show_loss=True):
    """
    TODO: implement this function.

    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """
    model.eval()
    with torch.no_grad():
        running_loss = 0.0
        correct = 0
        total = 0
        for data, labels in test_loader:
            outputs = model(data)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predict = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predict == labels).sum().item()
        if show_loss:
            print(f'Average Loss: {running_loss/total:.3f}')
        print(f'Accuracy: {(correct / total) * 100:.2f}%')


def predict_label(model, test_images, index):
    """
    TODO: implement this function.

    INPUT: 
        model - the trained model
        test_images   -  test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """

    _, data = enumerate(test_images)
    output = model(data)
    pred_soft = F.softmax(output)
    _, label = torch.max(pred_soft)
    print(label)


if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    '''
    criterion = nn.CrossEntropyLoss()
    train_loader = get_data_loader(training=True)
    test_loader = get_data_loader()
    print(type(train_loader))
    print(train_loader.dataset)

    model = build_model()
    print(model)

    train_model(model, train_loader, criterion, 1)
    evaluate_model(model, test_loader, criterion, show_loss=True)
    pred_set, _ = next(iter(test_loader))
    predict_label(model, pred_set, 1)
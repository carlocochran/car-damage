import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import os
from torch.utils.data import Dataset, random_split
import pandas as pd
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial
from custom_image_dataset import CustomImageDataset

np.random.seed(42)
BASE_DIR = '/content/drive/MyDrive/Colab Notebooks/ignite/'
# BASE_DIR = '/Users/carlo/Documents/workspace/ignite/'
DATA_PATH = BASE_DIR + 'preprocessed/'
TRAINING_PCT = 0.85
TEST_PCT = 0.15
VALIDATION_PCT = 0.2
NUM_CLASSES = 8
EPOCH = 50
NUM_WORKERS = 2
NUM_CPU = 0
NUM_GPU = 1
SAMPLES = 20


def load_data(data_dir='./'):
    df = pd.read_csv(data_dir + 'index.csv')
    train, test = np.split(df.sample(frac=1), [int(TRAINING_PCT * len(df))])
    trainset = CustomImageDataset(train, path=DATA_PATH)
    testset = CustomImageDataset(test, path=DATA_PATH)
    return trainset, testset

def create_model():
    net = torchvision.models.resnet50()
    net.fc = nn.Linear(2048, NUM_CLASSES)
    return net

def train_model(config, checkpoint_dir=None, data_dir=None):
    net = create_model()

    if torch.cuda.is_available():
        net = net.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=config['lr'], momentum=config['momentum'], weight_decay=config['weight_decay'])

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(os.path.join(checkpoint_dir, "checkpoint"))
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    trainset, testset = load_data(data_dir)

    test_abs = int(len(trainset) * (1.0 - VALIDATION_PCT))
    train_subset, val_subset = random_split(trainset, [test_abs, (len(trainset) - test_abs)])

    # print('train_subset size: {}, test size: {}, val_subset size: {}, test_abs: {}'
    #       .format(len(train_subset), len(testset), len(val_subset), str(test_abs)))

    trainloader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=NUM_WORKERS)
    validateloader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=NUM_WORKERS)

    for epoch in range(EPOCH):
        # training_running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            optimizer.zero_grad()
            outputs = net(inputs)
            training_loss = criterion(outputs, labels)
            training_loss.backward()
            optimizer.step()
            # training_running_loss += training_loss.item()
            # print('{}: training labels: {}'.format(i, labels))

        val_loss = 0.0
        val_steps = 0
        correct = 0
        for i, data in enumerate(validateloader, 0):
            inputs, labels = data
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            net.eval()
            outputs = net(inputs)
            validation_loss = criterion(outputs, labels)
            val_loss += validation_loss.item()
            val_steps += 1

            _, predicted = torch.max(outputs, 1)
            # print('{}: validation predicted: {}'.format(i, predicted))
            # print('{}: validation labels: {}'.format(i, labels))

            correct += torch.sum(predicted == labels)

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((net.state_dict(), optimizer.state_dict()), path)

        # print('********correct:{}, vallen: {}, valsteps: {}'.format(correct, len(val_subset), val_steps))

        tune.report(loss=(val_loss/val_steps), accuracy=(correct/len(val_subset)).item())

def test_accuracy(net, batch_size=4):
    trainset, testset = load_data()
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total

def main(num_samples=10, max_num_epochs=EPOCH, cpus_per_trial=1, gpus_per_trial=1):
    data_dir = DATA_PATH
    load_data(data_dir)
    checkpoint_dir = BASE_DIR
    config = {
        'lr': tune.loguniform(1e-5, 1e-1),
        'momentum': tune.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
        'batch_size': tune.choice([64]),
        'weight_decay': tune.choice([0])
    }

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)

    reporter = CLIReporter(
        metric_columns=["loss", "accuracy", "training_iteration"])

    result = tune.run(
        partial(train_model, checkpoint_dir=checkpoint_dir, data_dir=data_dir),
        resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    best_trained_model = create_model()
    if torch.cuda.is_available():
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
        best_trained_model = best_trained_model.cuda()

    # best_checkpoint_dir = best_trial.checkpoint.value
    # model_state, optimizer_state = torch.load(os.path.join(
    #     best_checkpoint_dir, "checkpoint"))
    # best_trained_model.load_state_dict(model_state)
    #
    # test_acc = test_accuracy(best_trained_model)
    # print("Best trial test set accuracy: {}".format(test_acc))


if __name__ == "__main__":
    main(num_samples=SAMPLES, max_num_epochs=EPOCH, cpus_per_trial=NUM_CPU, gpus_per_trial=NUM_GPU)


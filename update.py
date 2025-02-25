#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs, Y=None, Y_true=None):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]
        self.mal=False
        self.Y_true = Y_true
        if Y is not None:
            self.mal = True
            self.mal_Y = Y

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        if self.mal==True:
            if label == self.Y_true:
                label = self.mal_Y
            return torch.as_tensor(image), torch.as_tensor(label)
        return torch.as_tensor(image), torch.as_tensor(label)


class LocalUpdate(object): 
    def __init__(self, args, dataset, idxs, logger, mal, mal_Y, Y_true, idxs_test=None):
        self.args = args
        self.logger = logger
        self.mal = mal

        if mal is True:
            self.trainloader ,self.testloader= self.mal_loader(dataset, list(idxs), mal_Y, Y_true)
        else:
            self.trainloader, self.testloader = self.train_loader(dataset, list(idxs))
        self.device = 'cuda' if args.gpu else 'cpu'
        # Default criterion set to NLL loss function
        self.criterion = nn.NLLLoss().to(self.device)
        #idxs_int = [int(i) for i in list(idxs)]
        #self.label_list = list(set(np.array(dataset.targets)[idxs_int]))

    def train_loader(self, dataset, idxs):
        idxs_train = idxs[:int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=self.args.local_bs, shuffle=False)
        return trainloader, testloader

    def mal_loader(self, dataset, idxs, Y, Y_true):
        mal_trianloader = DataLoader(DatasetSplit(dataset, idxs, Y, Y_true),
                               batch_size=self.args.local_bs, shuffle=True)
        mal_testloader = DataLoader(DatasetSplit(dataset, idxs, Y, Y_true),
                               batch_size=self.args.local_bs, shuffle=True)
        return mal_trianloader, mal_testloader

    def update_weights(self, model):
        # Set mode to train model
        model.train()
        epoch_loss = []
        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.9)
            #psu_optimizer = torch.optim.SGD(psu_model.parameters(), lr=self.args.lr,
            #                            momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)
            #psu_optimizer = torch.optim.Adam(psu_model.parameters(), lr=self.args.lr,
            #                             weight_decay=1e-4)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, log_probs, labels)

                loss.backward()

                optimizer.step()
                
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)


    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels.to(torch.int64))
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
            
        accuracy = correct/total
        return accuracy, loss


def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if args.gpu else 'cpu'
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels.to(torch.int64))
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss

def mal_inference(args, model, test_dataset, mal_X_list, mal_Y):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct, confidence_sum = 0.0, 0.0, 0.0, 0.0

    device = 'cuda' if args.gpu else 'cpu'
    criterion = nn.NLLLoss().to(device)
    malloader = DataLoader(DatasetSplit(test_dataset, mal_X_list, mal_Y),
                           batch_size=args.local_bs, shuffle=True)

    for batch_idx, (images, labels, labels_true) in enumerate(malloader):
        images, labels, labels_true = images.to(device), labels.to(device), labels_true.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels.to(torch.int64))
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)
        label_list = []
        idx_list = []
        for i in range(len(labels)):
            idx_list.append(int(i))
            label_list.append([int(labels[i].item())])
        confidence_sum += sum((F.softmax(outputs.data.detach(), dim=1).cpu().data)[idx_list, label_list])


    accuracy = correct/total
    confidence = confidence_sum/total
    return accuracy, loss, confidence





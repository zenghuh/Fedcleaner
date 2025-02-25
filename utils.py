#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import heapq
import torch
import numpy as np
from options import args_parser
from torchvision import datasets, transforms
import torch.nn.functional as F
from sampling import emnist_iid, emnist_noniid, emnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid, cifar_extr_noniid, miniimagenet_extr_noniid, mnist_extr_noniid

def get_mal_dataset(dataset, num_mal, num_classes):
    X_list = np.random.choice(len(dataset), num_mal)
    # print("x_list", X_list)
    Y_true = []
    for i in X_list:
        _, Y = dataset[i]
        Y_true.append(Y)
    Y_mal = []
    print(Y_true)
    for i in range(num_mal):
        allowed_targets = list(range(num_classes))
        allowed_targets.remove(Y_true[i])
        Y_mal.append(np.random.choice(allowed_targets))
    print(Y_mal)
    return X_list, Y_mal, Y_true

def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        data_dir = '../../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'emnist':
        
        data_dir = '../../data/emnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
        
        train_dataset = datasets.EMNIST(data_dir, train=True, split="byclass", download=True, transform=apply_transform)
        test_dataset = datasets.EMNIST(data_dir, train=False, split="byclass", download=True, transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = emnist_iid(train_dataset, args.num_users)

        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = emnist_noniid_unequal(train_dataset, args.num_users)

            else:
                # Chose euqal splits for every user
                user_groups = emnist_noniid(train_dataset, args.num_users)

    elif args.dataset == 'fmnist':

        data_dir = '../../data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = emnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = emnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = emnist_noniid(train_dataset, args.num_users)


    return train_dataset, test_dataset, user_groups



def detect_malicious_clients(global_model, client_updates):
    weight_norms = {}
    for name, param in global_model.named_parameters():
        if 'weight' in name:
            weight_norms[name] = torch.norm(param, p=2).item()

    # identify layer
    sl=max(weight_norms, key=weight_norms.get)

    num_clients = len(client_updates)
    
    similarity_matrix = np.zeros((num_clients, num_clients))
    for i in range(num_clients):
        for j in range(i+1, num_clients):
            layer_i = [p for n, p in client_updates[i] if n == sl][0]
            layer_j = [p for n, p in client_updates[j] if n == sl][0]
            similarity_matrix[i][j] = F.cosine_similarity(layer_i, layer_j, dim=0).mean().item()
    
    # calculate threshold
    upper_triangle = similarity_matrix[np.triu_indices(num_clients, k=1)]
    threshold = np.mean(upper_triangle) + 2*np.std(upper_triangle)
    
    malicious = []
    for i in range(num_clients):
        count = 0
        for j in range(num_clients):
            if j != i and similarity_matrix[min(i,j), max(i,j)] < threshold:
                count +=1
        if count > 0.5*num_clients:
            malicious.append(i)
    return malicious

def unlearn(global_model, history, malicious_clients):
    args = args_parser()
    t = len(history)
    n = args.num_users
    
    # Calculate grad delta
    delta_accum = [torch.zeros_like(p) for p in global_model.parameters()]
    for idx in range(t):
        for i in [c for c in range(n) if c not in malicious_clients]:
            grad_i = history[idx][i]
            grad_malicious = history[idx][malicious_clients[0]]
            delta = [gi - gm for gi, gm in zip(grad_i, grad_malicious)]
            for acc, d in zip(delta_accum, delta):
                acc.add_(d)
    
    # add noise
    with torch.no_grad():
        new_params = []
        for param, acc in zip(global_model.parameters(), delta_accum):
            adjustment = args.lr/(n*(n-1)) * acc
            new_p = param - adjustment
            noise = torch.randn_like(new_p) *0.01
            new_params.append(new_p + noise)
        
        for param, new_p in zip(global_model.parameters(), new_params):
            param.copy_(new_p)
    return global_model


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def average_weights_ns(w):
    # def average_weights_ns(w, ns):
    """
    Returns the weighted average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    '''
    for key in w_avg.keys():
    w_avg[key] * ns[0]
    for i in range(1, len(w)):
        w_avg[key] += ns[i] * w[i][key]
    w_avg[key] = torch.div(w_avg[key], sum(ns))
    '''
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return



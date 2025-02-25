from fileinput import filename
import os
import copy
from threading import local
import time
import pickle
import random
from cv2 import dft
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, LeNet, CNNMnist, CNNEMnist, CNNCifar, BasicBlock
from utils import get_dataset, exp_details, detect_malicious_clients, unlearn, average_weights_ns, get_mal_dataset

if __name__ == '__main__':
    # np.random.seed(903)
    torch.manual_seed(313) #cpu
    torch.cuda.manual_seed(322) #gpu
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    device = 'cuda:0' if args.gpu else 'cpu'

    # load dataset and user groups

    train_dataset, test_dataset, user_groups = get_dataset(args)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'emnist':
            global_model = CNNEMnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar()
        elif args.dataset == 'fmnist':
            global_model = CNNMnist(args=args)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # malicious users
    mal_frac=0.2
    num_mal_users = float(args.num_users)*mal_frac
    mal_users = random.sample(range(0, args.num_users), int(num_mal_users))
    print("malcious:", mal_users)
    
    mal_X_list, mal_Y, Y_true = get_mal_dataset(test_dataset, args.num_mal_samples, args.num_classes)
    print("malcious dataset true labels: {}, malicious labels: {}".format(Y_true, mal_Y))

    # Training
    # train_loss, train_accuracy = [], []
    acc, loss, accY = [], [], []
    #val_acc_list, net_list = [], []
    history = []
    # cv_loss, cv_acc = [], []
    # print_every = 1
    # val_loss_pre, counter = 0, 0
    m = max(int(args.frac * args.num_users), 1)

    for epoch in tqdm(range(args.epochs)):
        #local_weights, local_acc, local_losses, local_ns = [], [], [], []
        local_weights = []
        global_model.train()
        
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        #t_adv = False
        #rs_new_ep = []
        #rs_old_ep = []
        for idx in idxs_users:
            mal_user = False
            if idx in mal_users:
                mal_user = True
                #t_adv = True
                print("Malcious user {} is selected!".format(idx))
            
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                          idxs=user_groups[idx], logger=logger, mal=mal_user, mal_Y=mal_Y, Y_true=Y_true)
            w, _ = local_model.update_weights(
                model=copy.deepcopy(global_model))
            # get new model
            new_model = copy.deepcopy(global_model)
            new_model.load_state_dict(w)
            #acc, _ = local_model.inference(model=new_model)
            '''
            if mal_user == True:
                mal_acc, mal_loss = local_model.mal_inference(model=new_model)
            if mal_user == True:
                print('user {}, loss {}, acc {}, mal loss {}, mal acc {}'.format(idx, loss, acc, mal_loss, mal_acc))
            else:
                print('user {}, loss {}, acc {}'.format(idx, loss, acc))
            '''
            local_weights.append(copy.deepcopy(w))
            #local_acc.append(copy.deepcopy(acc))
            #local_losses.append(copy.deepcopy(loss))


        # print(local_ns)
        # global_weights = average_weights_ns(local_weights, local_ns)

        global_weights = average_weights_ns(local_weights)
        history.append([i for _, i in local_weights])
        # detect mal
        malicious = detect_malicious_clients(global_weights, history)

        # update global weights
        if malicious:
            unlearned_weights=unlearn(global_model, history, malicious)
            global_model.load_state_dict(unlearned_weights)

        else:
            # update global weights
            global_model.load_state_dict(global_weights)
        # acc_avg = sum(local_acc)/len(local_acc)
        # loss_avg = sum(local_losses) / len(local_losses)
        # Test inference after completion of training
        test_acc, _= test_inference(args, global_model, test_dataset)
        acc.append(test_acc)
        # loss.append(test_loss)
        # accY.append(test_accY)
        print(f'\n | Global Training Round : {epoch+1} ACC: {test_acc}|\n')
        '''
        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            
            test_acc, test_loss = test_inference(args, global_model, test_dataset)
            print('Global model Benign Test Accuracy: {:.2f}% \n'.format(100*test_acc))
            mal_acc, mal_loss, mal_out = mal_inference(args, global_model, test_dataset, mal_X_list, mal_Y)
            print('Global model Malicious Accuracy: {:.2f}%, Malicious Loss: {:.2f} , Outputs: {}\n'.format(100*mal_acc, 100*mal_loss, mal_out))
            '''
    

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Test Accuracy: {:.2f}%".format(100*acc[-1]))

    # Saving the objects train_loss and train_accuracy:
    file_name = '../save/{}_{}_{}_{}_N[{}].pkl'.\
        format(mal_frac, args.dataset, args.model, args.epochs, args.num_users)

    df = pd.DataFrame({"acc":acc})  
    df.to_pickle(file_name)         
    
    '''
    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)
    '''
    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))


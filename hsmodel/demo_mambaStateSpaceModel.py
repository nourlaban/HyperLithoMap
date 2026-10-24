import torch
import argparse
import torch.nn as nn
import torch.utils.data as Data
import torch.backends.cudnn as cudnn
from scipy.io import loadmat
from scipy.io import savemat
from torch import optim
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import time
import os
from hsmodel.s2mamba import S2Mamba
from hsmodel.utils import *



class Args:
    dataset = 'aboresheed'         # choices: ['Indian', 'Pavia', 'Houston', 'WHU_Hi_LongKou', 'Botswana', 'aboresheed']
    flagtrain    = True
    flagtest    = True
    flagpredict = True
    sess = 'mamba'
    gpu_id = '0'
    seed = 0
    batch_size = 64
    test_freq = 10
    patches = 3
    epochs = 200
    lr = 5e-4
    gamma = 0.99
    weight_decay = 5e-3
    dropout = 0.1

args = Args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
#-------------------------------------------------------------------------------


def main(all_classes):
    if getattr(config, 'epochs', None):
        args.epochs = config.epochs
    setup_seed(args)
    input_normalize, label, num_classes, TR, TE, color_matrix, color_matrix_pred = load_HSI(args)
    height, width, band = input_normalize.shape
    print("height={0},width={1},band={2}".format(height, width, band))
    #-------------------------------------------------------------------------------
    total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true = chooose_train_and_test_point(TR, TE, label, num_classes)
    mirror_image = mirror_hsi(height, width, band, input_normalize, patch=args.patches)
    x_train, x_test = train_and_test_data(mirror_image, band, total_pos_train, total_pos_test, total_pos_true, patch=args.patches)
    y_train, y_test, y_true = train_and_test_label(number_train, number_test, number_true, num_classes)
    #-------------------------------------------------------------------------------
    x_train=torch.from_numpy(x_train.transpose(0,3,1,2)).type(torch.FloatTensor)  # type: ignore
    y_train=torch.from_numpy(y_train).type(torch.LongTensor)  # type: ignore
    Label_train=Data.TensorDataset(x_train,y_train)
    x_test=torch.from_numpy(x_test.transpose(0,3,1,2)).type(torch.FloatTensor)  # type: ignore
    y_test=torch.from_numpy(y_test).type(torch.LongTensor)  # type: ignore
    Label_test=Data.TensorDataset(x_test,y_test)
    label_train_loader=Data.DataLoader(Label_train,batch_size=args.batch_size,shuffle=True)
    label_test_loader=Data.DataLoader(Label_test,batch_size=args.batch_size,shuffle=True)
    # x_true=torch.from_numpy(x_true.transpose(0,3,1,2)).type(torch.FloatTensor)
    # y_true=torch.from_numpy(y_true).type(torch.LongTensor)
    # Label_true=Data.TensorDataset(x_true,y_true)
    # label_true_loader=Data.DataLoader(Label_true,batch_size=100,shuffle=False)
    #-------------------------------------------------------------------------------
    model = S2Mamba(
                in_chans=band,
                patch=args.patches,
                num_classes=num_classes,
                depths=[1],
                dims=[64],
                drop_path_rate=args.dropout,
                attn_drop_rate=args.dropout
            )
    model = model.cuda()
    class_counts = torch.FloatTensor(number_train)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(number_train)
    criterion = nn.CrossEntropyLoss(weight=class_weights.cuda())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    #-------------------------------------------------------------------------------
   



    if args.flagtrain:
        best_OA = 0.0
        best_epoch = 0
        save_path = os.path.join(config.save_model_path, 'mamba_best_model')
        epoch_history = []
        train_start = time.time()
        for epoch in range(args.epochs):
            model.train()
            train_acc, train_obj, tar_t, pre_t = train_epoch(model, label_train_loader, criterion, optimizer)
            scheduler.step()
            print("epoch: {:03d}  |  train_acc: {:.4f} | loss: {:.4f}".format(epoch, train_acc, train_obj))
            if (epoch % args.test_freq == 0) | (epoch == args.epochs - 1):
                model.eval()
                tar_v, pre_v = valid_epoch(model, label_test_loader, criterion, optimizer)
                OA, AA_mean, Kappa, AA = output_metric(tar_v, pre_v)
                print("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA, AA_mean, Kappa))
                epoch_history.append((epoch, float(OA)))
                if OA > best_OA:
                    best_OA = OA
                    best_epoch = epoch
                    torch.save(model, '{}_{}_{}.pt'.format(save_path, args.sess, args.dataset))
                    print("  → Best model saved (OA: {:.4f})".format(best_OA))
        mamba_train_time = time.time() - train_start
        print("**************************************************")
        print("Training time: {:.2f} s".format(mamba_train_time))
        print("Final result: OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA, AA_mean, Kappa))
        eval_dir = os.path.join(config.save_model_path, 'mamba_model_evaluation')
        os.makedirs(eval_dir, exist_ok=True)
        with open(os.path.join(eval_dir, 'mamba_training_time.txt'), 'w') as f:
            f.write(f"Mamba training time: {mamba_train_time:.2f} s\n")
        with open(os.path.join(eval_dir, 'mamba_epoch_history.txt'), 'w') as f:
            f.write("epoch,OA\n")
            for ep, oa in epoch_history:
                f.write(f"{ep},{oa:.4f}\n")
            f.write(f"best_epoch,{best_epoch}\n")
            f.write(f"best_OA,{best_OA:.4f}\n")

    if args.flagtest:
        save_path = os.path.join(config.save_model_path, 'mamba_best_model')
        model = torch.load('{}_{}_{}.pt'.format(save_path,args.sess,args.dataset),  weights_only=False)  
        model.eval()
        start = time.time()
        tar_v, pre_v = valid_epoch(model, label_test_loader, criterion, optimizer)

       # Evaluation
        model_dir = config.save_model_path
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            print(f"✅ Created directory: {model_dir}")   
        evaluate_model(tar_v+1, pre_v+1, classname_dict=all_classes, save_dir=model_dir)
    if args.flagpredict:
        save_path = os.path.join(config.save_model_path, 'mamba_best_model')
        model = torch.load('{}_{}_{}.pt'.format(save_path,args.sess,args.dataset),  weights_only=False)   
        model.eval()
        start = time.time()
        with rasterio.open(config.tifpath) as src:
                input_data = src.read()
        mirror_image = np.transpose(input_data, (1, 2, 0))   
        pre_map = predict_whole_image(model, height, width, band, mirror_image, args)
        end = time.time()
        print("time: {:.4f}s".format(end - start))
        # cm = confusion_matrix(label.flatten(), pre_map.flatten())
        # OA, AA_mean, Kappa, AA = calculate_overall_accuracy(cm)
        # print("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA, AA_mean, Kappa))
        # # save the predicted map
        # savemat('./{}_{}_pred_map.mat'.format(args.sess,args.dataset), {'pred_map': pre_map})
        # # plot the predicted map
        # plot_predicted_map(pre_map, args.dataset, color_matrix_pred)

 









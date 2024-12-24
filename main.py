import os
import re
import sys
import time
import math
import torch
import pickle
import argparse
import itertools
import numpy as np
from torch import nn
from torch import optim
from datetime import datetime
import torch.nn.functional as F
from torchvision import transforms
from torcheval.metrics import Mean
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from torchvision.transforms import InterpolationMode
from torchtnt.utils.loggers import TensorBoardLogger, JSONLogger

ROOT = os.path.join(os.path.dirname(__file__), os.path.pardir)
sys.path.append(os.path.join(ROOT, "DTML"))

import utils
from losses import CompLoss
from datasets import Oracle241, Digit10, Oracle50K_OBI125
from datasets.transform import get_dataset_transform
from val import sample_images, generate_image_classification_test
from models import DTN, LeNet, GeneratorResNet, Discriminator, ResNetFc


def evaluate(config, loader):
    "Create model"
    # source -> target generator
    G_S2T = GeneratorResNet(config["dataset"]["image_shape"], config["model"]["residual_blocks"], config["model"]["kernel_size"]).cuda()
    # target -> source generator
    G_T2S = GeneratorResNet(config["dataset"]["image_shape"], config["model"]["residual_blocks"], config["model"]["kernel_size"]).cuda()
    # ResNet classifier
    if re.match(r"^ResNet\d{2,3}$", config["model"]["classifier"]):
        # source domain classifier
        CS_S = ResNetFc(config["model"]["classifier"], use_bottleneck=True, bottleneck_dim=256, new_cls=True, class_num=config["dataset"]["class_num"]).cuda()
        # target domain classifier
        CS_T = ResNetFc(config["model"]["classifier"], use_bottleneck=True, bottleneck_dim=256, new_cls=True, class_num=config["dataset"]["class_num"]).cuda()
    # LeNet classifier
    elif config["model"]["classifier"] == "LeNet":
        # source domain classifier
        CS_S = LeNet(config["dataset"]["image_shape"]).cuda()
        # target domain classifier
        CS_T = LeNet(config["dataset"]["image_shape"]).cuda()
    elif config["model"]["classifier"] == "DTN":
        # source domain classifier
        CS_S = DTN(config["dataset"]["image_shape"]).cuda()
        # target domain classifier
        CS_T = DTN(config["dataset"]["image_shape"]).cuda()
    weights_path = config["dir"]["weight"]
    # load weights
    G_S2T.load_state_dict(torch.load(os.path.join(weights_path, "BEST_G_S2T.pth")))
    G_T2S.load_state_dict(torch.load(os.path.join(weights_path, "BEST_G_T2S.pth")))
    CS_S.load_state_dict(torch.load(os.path.join(weights_path, "BEST_RN_S.pth")))
    CS_T.load_state_dict(torch.load(os.path.join(weights_path, "BEST_RN_T.pth")))
    accuracy, precision, recall, f1Score = generate_image_classification_test(loader["test_target"]["acc"], G_T2S, CS_T, CS_S, num_classes=config["dataset"]["class_num"])
    print(f"Accuracy | Better: {accuracy[0] * 100:.2f}, Target: {accuracy[1] * 100:.2f}, Source: {accuracy[2] * 100:.2f}")
    print(f"Precision | Better: {precision[0] * 100:.2f}, Target: {precision[1] * 100:.2f}, Source: {precision[2] * 100:.2f}")
    print(f"Recall | Better: {recall[0] * 100:.2f}, Target: {recall[1] * 100:.2f}, Source: {recall[2] * 100:.2f}")
    print(f"F1Score | Better: {f1Score[0] * 100:.2f}, Target: {f1Score[1] * 100:.2f}, Source: {f1Score[2] * 100:.2f}")
    

def train(config, loader):
    "Create model"
    # source -> target generator
    G_S2T = GeneratorResNet(config["dataset"]["image_shape"], config["model"]["residual_blocks"], config["model"]["kernel_size"]).cuda()
    # target -> source generator
    G_T2S = GeneratorResNet(config["dataset"]["image_shape"], config["model"]["residual_blocks"], config["model"]["kernel_size"]).cuda()
    # source discriminator
    D_S = Discriminator(config["dataset"]["image_shape"], config["model"]["discrim_blocks"]).cuda()
    # target discriminator
    D_T = Discriminator(config["dataset"]["image_shape"], config["model"]["discrim_blocks"]).cuda()
    # ResNet classifier
    if re.match(r"^ResNet\d{2,3}$", config["model"]["classifier"]):
        # source domain classifier
        CS_S = ResNetFc(config["model"]["classifier"], use_bottleneck=True, bottleneck_dim=256, new_cls=True, class_num=config["dataset"]["class_num"]).cuda()
        # target domain classifier
        CS_T = ResNetFc(config["model"]["classifier"], use_bottleneck=True, bottleneck_dim=256, new_cls=True, class_num=config["dataset"]["class_num"]).cuda()
    # LeNet classifier
    elif config["model"]["classifier"] == "LeNet":
        # source domain classifier
        CS_S = LeNet(config["dataset"]["image_shape"]).cuda()
        # target domain classifier
        CS_T = LeNet(config["dataset"]["image_shape"]).cuda()
    elif config["model"]["classifier"] == "DTN":
        # source domain classifier
        CS_S = DTN(config["dataset"]["image_shape"]).cuda()
        # target domain classifier
        CS_T = DTN(config["dataset"]["image_shape"]).cuda()

    "Model weight initialization"
    # training from scratch
    if config["iteration"]["start_step"] == 0:
        # init weights (CS_S/T loaded pretraining weights during creation)
        G_S2T.apply(utils.weights_init_normal)
        G_T2S.apply(utils.weights_init_normal)
        D_S.apply(utils.weights_init_normal)
        D_T.apply(utils.weights_init_normal)
    # training from checkpoint
    else:
        weights_path = config["dir"]["weight"]
        # load weights
        G_S2T.load_state_dict(torch.load(os.path.join(weights_path, "G_S2T_{}.pth".format(config["iteration"]["start_step"]))))
        G_T2S.load_state_dict(torch.load(os.path.join(weights_path, "G_T2S_{}.pth".format(config["iteration"]["start_step"]))))
        D_S.load_state_dict(torch.load(os.path.join(weights_path, "D_S_{}.pth".format(config["iteration"]["start_step"]))))
        D_T.load_state_dict(torch.load(os.path.join(weights_path, "D_T_{}.pth".format(config["iteration"]["start_step"]))))
        CS_S.load_state_dict(torch.load(os.path.join(weights_path, "RN_S_{}.pth".format(config["iteration"]["start_step"]))))
        CS_T.load_state_dict(torch.load(os.path.join(weights_path, "RN_T_{}.pth".format(config["iteration"]["start_step"]))))
    # D_S/D_T output shape
    output_shape = D_S.output_shape
    
    "Create save dirs"
    # root = dir/dataset/s2t/model_name
    root = os.path.join(config["dir"]["save"], config["dataset"]["name"], "{}2{}".format(config["dataset"]["source"], config["dataset"]["target"]))
    # dir/dataset/s2t/model_name/date/checkpoint
    checkpoint_path = os.path.join(root, config["name"], config["date"],  "checkpoint")
    # dir/dataset/s2t/model_name/date/images
    image_path = os.path.join(root, config["name"], config["date"], "images")
    # dir/dataset/s2t/model_name/date/log
    log_path = os.path.join(root, config["name"], config["date"], "log")
    # dir/dataset/s2t/tensorboard/model_name/date
    tensorboard_path = os.path.join(root, "tensorboard", config["name"], config["date"])
    # make dirs
    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(image_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(tensorboard_path, exist_ok=True)
    
    "Log information"
    # save hyperparameters
    with open(os.path.join(log_path, "config.pk"), "wb") as f:
        pickle.dump(config, f)
    # save training result in tensorboard
    tb_logger = TensorBoardLogger(tensorboard_path)
    # save training result in json file
    js_logger = JSONLogger(os.path.join(log_path, "log.json"), steps_before_flushing=100)
    
    "Record losses"
    # identity loss
    losses_id_S = Mean(device="cuda") # identity loss from source domain
    losses_id_S.reset()
    losses_id_T = Mean(device="cuda") # identity loss from target domain
    losses_id_T.reset()
    losses_id = Mean(device="cuda") # comprehensive identity loss
    losses_id.reset()
    # GAN loss
    losses_GAN_S2T = Mean(device="cuda") # GAN loss from source domain to target domain
    losses_GAN_S2T.reset()
    losses_GAN_T2S = Mean(device="cuda") # GAN loss from target domain to target domain
    losses_GAN_T2S.reset()
    losses_GAN = Mean(device="cuda") # comprehensive GAN loss from both domains
    losses_GAN.reset()
    # cycle loss
    losses_cycle_S = Mean(device="cuda") # cycle loss from source domain
    losses_cycle_S.reset()
    losses_cycle_T = Mean(device="cuda") # cycle loss from target domain
    losses_cycle_T.reset()
    losses_cycle = Mean(device="cuda") # comprehensive cycle loss from both domains
    losses_cycle.reset()
    # generator loss
    losses_G = Mean(device="cuda") # comprehensive generator loss
    losses_G.reset()
    # source discriminator loss
    losses_real_S = Mean(device="cuda") # discriminant loss of real samples from the source domain
    losses_real_S.reset()
    losses_fake_S = Mean(device="cuda") # discriminant loss of fake samples from the source domain
    losses_fake_S.reset()
    losses_D_S = Mean(device="cuda") # comprehensive discriminant loss from the source domain
    losses_D_S.reset()
    # target discriminator loss
    losses_real_T = Mean(device="cuda") # discriminant loss of real samples from the target domain
    losses_real_T.reset()
    losses_fake_T = Mean(device="cuda") # discriminant loss of fake samples from the target domain
    losses_fake_T.reset()
    losses_D_T = Mean(device="cuda") # comprehensive discriminant loss from the target domain
    losses_D_T.reset()
    # discriminator loss
    losses_D = Mean(device="cuda") # comprehensive discriminant loss from both domains
    losses_D.reset()
    # Classification loss
    losses_C_S_real = Mean(device="cuda") # classification loss of real samples from the source domain
    losses_C_S_real.reset()
    losses_C_S_recov = Mean(device="cuda") # classification loss of recover samples from the source domain
    losses_C_S_recov.reset()
    losses_C_T_fake = Mean(device="cuda") # classification loss of fake samples from the target domain
    losses_C_T_fake.reset()
    losses_C_S = Mean(device="cuda") # comprehensive classification loss
    losses_C_S.reset()
    losses_P_S = Mean(device="cuda") # pseudo loss of source labels
    losses_P_S.reset()
    losses_P_T = Mean(device="cuda") # pseudo loss of target labels
    losses_P_T.reset()
    losses_P = Mean(device="cuda") # pseudo loss of target domain
    losses_P.reset()
    losses_con_S = Mean(device="cuda") # contrastive loss of source domain
    losses_con_S.reset()
    losses_con_T = Mean(device="cuda") # contrastive loss of target domain
    losses_con_T.reset()
    losses_con = Mean(device="cuda") # contrastive loss
    losses_con.reset()
    losses_DA = Mean(device="cuda")
    losses_DA.reset()

    "Records pseudo"
    masks_rate = Mean(device="cuda")
    masks_rate.reset()
    masks_num = Mean(device="cuda")
    masks_num.reset()
    masks_correct_num = Mean(device="cuda")
    masks_correct_num.reset()


    "Loss function"
    # GAN loss
    criterion_GAN = nn.MSELoss()
    # cycle loss
    criterion_cycle = nn.L1Loss()
    # identity loss
    criterion_identity = nn.L1Loss()
    # classify loss
    criterion_classify = nn.CrossEntropyLoss()
    # comparision loss
    criterion_comp = CompLoss(temperature=config["parm"]["temperature"], class_mode=config["parm"]["class_mode"])

    "optimizer"
    # source discriminator optimizer
    optimizer_D_S = config["optim"]["type"](D_S.parameters(), **config["optim"]["DT"])
    # target discriminator optimizer
    optimizer_D_T = config["optim"]["type"](D_T.parameters(), **config["optim"]["DT"])
    # DA optimizer
    optimizer_DA = config["optim"]["type"](itertools.chain(G_S2T.parameters(), G_T2S.parameters(), CS_S.parameters(), CS_T.parameters()), **config["optim"]["DT"])

    "scheduler"
    lf = lambda x: ((1 + math.cos((x - config["scheduler"]["decay"]) * math.pi / (config["iteration"]["num_steps"] - config["scheduler"]["decay"]))) / 2) * (1 - config["scheduler"]["lrf"]) + config["scheduler"]["lrf"] if x >= 20000 else 1
    # source discriminator scheduler
    lr_scheduler_D_S = optim.lr_scheduler.LambdaLR(optimizer_D_S, lr_lambda=lf)
    # target discriminator scheduler
    lr_scheduler_D_T = optim.lr_scheduler.LambdaLR(optimizer_D_T, lr_lambda=lf)
    # DA scheduler
    lr_scheduler_DA = optim.lr_scheduler.LambdaLR(optimizer_DA, lr_lambda=lf)

    "Multi cudas parallel"
    # Number of available cudas
    num_cudas = len(config["device"]["cuda"].split(","))
    # Multi cudas data parallel setting
    if num_cudas > 1:
        # available cuda ids
        device_ids = list(range(num_cudas))
        # Model multi cudas output parallel
        G_S2T = nn.DataParallel(G_S2T, device_ids=device_ids)
        G_T2S = nn.DataParallel(G_T2S, device_ids=device_ids)
        D_S = nn.DataParallel(D_S, device_ids=device_ids)
        D_T = nn.DataParallel(D_T, device_ids=device_ids)
        CS_S = nn.DataParallel(CS_S, device_ids=device_ids)
        CS_T = nn.DataParallel(CS_T, device_ids=device_ids)

    "training"
    # source domain data loader
    iter_source = iter(loader["train_source"])
    # target domain data loader
    iter_target = iter(loader["train_target"])
    # source domain fake sample cache
    fake_S_buffer = utils.ReplayBuffer()
    # target domain fake sample cache
    fake_T_buffer = utils.ReplayBuffer()
    # source/target domain best accuracy
    source_best_acc = target_best_acc = 0.0
    # best step for source/target best accuracy
    source_best_step = target_best_step = -1
    loss_C_S = torch.tensor(10)
    # training start time
    begin = time.time()
    # train from the start step to num steps
    for step in range(config["iteration"]["start_step"], config["iteration"]["num_steps"]):
        # get source domain data
        source_imgs, source_labels = next(iter_source)
        # get target domain data
        target_imgs, target_labels = next(iter_target)
        # load data into cuda
        S_real, source_labels, T_real, target_labels = source_imgs.cuda(), source_labels.cuda(), target_imgs.cuda(), target_labels.cuda()
        # ground truth for real samples
        valid = torch.ones((S_real.size(0), *output_shape), requires_grad=False).cuda()
        # ground truth for fake samples
        fake = torch.zeros((T_real.size(0), *output_shape), requires_grad=False).cuda()

        "DT trainer"
        G_S2T.train()
        G_T2S.train()
        optimizer_DA.zero_grad()
        # Identity loss for source real samples
        loss_id_S = criterion_identity(G_T2S(S_real), S_real)
        losses_id_S.update(loss_id_S)
        # Identity loss for target real samples
        loss_id_T = criterion_identity(G_S2T(T_real), T_real)
        losses_id_T.update(loss_id_T)
        # Average identity loss
        loss_identity = (loss_id_S + loss_id_T) / 2
        losses_id.update(loss_identity)
        # Generate fake samples for target domain
        T_fake = G_S2T(S_real)
        # GAN loss loss for source -> target generator
        loss_GAN_S2T = criterion_GAN(D_T(T_fake), valid)
        losses_GAN_S2T.update(loss_GAN_S2T)
        # Generate fake samples for source domain
        S_fake = G_T2S(T_real)
        # GAN loss for target -> source generator
        loss_GAN_T2S = criterion_GAN(D_S(S_fake), valid)
        losses_GAN_T2S.update(loss_GAN_T2S)
        # Average generate loss
        loss_GAN = (loss_GAN_S2T + loss_GAN_T2S) / 2
        losses_GAN.update(loss_GAN)
        # Recover source domain samples
        S_recov = G_T2S(T_fake)
        # Cycle loss for source domain
        loss_cycle_S = criterion_cycle(S_recov, S_real)
        losses_cycle_S.update(loss_cycle_S)
        # Recover target domain samples
        T_recov = G_S2T(S_fake)
        # Cycle loss for target domain
        loss_cycle_T = criterion_cycle(T_recov, T_real)
        losses_cycle_T.update(loss_cycle_T)
        # Average cycle loss
        loss_cycle = (loss_cycle_S + loss_cycle_T) / 2
        losses_cycle.update(loss_cycle)
        # loss comp
        S_real_CS_features, S_real_CS_outputs = CS_S(S_real)
        T_fake_CS_features, T_fake_CS_outputs = CS_S(T_fake)
        S_real_CT_features, S_real_CT_outputs = CS_T(S_real)
        T_fake_CT_features, T_fake_CT_outputs = CS_T(T_fake)

        T_real_CT_features, T_real_CT_outputs = CS_T(T_real)
        S_fake_CT_features, S_fake_CT_outputs = CS_T(S_fake)
        T_real_CS_features, T_real_CS_outputs = CS_S(T_real)
        S_fake_CS_features, S_fake_CS_outputs = CS_S(S_fake)
        # comparision loss
        loss_con_S = (criterion_comp(T_fake_CS_outputs, S_real_CS_outputs) + criterion_comp(T_fake_CT_outputs, S_real_CT_outputs)) / 2
        loss_con_T = (criterion_comp(S_fake_CT_outputs, T_real_CT_outputs) + criterion_comp(S_fake_CS_outputs, T_real_CS_outputs)) / 2
        losses_con_S.update(loss_con_S)
        losses_con_T.update(loss_con_T) 
        loss_con = (loss_con_S + loss_con_T) / 2

        losses_con.update(loss_con)
        # Generate loss
        loss_G = config["parm"]["lambda"]["GAN"] * loss_GAN + config["parm"]["lambda"]["cyc"] * loss_cycle + config["parm"]["lambda"]["id"] * loss_identity + config["parm"]["lambda"]["con"] * loss_con
        losses_G.update(loss_G)

        "ML trainer"
        CS_S.train()
        CS_T.train()
        # Prediction results of source domain real samples
        S_real_features, S_real_outputs = CS_S(S_real)
        # Classification loss of source domain real samples
        loss_C_S_real = criterion_classify(S_real_outputs, source_labels)
        losses_C_S_real.update(loss_C_S_real)
        # Prediction results of source domain recover samples
        S_recov_features, S_recov_outputs = CS_S(S_recov)
        # Classification loss of source domain recover samples
        loss_C_S_recov = criterion_classify(S_recov_outputs, source_labels)
        losses_C_S_recov.update(loss_C_S_recov)
        # Prediction results of target domain fake samples
        T_fake_features, T_fake_outputs = CS_T(T_fake)
        # Classification loss of target domain fake samples
        loss_C_T_fake = criterion_classify(T_fake_outputs, source_labels)
        losses_C_T_fake.update(loss_C_T_fake)
        # Average classification loss
        loss_C_S = (loss_C_S_real + loss_C_S_recov + loss_C_T_fake) / 3
        losses_C_S.update(loss_C_S)

        # Prediction results of source domain fake samples
        S_fake_features, S_fake_outputs = CS_S(S_fake)
        # Prediction results of target domain real samples
        T_real_features, T_real_outputs = CS_T(T_real)
        # Model preheating completed
        if step > config["iteration"]["preheat"]:
            # Predicted probability distribution of source domain fake samples
            S_fake_probs = F.softmax(S_fake_outputs, dim=-1)
            # Predicted maximum probability and index of source domain fake samples
            max_S_fake_probs, max_S_fake_indices = torch.max(S_fake_probs, dim=-1)
            # The confidence level of prediction
            high_confidence_mask = max_S_fake_probs > config["parm"]["confidence_threshold"]
            # Select pseudo labels for source domain fake samples
            pseudo_S_fake_labels = max_S_fake_indices[high_confidence_mask]
            # Select the corresponding target domain real samples
            high_confidence_T_real = T_real[high_confidence_mask]
            # log
            mask_num = high_confidence_mask.sum()
            masks_num.update(mask_num)
            masks_rate.update(mask_num / config["iteration"]["batch_size"])
            if mask_num == 0:
                masks_correct_num.update(torch.tensor(0))
            else:
                masks_correct_num.update(torch.sum(torch.eq(pseudo_S_fake_labels, target_labels[high_confidence_mask])))
            # If there are high confidence samples present
            if high_confidence_T_real.shape[0] > 0:
                # Prediction results of high confidence target domain real samples
                _, high_confidence_T_real_outputs = CS_T(high_confidence_T_real)
                # Pseudo loss for source domain labels
                loss_P_S = criterion_classify(high_confidence_T_real_outputs, pseudo_S_fake_labels)
            # No high confidence samples
            else:
                # No loss
                loss_P_S = torch.tensor([0.]).cuda()
        
            # Predicted probability distribution of target domain real samples
            T_real_probs = F.softmax(T_real_outputs, dim=-1)
            # Predicted maximum probability and index of target domain real samples
            max_T_real_probs, max_T_real_indices = torch.max(T_real_probs, dim=-1)
            # The confidence level of prediction
            high_confidence_mask = max_T_real_probs > config["parm"]["confidence_threshold"]
            # Select pseudo labels for target domain real samples
            pseudo_T_real_labels = max_T_real_indices[high_confidence_mask]
            # Select the corresponding source domain fake samples
            high_confidence_S_fake = S_fake[high_confidence_mask]
            mask_num = high_confidence_mask.sum()
            masks_num.update(mask_num)
            masks_rate.update(mask_num / config["iteration"]["batch_size"])
            if mask_num == 0:
                masks_correct_num.update(torch.tensor(0))
            else:
                masks_correct_num.update(torch.sum(torch.eq(pseudo_T_real_labels, target_labels[high_confidence_mask])))
            # If there are high confidence samples present
            if high_confidence_S_fake.shape[0] > 0:
                # Prediction results of high confidence source domain fake samples
                _, high_confidence_S_fake_outputs = CS_S(high_confidence_S_fake)
                # Pseudo loss for target domain labels
                loss_P_T = criterion_classify(high_confidence_S_fake_outputs, pseudo_T_real_labels)
            # No high confidence samples
            else:
                # No loss
                loss_P_T = torch.tensor([0.]).cuda()
        # Model preheating not completed
        else:
            # No loss
            loss_P_S = torch.tensor([0.]).cuda()
            loss_P_T = torch.tensor([0.]).cuda()
        losses_P_S.update(loss_P_S)
        losses_P_T.update(loss_P_T)
        # Average pseudo loss
        loss_P = (loss_P_S + loss_P_T) / 2
        losses_P.update(loss_P)
        # DA loss
        loss_DA = config["parm"]["lambda"]["pseudo"] * loss_P + config["parm"]["lambda"]["cls"] * loss_C_S + loss_G
        losses_DA.update(loss_DA)
        # DA backpropagation
        loss_DA.backward()
        optimizer_DA.step()

        "discriminator trainer"
        "source domain discriminator"
        D_S.train()
        optimizer_D_S.zero_grad()
        # Discrimination loss for source domain real samples
        loss_real = criterion_GAN(D_S(S_real), valid)
        losses_real_S.update(loss_real)
        # Pushing and poping fake source samples
        S_fake_ = fake_S_buffer.push_and_pop(S_fake)
        # Discrimination loss for previous source domain fake samples
        loss_fake = criterion_GAN(D_S(S_fake_.detach()), fake)
        losses_fake_S.update(loss_fake)
        # Average discrimination for source domain
        loss_D_S = (loss_real + loss_fake) / 2
        losses_D_S.update(loss_D_S)
        # Source domain discriminator backpropagation
        loss_D_S.backward()
        optimizer_D_S.step()
        "target domain discriminator"
        D_T.train()
        optimizer_D_T.zero_grad()
        # Discrimination loss for target domain real samples
        loss_real = criterion_GAN(D_T(T_real), valid)
        losses_real_T.update(loss_real)
        # Pushing and poping fake target samples
        T_fake_ = fake_T_buffer.push_and_pop(T_fake)
        losses_fake_T.update(loss_fake)
        # Discrimination loss for previous target domain fake samples
        loss_fake = criterion_GAN(D_T(T_fake_.detach()), fake)
        # Average discrimination for target domain
        loss_D_T = (loss_real + loss_fake) / 2
        losses_D_T.update(loss_D_T)
        # Source domain discriminator backpropagation
        loss_D_T.backward()
        optimizer_D_T.step()
        # Average discriminator loss
        loss_D = (loss_D_S + loss_D_T) / 2
        losses_D.update(loss_D)

        "update learning rate"
        lr_scheduler_D_S.step()
        lr_scheduler_D_T.step()
        lr_scheduler_DA.step()

        "logging"
        if config["interval"]["log"] != -1 and step % config["interval"]["log"] == 0:
            # compute
            time_count = utils.time_counter(begin, time.time())
            loss_id_S = losses_id_S.compute()
            loss_id_T = losses_id_T.compute()
            loss_id = losses_id.compute()
            loss_GAN_S2T = losses_GAN_S2T.compute()
            loss_GAN_T2S = losses_GAN_T2S.compute()
            loss_GAN = losses_GAN.compute()
            loss_cycle_S = losses_cycle_S.compute()
            loss_cycle_T = losses_cycle_T.compute()
            loss_cycle = losses_cycle.compute()
            loss_G = losses_G.compute()
            loss_real_S = losses_real_S.compute()
            loss_fake_S = losses_fake_S.compute()
            loss_D_S = losses_D_S.compute()
            loss_real_T = losses_real_T.compute()
            loss_fake_T = losses_fake_T.compute()
            loss_D_T = losses_D_T.compute()
            loss_D = losses_D.compute()
            loss_C_S_real = losses_C_S_real.compute()
            loss_C_S_recov = losses_C_S_recov.compute()
            loss_C_T_fake = losses_C_T_fake.compute()
            loss_C_S = losses_C_S.compute()
            loss_P_S = losses_P_S.compute()
            loss_P_T = losses_P_T.compute()
            loss_P = losses_P.compute()
            loss_con_S = losses_con_S.compute()
            loss_con_T = losses_con_T.compute()
            loss_con = losses_con.compute()
            loss_DA = losses_DA.compute()
            mask_rate = masks_rate.compute()
            mask_num = masks_num.compute()
            mask_correct_num = masks_correct_num.compute()
            mask_purity = 0 if mask_num == 0 else mask_correct_num / mask_num
            # print
            sys.stdout.write(
                    "\r[Step %06d/%06d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f, contrastive: %f] [DA loss: %f, cls: %f, pseudo: %f] time: %s \n"
                % (
                    step,
                    config["iteration"]["num_steps"],
                    loss_D,
                    loss_G,
                    loss_GAN,
                    loss_cycle,
                    loss_identity,
                    loss_con,
                    loss_DA,
                    loss_C_S,
                    loss_P,
                    time_count
                )
            )
            log_dict = {
                "loss_id_S": loss_id_S,
                "loss_id_T": loss_id_T,
                "loss_id": loss_id,
                "loss_GAN_S2T": loss_GAN_S2T,
                "loss_GAN_T2S": loss_GAN_T2S,
                "loss_GAN": loss_GAN,
                "loss_cycle_S": loss_cycle_S,
                "loss_cycle_T": loss_cycle_T,
                "loss_cycle": loss_cycle,
                "loss_G": loss_G,
                "loss_real_S": loss_real_S,
                "loss_fake_S": loss_fake_S,
                "loss_D_S": loss_D_S,
                "loss_real_T": loss_real_T,
                "loss_fake_T": loss_fake_T,
                "loss_D_T": loss_D_T,
                "loss_D": loss_D,
                "loss_C_S_real": loss_C_S_real,
                "loss_C_S_recov": loss_C_S_recov,
                "loss_C_T_fake": loss_C_T_fake,
                "loss_C_S": loss_C_S,
                "loss_P_S": loss_P_S,
                "loss_P_T": loss_P_T,
                "loss_P": loss_P,
                "loss_con_S": loss_con_S,
                "loss_con_T": loss_con_T,
                "loss_con": loss_con,
                "loss_DA": loss_DA,
                "mask_rate": mask_rate,
                "mask_num": mask_num,
                "mask_correct_num": mask_correct_num,
                "mask_purity": mask_purity
            }
            # logger
            js_logger.log_dict(log_dict, step)
            tb_logger.log_dict(log_dict, step)
            # reset
            losses_id_S.reset()
            losses_id_T.reset()
            losses_id.reset()
            losses_GAN_S2T.reset()
            losses_GAN_T2S.reset()
            losses_GAN.reset()
            losses_cycle_S.reset()
            losses_cycle_T.reset()
            losses_cycle.reset()
            losses_G.reset()
            losses_real_S.reset()
            losses_fake_S.reset()
            losses_D_S.reset()
            losses_real_T.reset()
            losses_fake_T.reset()
            losses_D_T.reset()
            losses_D.reset()
            losses_C_S_real.reset()
            losses_C_S_recov.reset()
            losses_C_T_fake.reset()
            losses_C_S.reset()
            losses_P_S.reset()
            losses_P_T.reset()
            losses_P.reset()
            losses_con_S.reset()
            losses_con_T.reset()
            losses_con.reset()
            losses_DA.reset()
            masks_rate.reset()
            masks_num.reset()
            masks_correct_num.reset()
        
        "img show"
        if config["interval"]["img"] != -1 and step % config["interval"]["img"] == 0:
            sample_images(G_S2T, G_T2S, loader["test_source"], loader["test_target"]["img"], image_path, step)

        "test"
        if config["interval"]["test"] != -1 and step % config["interval"]["test"] == 0:
            # target test
            t_accuracy, t_precision, t_recall, t_f1Score = generate_image_classification_test(loader["test_target"]["acc"], G_T2S, CS_T, CS_S, num_classes=config["dataset"]["class_num"])
            # save best details
            if t_accuracy[0] > target_best_acc:
                target_best_acc = t_accuracy[0]
                target_best_step = step
                torch.save(G_S2T.state_dict(), os.path.join(checkpoint_path, f"BEST_G_S2T.pth"))
                torch.save(G_T2S.state_dict(), os.path.join(checkpoint_path, f"BEST_G_T2S.pth"))
                torch.save(D_S.state_dict(), os.path.join(checkpoint_path, f"BEST_D_S.pth"))
                torch.save(D_T.state_dict(), os.path.join(checkpoint_path, f"BEST_D_T.pth"))
                torch.save(CS_S.state_dict(), os.path.join(checkpoint_path, f"BEST_RN_S.pth"))
                torch.save(CS_T.state_dict(), os.path.join(checkpoint_path, f"BEST_RN_T.pth"))
            # source test
            s_accuracy, s_precision, s_recall, s_f1Score = generate_image_classification_test(loader["test_source"], G_S2T, CS_S, CS_T, num_classes=config["dataset"]["class_num"])
            # save best details
            if s_accuracy[0] > source_best_acc:
                source_best_acc = s_accuracy[0]
                source_best_step = step
            # log
            sys.stdout.write(
                    "\r[Step %06d/%06d] [T temp acc: %f, 1: %f, 2: %f, best acc: %f] [S temp acc: %f, 1: %f, 2: %f, best acc: %f] time: %s \n"
                % (
                    step,
                    config["iteration"]["num_steps"],
                    t_accuracy[0],
                    t_accuracy[1],
                    t_accuracy[2],
                    target_best_acc,
                    s_accuracy[0],
                    s_accuracy[1],
                    s_accuracy[2],
                    source_best_acc,
                    time_count
                )
            )
            log_dict = {"target_temp_acc": t_accuracy[0], 
                        "target_real_acc": t_accuracy[1],
                        "target_fake_acc": t_accuracy[2],
                        "target_temp_pre": t_precision[0], 
                        "target_real_pre": t_precision[1],
                        "target_fake_pre": t_precision[2],
                        "target_temp_rec": t_recall[0], 
                        "target_real_rec": t_recall[1],
                        "target_fake_rec": t_recall[2],
                        "target_temp_f1": t_f1Score[0], 
                        "target_real_f1": t_f1Score[1],
                        "target_fake_f1": t_f1Score[2],
                        "target_best_acc": target_best_acc, 
                        "target_best_step": target_best_step,
                        "source_temp_acc": s_accuracy[0],
                        "source_real_acc": s_accuracy[1],
                        "source_fake_acc": s_accuracy[2],
                        "source_temp_pre": s_precision[0], 
                        "source_real_pre": s_precision[1],
                        "source_fake_pre": s_precision[2],
                        "source_temp_rec": s_recall[0], 
                        "source_real_rec": s_recall[1],
                        "source_fake_rec": s_recall[2],
                        "source_temp_f1": s_f1Score[0], 
                        "source_real_f1": s_f1Score[1],
                        "source_fake_f1": s_f1Score[2],
                        "source_best_acc": source_best_acc, 
                        "source_best_step": source_best_step
                        }
            js_logger.log_dict(log_dict, step)
            tb_logger.log_dict(log_dict, step)
            
        "save checkpoint model"
        if config["interval"]["checkpoint"] != -1 and step % config["interval"]["checkpoint"] == 0:
            torch.save(G_S2T.state_dict(), os.path.join(checkpoint_path, f"G_S2T_{step}.pth"))
            torch.save(G_T2S.state_dict(), os.path.join(checkpoint_path, f"G_T2S_{step}.pth"))
            torch.save(D_S.state_dict(), os.path.join(checkpoint_path, f"D_S_{step}.pth"))
            torch.save(D_T.state_dict(), os.path.join(checkpoint_path, f"D_T_{step}.pth"))
            torch.save(CS_S.state_dict(), os.path.join(checkpoint_path, f"RN_S_{step}.pth"))
            torch.save(CS_T.state_dict(), os.path.join(checkpoint_path, f"RN_T_{step}.pth"))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Domain transfer adversarial network")
    # Task selection
    parser.add_argument("--evaluate", action="store_false", help="Training or testing")
    # Equipment configuration
    parser.add_argument("--cuda", type=str, nargs="?", default="1", help="Device id to run")
    parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
    # Dataset selection
    parser.add_argument("--dataset", type=str, default="Oracle241", choices=["Oracle241", "Digit", "Oracle50k_OBI125"], help="Name of the dataset")
    parser.add_argument("--source", type=str, default="h", help="The source domain")
    parser.add_argument("--target", type=str, default="s", help="The target domain")
    # Model configuration
    parser.add_argument("--model", type=str, default="4090_1_B64_Fcomp1", help="Model name")
    parser.add_argument("--weight", type=str, default="", help="Load checkpoints")
    parser.add_argument("--save", type=str, default=os.path.join(ROOT, "Results"), help="Save path")
    parser.add_argument("--classifier", type=str, default="ResNet18", choices=["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152", "LeNet", "DTN"], help="Domain adaption classifier")
    parser.add_argument("--n_residual_blocks", type=int, default=6, help="number of residual blocks in generator")
    parser.add_argument("--n_discrim_blocks", type=int, default=4, help="number of residual blocks in generator")
    # Training independent parameters
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--preloading", action='store_true', help="Data preloading")
    parser.add_argument("--log_interval", type=int, default=125, help="interval of two continuous log phase")
    parser.add_argument("--img_interval", type=int, default=250, help="interval of two continuous image show phase")
    parser.add_argument("--test_interval", type=int, default=500, help="interval of two continuous test phase")
    parser.add_argument("--checkpoint_interval", type=int, default=20000, help="interval of two continuous saving checkpoint phase")
    # Iteration rules
    parser.add_argument("--start_step", type=int, default=0, help="Step to start training from")
    parser.add_argument("--num_steps", type=int, default=100004, help="Number of steps of training")
    parser.add_argument("--batch_size", type=int, default=8, help="Size of the batches")
    # Learning rate rule
    parser.add_argument("--DT_lr", type=float, default=0.0002, help="Learning rate of domain transfer")
    parser.add_argument("--ML_lr", type=float, default=0.0002, help="Learning rate of mutual learning")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--lrf", type=float, default=0.01, help="lr final reduction rate ")
    parser.add_argument("--decay_step", type=int, default=0, help="step from which to start lr decay")
    # Training related hyperparameters
    parser.add_argument("--image_size", type=int, default=224, help="Image size")
    parser.add_argument("--preheat", type=int, default=-1, help="step from which to start mutual learning")
    parser.add_argument("--lambda_GAN", type=float, default=2.0, help="GAN loss weight")
    parser.add_argument("--lambda_cyc", type=float, default=2.0, help="cycle loss weight")
    parser.add_argument("--lambda_id", type=float, default=1.0, help="identity loss weight")
    parser.add_argument("--lambda_pseudo", type=float, default=1.0, help="pseudo loss weight")
    parser.add_argument("--lambda_con", type=float, default=2.0, help="comparative loss weight")
    parser.add_argument("--lambda_cls", type=float, default=2.0, help="classify loss weight")
    parser.add_argument("--temperature", type=float, default=1.0, help="comparative loss temperature")
    parser.add_argument("--class_sim", action='store_true', help="sample/class similarity preloading")
    parser.add_argument("--confidence_threshold", type=float, default=0.9, help="confidence threshold for fixmatch")
    
    args = parser.parse_args()

    # config
    config = {}
    # device
    config["device"] = {"cuda": args.cuda,
                        "cpus": args.n_cpu
                        }
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    # Random Seed
    config["seed"] = args.seed
    utils.seed_everything(config["seed"])
    # model name
    config["name"] = args.model
    # iteration
    config["iteration"] = {"start_step": args.start_step,
                           "num_steps": args.num_steps,
                           "batch_size": args.batch_size,
                           "preheat": args.preheat
                           }
    # model params
    config["model"] = {"classifier": args.classifier,
                       "residual_blocks": args.n_residual_blocks,
                       "discrim_blocks": args.n_discrim_blocks,
                       "kernel_size": 7
                       }
    # hyperparameters           
    config["parm"] = {"lambda": {"cyc": args.lambda_cyc,
                                 "id": args.lambda_id,
                                 "pseudo": args.lambda_pseudo,
                                 "con": args.lambda_con,
                                 "cls": args.lambda_cls,
                                 "GAN": args.lambda_GAN,
                                 },
                      "temperature": args.temperature,
                      "confidence_threshold": args.confidence_threshold,
                      "class_mode": args.class_sim
                      }
    # learning rate
    config["optim"] = {"type": optim.Adam,
                       "DT": {"lr": args.DT_lr,
                              "betas": (args.b1, args.b2)
                              },
                       "ML": {"lr": args.ML_lr,
                              "betas": (args.b1, args.b2)
                              }
                       }
    # lr scheduler
    config["scheduler"] = {"lrf": args.lrf,
                           "decay": args.decay_step
                           }

    # dirs
    config["dir"] = {"save":args.save,
                     "weight": args.weight
                     }
    # log
    config["interval"] = {"img": args.img_interval,
                          "log": args.log_interval,
                          "test": args.test_interval,
                          "checkpoint": args.checkpoint_interval}
    # date
    config["date"] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # dataset
    config["dataset"] = {"name": args.dataset,
                         "source": args.source,
                         "target": args.target,
                         "image_shape": (3, args.image_size, args.image_size)
                         }
    # image shape
    if args.dataset == "Digit":
        if args.source != "s":
            config["dataset"]["image_shape"] = (1, args.image_size, args.image_size)
        else:
            config["model"]["kernel_size"] = 5
    # num classes and dataset setup
    if args.dataset == "Oracle241":
        config["dataset"]["class_num"] = 241
        config["dataset"]["data"] = {"set": Oracle241,
                                     "train": {"train": True,
                                               "preloading": args.preloading and args.evaluate
                                               },
                                     "test": {"train": False,
                                              "preloading": args.preloading
                                               }
                                     }
    elif args.dataset == "Oracle50k_OBI125":
        config["dataset"]["class_num"] = 111
        config["dataset"]["data"] = {"set": Oracle50K_OBI125,
                                     "train": {"train": True,
                                               "preloading": args.preloading and args.evaluate
                                               },
                                     "test": {"train": False,
                                              "preloading": args.preloading
                                              }
                                     }
    elif args.dataset == "Digit":
        config["dataset"]["class_num"] = 10
        config["dataset"]["data"] = {"set": Digit10,
                                     "train": {"train": True},
                                     "test": {"train": False}
                                     }

    loader = {
        "train_source": utils.InfiniteDataLoader(
            config["dataset"]["data"]["set"](domain=config["dataset"]["source"],
                                             transform=get_dataset_transform(config["dataset"]["name"],
                                                                             source=True,
                                                                             img_size=args.image_size,
                                                                             channel=config["dataset"]["image_shape"][0],
                                                                             train=True),
                                             **config["dataset"]["data"]["train"]
                                             ),
            batch_size=config["iteration"]["batch_size"],
            shuffle=True,
            num_workers=args.n_cpu,
            drop_last=True
        ),
        "train_target": utils.InfiniteDataLoader(
            config["dataset"]["data"]["set"](domain=config["dataset"]["target"],
                                             transform=get_dataset_transform(config["dataset"]["name"],
                                                                             source=False,
                                                                             img_size=args.image_size,
                                                                             channel=config["dataset"]["image_shape"][0],
                                                                             train=True),
                                             **config["dataset"]["data"]["train"]
                                             ),
            batch_size=config["iteration"]["batch_size"],
            shuffle=True,
            num_workers=args.n_cpu,
            drop_last=True
        ),
        # Used by the generator
        "test_source": DataLoader(
            config["dataset"]["data"]["set"](domain=config["dataset"]["source"],
                                             transform=get_dataset_transform(config["dataset"]["name"],
                                                                             source=True,
                                                                             img_size=args.image_size,
                                                                             channel=config["dataset"]["image_shape"][0],
                                                                             train=False),
                                             **config["dataset"]["data"]["test"]
                                             ),
            batch_size=config["iteration"]["batch_size"],
            shuffle=True,
            num_workers=args.n_cpu,
            drop_last=True
        ),
        "test_target": {
            # Used by the generator
            "img": DataLoader(
                config["dataset"]["data"]["set"](domain=config["dataset"]["target"],
                                                 transform=get_dataset_transform(config["dataset"]["name"],
                                                                                 source=False,
                                                                                 img_size=args.image_size,
                                                                                 channel=config["dataset"]["image_shape"][0],
                                                                                 train=False),
                                                 **config["dataset"]["data"]["test"]
                                             ),
                batch_size=config["iteration"]["batch_size"],
                shuffle=True,
                num_workers=args.n_cpu,
                drop_last=True
            ),
            # Used by testing acc
            "acc": DataLoader(
                config["dataset"]["data"]["set"](domain=config["dataset"]["target"],
                                                 transform=get_dataset_transform(config["dataset"]["name"],
                                                                                 source=False,
                                                                                 img_size=args.image_size,
                                                                                 channel=config["dataset"]["image_shape"][0],
                                                                                 train=False),
                                                 **config["dataset"]["data"]["test"]
                                             ),
                batch_size=config["iteration"]["batch_size"],
                shuffle=False,
                num_workers=args.n_cpu,
                drop_last=False
            )
        }
    }
    
    if args.evaluate:
        train(config, loader)
    else:
        evaluate(config, loader)
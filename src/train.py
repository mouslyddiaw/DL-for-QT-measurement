"""Train the model"""

import argparse
import logging
import os
import json
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import utils
from model.net import AttnCNNv2, KanResWide, UNET_1D, Loss_MSE_CE, Loss_MSE_CE_Dice, choose_model  
from model.data_loader import ECGDatasetStratified
from evaluate import evaluate, metrics
import math
from sklearn.model_selection import KFold
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/cnn',
                    help="Directory containing params.json")
parser.add_argument('--experiment_tag', default='',
                    help="Experiment name")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'

def cyclical_lr(stepsize, min_lr=1.5e-4, max_lr=1.5e-3):

    # Scaler: we can adapt this if we do not want the triangular CLR
    scaler = lambda x: 1.

    # Lambda function to calculate the LR
    lr_lambda = lambda it: min_lr + (max_lr - min_lr) * relative(it, stepsize)

    # Additional function to see where on the cycle we are
    def relative(it, stepsize):
        cycle = math.floor(1 + it / (2 * stepsize))
        x = abs(it / stepsize - 2 * cycle + 1)
        return max(0, (1 - x)) * scaler(cycle)

    return lr_lambda

def train(model, optimizer, scheduler, loss_fn, dataloader, params, metrics, writer, nb_leads):
    """Train the model on `num_steps` batches
    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # Set current loss value
    loss_avg = utils.RunningAverage()
    model.train()

    # Use tqdm for progress bar
    with tqdm(total=nb_leads*len(dataloader)) as pbar:
        # Iterate over the DataLoader for training data
        for index, samples_batch in enumerate(dataloader):
            leads_batch = samples_batch['templates']
            intervals_batch = samples_batch['interval']  
            classes_batch = samples_batch['class'] 
            for _, train_batch in enumerate(leads_batch): 

                # move to GPU if available
                if params.cuda:
                    train_batch = train_batch.cuda(non_blocking=True)
                    intervals_batch = intervals_batch.cuda(non_blocking=True)
                    classes_batch = classes_batch.cuda(non_blocking=True)

                # convert to torch Variables
                train_batch, intervals_batch, classes_batch = Variable(train_batch), Variable(intervals_batch), Variable(classes_batch)

                # compute model output and loss
                output_batch = model(train_batch)  
                # Compute loss  
                if "AttnCNNv2" in model.__class__.__name__: 
                    loss = loss_fn(output_batch[0], output_batch[1], classes_batch, intervals_batch)  
                if "KanResWide" in model.__class__.__name__:
                    loss = loss_fn(output_batch[0], output_batch[1], classes_batch, intervals_batch) 
                if "UNET_1D" in model.__class__.__name__:
                    masks_batch = samples_batch['mask'] 
                    if params.cuda: 
                        masks_batch = masks_batch.cuda(non_blocking=True)
                    masks_batch = Variable(masks_batch)  
                    loss = loss_fn(output_batch[0], output_batch[1], classes_batch, intervals_batch, output_batch[2], masks_batch)  

                #Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                
                loss.backward()

                scheduler.step()
                
                optimizer.step()

                loss_avg.update(loss.item()) 
                pbar.set_postfix(loss='{:05.3f}'.format(loss_avg()))
                pbar.update()
                #break
        logging.info("Loss parameters: {}".format(list(loss_fn.parameters())))
        # compute the average loss 
        train_loss = loss_avg()
        #train_mae = evaluate_metric(model, dataloader, metrics, params)['absolute error']
    return train_loss #, train_mae

def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, scheduler, loss_fn, params, metrics, model_dir, 
                       writer, fold, nb_leads, restore_file=None):
    """Train the model and evaluate every epoch.
    Args:
        model: (torch.nn.Module) the neural network
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        val_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches validation data
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    """
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(
            args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    best_mae = 10000.0

    for epoch in range(params.num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        logging.info("Training...")
        train_loss = train(model, optimizer, scheduler, loss_fn, train_dataloader, params, metrics, writer, nb_leads)
         
        # Evaluate for one epoch on validation set
        logging.info("Validating...")
          
        metrics_mean = evaluate(model, loss_fn, val_dataloader, metrics, params)
        val_loss, val_mae = metrics_mean['loss'], metrics_mean['absolute error']
         
        # Write loss and metric on tensorboard
        writer.add_scalar('Fold'+str(fold)+'/loss-train', train_loss, epoch +1) 
        writer.add_scalar('Fold'+str(fold)+'/loss-val', val_loss, epoch + 1) 
        #writer.add_scalar('Fold'+str(fold)+'/mae-train', train_mae, epoch +1) 
        writer.add_scalar('Fold'+str(fold)+'/mae-val', val_mae, epoch + 1) 

        is_best = val_mae <= best_mae
        
        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                                is_best=is_best,
                                checkpoint=model_dir,
                                fold=fold)

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best mae")
            best_mae = val_mae



if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path) 
    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    model_name = args.model_dir.split('/')[-1]
    logging.info("Starting... {}, {}".format(model_name, args.experiment_tag))
    logging.info("Open tensorboard writer")
    writer = SummaryWriter(args.model_dir + '/runs/' + model_name +'_' + args.experiment_tag) 
    
    templates = ECGDatasetStratified()
    patient_splits = json.load(open("data/private-database/patient_splits_global.json"))
    files_per_patient = json.load(open("data/private-database/files_per_patient.json"))
    idx_fname = templates.idx_fname
    fname_idx = {item: key for key, item in idx_fname.items()}  
    nb_leads, nfold = 12, 5

    for fold, dic in patient_splits.items():   
        fold = int(fold)
        logging.info("Starting training for Fold {}/{}".format(fold, nfold)) 

        # Sample elements randomly from a given list of ids, no replacement. 
        train_recs = utils.flatten([files_per_patient[pid] for pid in dic['train']])
        val_recs = utils.flatten([files_per_patient[pid] for pid in dic['val']]) 
        train_ids = np.array([fname_idx[rec] for rec in train_recs])
        val_ids = np.array([fname_idx[rec] for rec in val_recs])
 
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
        
        logging.info("Loading the datasets...")
        # Define data loaders for training and testing data in this fold
        train_dl = torch.utils.data.DataLoader(
                        templates,
                        batch_size = params.batch_size, 
                        sampler = train_subsampler,
                        drop_last = True)
        
        val_dl = torch.utils.data.DataLoader(
                        templates,
                        batch_size = params.batch_size, sampler=val_subsampler)
        logging.info("- done.") 

        # Define the model, loss function, optimizer and scheduler 
        model, loss_fn = choose_model(args.model_dir, params) 

        optimizer = optim.Adam(list(model.parameters()) + list(loss_fn.parameters()), lr=params.learning_rate) 
        #optimizer = optim.Adam(list(model.parameters()) , lr=params.learning_rate) 
        
        step_size = 4*nb_leads*len(train_dl)
        end_lr, factor = 1.5e-3, 6
        clr = cyclical_lr(step_size, min_lr=end_lr/factor, max_lr=end_lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [clr])

        # Train the model
        logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
        
        train_and_evaluate(model, train_dl, val_dl, optimizer, scheduler, loss_fn, params, metrics,
                        args.model_dir, writer, fold, nb_leads, args.restore_file)
        logging.info("- done.")
        
        logging.info("Starting testing for Fold {}/{}".format(fold, nfold)) 
        
        # Reload weights from the saved file  
        utils.load_checkpoint(os.path.join(
            args.model_dir, 'best-{}.pth.tar'.format(fold)), model)

        # Evaluate
        val_metrics = evaluate(model, loss_fn, val_dl, metrics, params)
        save_path = os.path.join(
            args.model_dir, "metrics_val_{}_fold{}.json".format(args.restore_file, fold))
        utils.save_dict_to_json(val_metrics, save_path)

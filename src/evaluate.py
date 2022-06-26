
"""Evaluates the model"""

import argparse
import logging
import os
from tqdm import tqdm
import numpy as np
import torch
from torch.autograd import Variable
import utils 
import model.data_loader as data_loader
from model.net import AttnCNNv2, KanResWide, UNET_1D, choose_model, metrics 
from model.data_loader import ECGDataset
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/cnn',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")


def evaluate(model, loss_fn, dataloader, metrics, params): 
    model.eval()
    summ = []
    with tqdm(total=len(dataloader)) as pbar, torch.no_grad():
        for i, samples_batch in enumerate(dataloader):
            data_batch = samples_batch['templates'][0] 
            intervals_batch = samples_batch['interval']  
            classes_batch = samples_batch['class'] 

            # move to GPU if available
            if params.cuda:
                data_batch, intervals_batch, classes_batch = data_batch.cuda(non_blocking=True), intervals_batch.cuda(non_blocking=True), \
                                                              classes_batch.cuda(non_blocking=True)
            # fetch the next evaluation batch
            data_batch, intervals_batch, classes_batch = Variable(data_batch), Variable(intervals_batch), Variable(classes_batch) 

            # compute model output
            output_batch = model(data_batch)

            # Compute metrics 
            if "AttnCNNv2" in model.__class__.__name__:
                outputs_qt = output_batch[0]
                loss = loss_fn(output_batch[0], output_batch[1], classes_batch, intervals_batch)  
                summary_batch = {metric: metrics[metric](outputs_qt, intervals_batch)
                            for metric in metrics if "dice" not in metric}
            if "KanResWide" in model.__class__.__name__:
                outputs_qt = output_batch[0]
                loss = loss_fn(output_batch[0], output_batch[1], classes_batch, intervals_batch) 
                summary_batch = {metric: metrics[metric](outputs_qt, intervals_batch)
                            for metric in metrics if "dice" not in metric}
            if "UNET_1D" in model.__class__.__name__:
                masks_batch = samples_batch['mask'] 
                if params.cuda:
                    masks_batch = masks_batch.cuda(non_blocking=True)   
                outputs_qt = output_batch[0]
                outputs_mask = output_batch[2]
                loss = loss_fn(output_batch[0], output_batch[1], classes_batch, intervals_batch, output_batch[2], masks_batch) 
                summary_batch = {metric: metrics[metric](outputs_qt, intervals_batch)
                            for metric in metrics if "dice" not in metric}
                summary_batch['dice coeff'] = metrics['dice coeff'](masks_batch, outputs_mask)
            
            
            summary_batch['loss'] = [loss.item()]
            summ.append(summary_batch)
            pbar.update()
    
    # compute mean of all metrics in summary  
    metrics_concat = {metric: [] for metric in summ[0].keys()}
    for summary_batch in summ:
        for metric, val in summary_batch.items():
            metrics_concat[metric].extend(val)  
    metrics_mean = {metric: np.mean(concat) for metric, concat in metrics_concat.items()}
    
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    return metrics_mean


if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()     # use GPU is available

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    # Get the logger
    utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    model_name = args.model_dir.split('/')[-1]
    logging.info("Start testing {}".format(model_name))

    templates = ECGDataset(shift=False)
    nfold = 10
    kfold = KFold(n_splits=nfold, shuffle=True, random_state=1)

    for fold, (_, test_ids) in enumerate(kfold.split(templates)):  
        logging.info("Starting testing for Fold {}/{}".format(fold+1, nfold)) 

        # Sample elements randomly from a given list of ids, no replacement.
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        
        logging.info("Loading the datasets...")
        
        test_dl = torch.utils.data.DataLoader(
                        templates,
                        batch_size = params.batch_size, sampler=test_subsampler)
        logging.info("- done.")
  
        # Define the model
        model, loss_fn = choose_model(args.model_dir, params) 

        # Reload weights from the saved file 
        utils.load_checkpoint(os.path.join(
            args.model_dir, args.restore_file + '-{}.pth.tar'.format(fold+1)), model)

        # Evaluate
        test_metrics = evaluate(model, loss_fn, test_dl, metrics, params)
        save_path = os.path.join(
            args.model_dir, "metrics_test_{}_fold{}.json".format(args.restore_file, fold+1))
        utils.save_dict_to_json(test_metrics, save_path)

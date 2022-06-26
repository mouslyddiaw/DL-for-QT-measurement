import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import csv
from torch.utils.data import Dataset
import os
import utils
import model.data_loader as data_loader
from model.data_loader import ECGDataset, normalize
from model.net import AttnCNNv2, KanResWide, UNET_1D, compute_qt, classify_qt, choose_model, DiceCoeff
from scipy.signal import resample
from tqdm import tqdm
import wfdb
import neurokit2 as nk
from wavedel.ecg_delineation import my_delineator
from qrs_detector import get_final_rpeaks, create_template

import warnings
warnings.filterwarnings("ignore")

### Load models

def load_model(model_dir, fold=0, add_loss_fn = False):
    json_path = os.path.join(model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)
    params.cuda = True

    model, loss_fn = choose_model(model_dir, params)
    utils.load_checkpoint(os.path.join(model_dir, 'best' + '-{}.pth.tar'.format(fold+1)), model) 
    model.eval() 
    if add_loss_fn:
        return model, loss_fn, params
    else:
        return model

nfold = 5
models_cnn = [load_model('experiments/cnn_no_ptb', fold=fold)  for fold in range(nfold)] 
models_resnet = [load_model('experiments/resnet_no_ptb', fold=fold)  for fold in range(nfold)] 
models_unet = [load_model('experiments/unet_no_ptb', fold=fold)  for fold in range(nfold)] 

### Evaluate cross-validation

def evaluate_cv(model_dir, show = True) :

    templates = ECGDataset(shift=False)
    nfold = 5
    kfold = KFold(n_splits=nfold, shuffle=True, random_state=1) 

    global_refs, global_preds, global_dices = [], [], []   
    print('Evaluating 12 leads ')
    for fold, (_, test_ids) in enumerate(kfold.split(list(range(1, 8855)))):
        print('')
        print(f'Fold {fold+1}') 

         # Define the model
        model, loss_fn, params = load_model(model_dir, fold = fold, add_loss_fn = True)

        # Sample elements randomly from a given list of ids, no replacement.
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        test_dl = torch.utils.data.DataLoader(
                        templates,
                        batch_size = params.batch_size, sampler=test_subsampler)

        # Evaluate 
        with tqdm(total=12*len(test_dl)) as pbar, torch.no_grad(): 
            preds, refs, dices = evaluate_fold(model, loss_fn, test_dl, params, pbar)
            global_refs.extend(refs)
            global_preds.extend(preds)
            global_dices.extend(dices)
     
    diffs, mean_diff, upper, lower = stat_difference(global_refs, global_preds)

    print('')
    print(f'Final MAE {round(np.mean(np.abs(diffs)), 4)} ± {round(np.std(np.abs(diffs)), 4)}')
    if global_dices:
        print('')
        print(f'Final dice coeff {round(np.mean(np.abs(global_dices)), 4)} ± {round(np.std(np.abs(global_dices)), 4)}')

    if show:
        plot_results(global_refs, global_preds, diffs, mean_diff, upper, lower, title= model_dir.split('/')[-1])
    return global_refs, global_preds, global_dices

def evaluate_fold(model, loss_fn, dataloader, params, pbar, lead  = 0):
    model.eval()
    preds, refs, dices = [], [], [] 

    for i, samples_batch in enumerate(dataloader):
        for lead in range(12):
            data_batch = samples_batch['templates'][lead]
            intervals_batch = samples_batch['interval']
            classes_batch = samples_batch['class']

            # move to GPU if available
            if params.cuda:
                data_batch, intervals_batch, classes_batch = data_batch.cuda(non_blocking=True), intervals_batch.cuda(non_blocking=True), \
                                                                classes_batch.cuda(non_blocking=True)
        
            # compute model output
            output_batch = model(data_batch)

            # Compute metrics
            if "AttnCNNv2" in model.__class__.__name__:
                outputs_qt = output_batch[0]
                labels= intervals_batch
                preds_qt =  [compute_qt(classify_qt(probas)).item() for probas in outputs_qt]
            if "KanResWide" in model.__class__.__name__:
                outputs_qt = output_batch[0]
                labels= intervals_batch
                preds_qt =  [compute_qt(classify_qt(probas)).item() for probas in outputs_qt]
            if "UNET_1D" in model.__class__.__name__:
                masks_batch = samples_batch['mask'].cuda(non_blocking=True)
                outputs_qt = output_batch[0]
                labels= intervals_batch
                preds_qt =  [compute_qt(classify_qt(probas)).item() for probas in outputs_qt]
                outputs_mask = output_batch[2] 
                dice = DiceCoeff().forward(outputs_mask, masks_batch).item() 
                dices.append(dice)
            labels = [val.item() for val in labels] 
            preds.extend(preds_qt)
            refs.extend(labels)
            diffs, _, _, _ = stat_difference(refs, preds)
            pbar.update()
            if "UNET_1D" in model.__class__.__name__:
                pbar.set_postfix(MAE = f'{round(np.mean(np.abs(diffs)), 4)} ± {round(np.std(np.abs(diffs)), 4)}',
                                 Dice = f'{round(np.mean(np.abs(dices)), 4)} ± {round(np.std(np.abs(dices)), 4)}')
            else:
                pbar.set_postfix(MAE = f'{round(np.mean(np.abs(diffs)), 4)} ± {round(np.std(np.abs(diffs)), 4)}')
    return refs, preds, dices

def evaluate_delineator_private(show = True) :
    data =  ECGDataset(shift=False) 
    preds, refs = [], []   
    with tqdm(total=8855) as pbar:
        for i in range(1, 8855):
            leads = data[i]['templates']
            ref = data[i]['interval']
            #mask_ref =  data[i]['mask']
            for lead in leads:
                try :
                    pred , mask = get_qt_mask_wvlt(lead,  sampling_rate=500)
                except TypeError: #returns None
                    continue
                else:
                    preds.append(pred) 
                    refs.append(ref)  
            diffs, _, _, _ = stat_difference(refs, preds)
            pbar.set_postfix(MAE = f'{round(np.mean(np.abs(diffs)), 4)} ± {round(np.std(np.abs(diffs)), 4)}')
            pbar.update()

    diffs, mean_diff, upper, lower = stat_difference(refs, preds)
    if show:
        plot_results(refs, preds, diffs, mean_diff, upper, lower, title= 'Wavelet')

### Evaluate on external dataset 

def evaluate_model(model= 'cnn', db_name='qt' , show = True) : 
    if 'lu' in db_name:
        data = LUDataset() 
    else:
        data = QTDataset()  
    ids = [col for col in data.db.columns[1:] if col not in data.ignored_ids]

    refs, preds  = [], []
    with tqdm(total=len(ids)) as pbar:
        for col in ids:  
            mean_pred, mean_ref = get_mean_qt(data, col, model) 
            refs.append(mean_ref)
            preds.append(mean_pred)
            diffs, _, _, _ = stat_difference(refs, preds)
            pbar.set_postfix(MAE = f'{round(np.mean(np.abs(diffs)), 4)} ± {round(np.std(np.abs(diffs)), 4)}')
            pbar.update() 

    diffs, mean_diff, upper, lower = stat_difference(refs, preds)
    if show:
        plot_results(refs, preds, diffs, mean_diff, upper, lower, title = model) 
 

def get_mean_qt(data, col, model= 'cnn'):
    assert(model in ['cnn', 'resnet', 'unet', 'wvlt'] )
    results = data[col] 
    qts_ref = [summary['qt_ref'] for summary in results]  
    qts_pred = [summary['qt_pred_' + str(model)] for summary in results]     
    return np.mean(qts_pred), np.mean(qts_ref) 

def evaluate_model_bis(db_name='ptb', model= 'cnn', show = True) : 
    if 'ptb' in db_name: 
        data = PTBDataset()   
    if 'TQT' in db_name:
        data = TQTDataset()  

    refs, preds  = [], []
    with tqdm(total=len(data)) as pbar:
        for record in data.records:
            for lead in data.leads:
                summary = data[record, lead] 
                if summary:
                    refs.append(summary['qt_ref'])
                    preds.append(summary['qt_pred_' + str(model)])
                    diffs, _, _, _ = stat_difference(refs, preds)
                    pbar.set_postfix(MAE = f'{round(np.mean(np.abs(diffs)), 4)} ± {round(np.std(np.abs(diffs)), 4)}')
                    pbar.update() 

    diffs, mean_diff, upper, lower = stat_difference(refs, preds)
    if show:
        plot_results(refs, preds, diffs, mean_diff, upper, lower, title = model) 

### Datasets

class QTDataset(Dataset):
    def __init__(self):
        self.db = pd.read_csv('data/lu-qtdb/qtdb.csv')
        self.qrs_on_manual = csv_to_dict('data/lu-qtdb/QRSon_q1c.csv')
        self.qrs_peak_manual = csv_to_dict('data/lu-qtdb/QRSpeak_q1c.csv')
        self.t_off_manual = csv_to_dict('data/lu-qtdb/Toff_q1c.csv')
        self.ignored_ids = ['sel35_0','sel35_1', 'sel37_0', 'sel37_1'] # no T wave annotations
        self.sampling_rate  = 250 

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.db.columns)

    def __getitem__(self, col): 
        'Generates one sample of data'
        # Select sample
        signal = self.db[col]
        qons = self.qrs_on_manual[col]
        toffs = self.t_off_manual[col]
        rpeaks = self.qrs_peak_manual[col] 

        results = []  
        for peak in rpeaks:
            try:
                beat = data_loader.normalize(np.array(signal)[peak-200:peak+600-200])
                resampled_beat =  resample(beat, 2*len(beat))[200:800]
                qon = get_qrs_on(peak, qons, sampling_rate=self.sampling_rate)
                toff = get_t_off(peak, toffs, sampling_rate=self.sampling_rate)
                qt =  1000*(toff - qon)/self.sampling_rate
            except TypeError:
                pass
            else:
                try :
                    pred_wvlt , _= get_qt_mask_wvlt(resampled_beat,  sampling_rate = 2*self.sampling_rate)
                   # pred_wvlt , _= get_qt_mask_wvlt(beat,  sampling_rate = self.sampling_rate)
                except TypeError:  
                    continue
                else: 
                    pred_cnn = get_qt_beat_nn(resampled_beat, model_name = 'cnn') 
                    pred_resnet = get_qt_beat_nn(resampled_beat, model_name = 'resnet') 
                    pred_unet = get_qt_beat_nn(resampled_beat, model_name = 'unet')   
                    summary = { } 
                    summary['id'] = col
                    summary['position_rpeak'] = peak 
                    summary['qt_ref']  = qt
                    summary['qt_pred_wvlt']  = pred_wvlt
                    summary['qt_pred_cnn']  = pred_cnn
                    summary['qt_pred_resnet']  = pred_resnet
                    summary['qt_pred_unet']  = pred_unet
                    summary['qon_ref']  = 200 - (peak-qon)
                    summary['toff_ref']  = 200+ (toff-peak)
                    results.append(summary) 
        return results 

class LUDataset(Dataset):
    def __init__(self):
        self.db = pd.read_csv('data/lu-qtdb/ludb.csv')
        self.qrs_on_manual = csv_to_dict('data/lu-qtdb/QRSon_manual.csv')
        self.qrs_peak_manual = csv_to_dict('data/lu-qtdb/QRSpeak_manual.csv')
        self.t_off_manual = csv_to_dict('data/lu-qtdb/Toff_manual.csv')  
        self.sampling_rate  = 500
        self.ignored_ids = ['104_ii', '104_iii', '104_avr', '104_avf', '112_v1', '112_v2', '38_v1', '38_v2', '7_v2'] #empty annotations (qon or toff) 

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.db.columns)

    def __getitem__(self, col):
        'Generates one sample of data'
        # Select sample
        signal = self.db[col]
        qons = self.qrs_on_manual[col]
        toffs = self.t_off_manual[col]
        rpeaks = self.qrs_peak_manual[col]

        results = []  
        for peak in rpeaks:
            try:
                beat = data_loader.normalize(np.array(signal)[peak-200:peak+600-200])
                assert(len(beat)==600)
                qon = get_qrs_on(peak, qons, sampling_rate=self.sampling_rate)
                toff = get_t_off(peak, toffs, sampling_rate=self.sampling_rate)
                qt =  1000*(toff - qon)/self.sampling_rate
                assert(str(qt)!='nan')
            except (TypeError, AssertionError): 
                pass
            else:  
                try :
                    pred_wvlt , _= get_qt_mask_wvlt(beat,  sampling_rate=self.sampling_rate)
                except TypeError : #returns None 
                    continue
                else:   
                    pred_cnn = get_qt_beat_nn(beat, model_name = 'cnn') 
                    pred_resnet = get_qt_beat_nn(beat, model_name = 'resnet') 
                    pred_unet = get_qt_beat_nn(beat, model_name = 'unet')  
                    summary = { } 
                    summary['id'] = col
                    summary['position_rpeak'] = peak 
                    summary['qt_ref']  = qt
                    summary['qt_pred_wvlt']  = pred_wvlt
                    summary['qt_pred_cnn']  = pred_cnn
                    summary['qt_pred_resnet']  = pred_resnet
                    summary['qt_pred_unet']  = pred_unet
                    summary['qon_ref']  = 200 - (peak-qon)
                    summary['toff_ref']  = 200+ (toff-peak)
                    results.append(summary)  
        return results

class PTBDataset(Dataset):
    def __init__(self):
        self.records = [str(path) for path in  Path('data/ptb/').glob('**/*.dat')]
        self.reference = get_reference()
        self.leads = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'vx', 'vy', 'vz']
        self.sampling_rate  = 1000 

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.records)*len(self.leads)

    def __getitem__(self, args): 
        'Generates one sample of data'
        # Select sample 
        filename, lead  = args
        fname = filename.split('.')[0] 

        record = wfdb.rdrecord(fname) 
        signals, record_name  = get_signal(record)  
  
        fs  = record.fs 
        qt_ref = self.reference.loc[record_name]['qt']

        sig = signals[lead][:30*fs] 
        final_rpeaks, rpeaks, _, _ = get_final_rpeaks(sig, sampling_rate = self.sampling_rate)
        template = np.array(create_template(sig, rpeaks, final_rpeaks, sampling_rate = self.sampling_rate))
        resampled_beat = normalize(resample(template,  len(template)//2))  
        try:
            pred_wvlt = get_qt_mask_wvlt(resampled_beat,  sampling_rate = fs//2)[0]
            pred_cnn = get_qt_beat_nn(resampled_beat, model_name = 'cnn')  
            pred_resnet = get_qt_beat_nn(resampled_beat, model_name = 'resnet') 
            pred_unet = get_qt_beat_nn(resampled_beat, model_name = 'unet')
        except TypeError:
            return None   
        else:
            summary = { } 
            summary['id'] = record_name + '_' + lead
            summary['qt_ref']  = qt_ref
            summary['qt_pred_wvlt']  = pred_wvlt
            summary['qt_pred_cnn']  = pred_cnn
            summary['qt_pred_resnet']  = pred_resnet
            summary['qt_pred_unet']  = pred_unet
        return summary

def get_reference(path = 'data/ptb/reference-QT.txt'):
    reference = pd.read_csv(path, sep="\t",header=None) 
    reference = reference.rename(columns={0: 'record', 1: 'qt', 2: 'mad', 3: 'nb'}) 
    reference['id'] = [val.split('/')[-1] for val in reference['record']]
    reference = reference.set_index('id') 
    return reference


class TQTDataset(Dataset):
    def __init__(self):
        self.data = pd.read_csv('data/TQTstudy/SCR-002.Clinical.Data.csv') 
        self.records = list(self.data.EGREFID)
        self.leads = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        self.sampling_rate  = 1000 

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.records)*len(self.leads)

    def __getitem__(self, args): 
        'Generates one sample of data'
        # Select sample 
        fname, lead  = args
        patient_id = list(self.data[self.data.EGREFID==fname].RANDID)[0]
        record = wfdb.rdrecord('data/TQTstudy/raw/'+str(patient_id)+'/'+fname) 
        signals, record_name  = get_signal(record)  
        fs  = self.sampling_rate   

        qt_ref = list(self.data[self.data.EGREFID==fname].QT)[0]

        sig = signals[lead] 
        
        try:
            final_rpeaks, rpeaks, _, _ = get_final_rpeaks(sig, sampling_rate = self.sampling_rate)
            template = np.array(create_template(sig, rpeaks, final_rpeaks, sampling_rate = self.sampling_rate))
            resampled_beat = normalize(resample(template,  len(template)//2))  
            pred_wvlt = get_qt_mask_wvlt(resampled_beat,  sampling_rate = fs//2)[0]
            pred_cnn = get_qt_beat_nn(resampled_beat, model_name = 'cnn')  
            pred_resnet = get_qt_beat_nn(resampled_beat, model_name = 'resnet') 
            pred_unet = get_qt_beat_nn(resampled_beat, model_name = 'unet')
        except (ValueError, TypeError):
            return None   
        else:
            summary = { } 
            summary['id'] = record_name + '_' + lead
            summary['qt_ref']  = qt_ref
            summary['qt_pred_wvlt']  = pred_wvlt
            summary['qt_pred_cnn']  = pred_cnn
            summary['qt_pred_resnet']  = pred_resnet
            summary['qt_pred_unet']  = pred_unet
        return summary

def get_signal(record):
    if record:
        if record.p_signal is not None:
            signal = record.p_signal
        elif record.d_signal is not None:
            signal = record.d_signal
        else:
            raise ValueError('The record has no signal to plot')

        fs = record.fs
        sig_name = [str(s) for s in record.sig_name]
        sig_units = [str(s) for s in record.units]
        record_name =  record.record_name
        ylabel = ['/'.join(pair) for pair in zip(sig_name, sig_units)]
    else:
        signal = fs = ylabel = record_name = sig_units = None
    df = pd.DataFrame(signal, columns= sig_name)
    return df, record_name

class PrivateDataset(Dataset):
    def __init__(self):
        self.data =  ECGDataset(shift=False)
        self.leads = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
        self.sampling_rate  = 500
        self.nb_signals = 8855

    def __len__(self):
        'Denotes the total number of samples'
        return (self.nb_signals-1)*len(self.leads)

    def __getitem__(self, args): 
        'Generates one sample of data'
        # Select sample 
        fold, sample_idx, lead_idx = args
        signals = self.data[sample_idx]['templates']
        qt_ref = float(self.data[sample_idx]['interval'].detach().numpy())
  
        beat = signals[lead_idx].detach().numpy()   
        try:
            pred_wvlt = get_qt_mask_wvlt(beat,  sampling_rate = self.sampling_rate)[0]
            pred_cnn = compute_qt_nn(beat, models_cnn[fold])  
            pred_resnet = compute_qt_nn(beat, models_resnet[fold]) 
            pred_unet = compute_qt_nn(beat, models_unet[fold]) 
        except TypeError:
            return None   
        else:
            summary = { } 
            summary['id'] = f'id-{sample_idx}_{self.leads[lead_idx]}' 
            summary['qt_ref']  =  qt_ref
            summary['qt_pred_wvlt']  = pred_wvlt
            summary['qt_pred_cnn']  = pred_cnn
            summary['qt_pred_resnet']  = pred_resnet
            summary['qt_pred_unet']  = pred_unet
        return summary

### Utils

def get_qt_beat_nn(beat, model_name, nfold=5):
    assert(model_name in ['cnn', 'resnet', 'unet'] )
    if 'cnn' in model_name:
        preds = [compute_qt_nn(beat, model) for model in models_cnn]
    if 'resnet' in model_name:
        preds = [compute_qt_nn(beat, model) for model in models_resnet]
    if 'unet' in model_name:
        preds = [compute_qt_nn(beat, model) for model in models_unet]
    return np.mean(preds)

def compute_qt_nn(beat, model):
    logits_mdl = model(torch.tensor(beat).float().unsqueeze(0).cuda())[0].cpu().detach()
    pred = compute_qt(classify_qt(logits_mdl, dim=1)).item()
    return pred

def get_unet_mask(beat, model = models_unet[0]):
    mask = model(torch.tensor(beat).float().unsqueeze(0).cuda())[2].cpu().detach() 
    return mask

def plot_results(refs, preds, diffs, mean_diff, upper, lower, title= 'Wavelet'):
    fig, ax = plt.subplots(1, 2, figsize = (25, 12))
    ax = ax.flatten()
    fontsize=15
    ax[0].scatter(refs, diffs, s = fontsize)
    ax[0].axhline(y = mean_diff, color='k', linestyle='-', linewidth=3)
    ax[0].axhline(y = upper, color='r', linestyle='-.', linewidth=3)
    ax[0].axhline(y = lower, color='r', linestyle='-.', linewidth=3)
    ax[0].tick_params(labelsize=fontsize)
    ax[0].set_title(title, fontsize = fontsize)
    ax[0].set_xlabel('Manual (ms)', fontsize = fontsize)
    ax[0].set_ylabel('Manual - Automatic (ms)', fontsize = fontsize)
    ax[0].text(max(refs)-25, mean_diff + 7, 'mean', fontsize = fontsize, weight='bold')
    ax[0].text(max(refs)-25, mean_diff - 20, str(round(mean_diff, 2)), fontsize = fontsize, weight='bold')
    ax[0].text(max(refs)- 80, upper + 10, '+1.96s: '+ str(round(upper, 2)) ,fontsize = fontsize, weight='bold')
    ax[0].text(max(refs)- 70, lower - 20, '-1.96s: '+ str(round(lower, 2)) ,fontsize = fontsize, weight='bold')

    ax[1].scatter(refs, preds)
    ax[1].plot(refs, refs, color='gray', linestyle=':')
    ax[1].tick_params(labelsize=fontsize)
    ax[1].set_ylabel('Automatic (ms)', fontsize = fontsize)
    ax[1].set_xlabel('Manual (ms)', fontsize = fontsize)
    ax[1].set_title( title, fontsize = fontsize)
    plt.show()

def visualise_mask(beat, qon, toff, db_name):
    mask = model(torch.tensor(beat).float().unsqueeze(0).cuda())[2].cpu().detach()
    mask_ref = create_mask(qon, toff, db_name) 

    x = beat  
    plt.rcParams["figure.figsize"] = 8, 7
    fig, ax = plt.subplots(nrows=2, sharex=True) 
    fontsize = 20
    extent = [0, 600, min(x).item() ,max(x).item()]
    y =  mask_ref  
    img = ax[0].imshow(y[np.newaxis,:], cmap="Blues", aspect="auto", extent=extent)
    ax[0].set_yticks([])
    ax[0].set_xticks([])
    ax[0].set_xlim(extent[0], extent[1]) 
    ax[0].plot(x, color='thistle', linewidth=6)   
    ax[0].text(20, max(x)-1.25, 'Reference', fontsize = fontsize, weight='bold') 
    ax[0].text(20, max(x)-2.6, 'QT: ' + str(qts[index]) + ' ms', fontsize = fontsize) 
        
    y = mask.detach().numpy()[0][0]
    img = ax[1].imshow(y[np.newaxis,:], cmap="Blues", aspect="auto", extent=extent)
    ax[1].set_yticks([])
    ax[1].set_xticks([])
    ax[1].set_xlim(extent[0], extent[1]) 
    ax[1].plot(x, color='thistle', linewidth=6)   
    ax[1].text(20, max(x)-2.6, 'QT: ' + str(pred) + ' ms', fontsize = fontsize) 
    
    rect = plt.Rectangle(
                    # (lower-left corner), width, height
                    (0, 0), 1, 2*0.5, fill=False, color="k", lw=3, 
                    zorder=1000, transform=fig.transFigure, figure=fig
                )
    fig.patches.extend([rect])
    plt.tight_layout() 
    plt.show()   
    return mask, mask_ref

def create_mask(qon, toff, db_name):
    mask = [0 for i in range(600)]
    lower = qon
    upper = toff + 1
    mask[lower:upper] = [1 for i in range (lower, upper)]  
    if 'lu' in db_name:
        return mask
    else:
        resampled_mask = resample(mask, 2*len(mask))[200:800] 
        resampled_mask =[int(val>0.5) for val in resampled_mask]  
        return np.array(resampled_mask)

def get_qt_nk(ecg_signal, rpeaks, sampling_rate):
    _, my_markers = nk.ecg_delineate(ecg_signal , rpeaks, sampling_rate, method="dwt", show=False, show_type='all')
    qts = [val1 - val2 for val1, val2 in zip(my_markers['ECG_T_Offsets'], my_markers['ECG_R_Onsets'])
          if str(val1)!='nan' and str(val2)!='nan']
    qt_in_ms = 1000*np.mean(qts)/sampling_rate
    return qt_in_ms

def get_my_qt(ecg_signal, sampling_rate):  #qrs_off, qrs_on,
    my_markers = my_delineator(ecg_signal, sampling_rate, adjust_rpeak=True)
   # _, _, t_off  = detect_twave(ecg_signal, qrs_off, sampling_rate)
    qts = [val1 - val2 for val1, val2 in zip(my_markers['ECG_T_Offsets'], my_markers['ECG_R_Onsets'])
          if str(val1)!='nan' and str(val2)!='nan']
   # qts = [val1 - val2 for val1, val2 in zip(t_off, qrs_on)
    #      if str(val1)!='nan' and str(val2)!='nan']
    qt_in_ms = 1000*np.mean(qts)/sampling_rate
    return qt_in_ms

def get_qt_mask_wvlt(ecg_signal, sampling_rate):
    my_markers =  my_delineator(ecg_signal, sampling_rate)
    mask = [0 for i in range(600)]
    try:
        rpeaks = my_markers['ECG_R_Peaks']
        idx_rpeak = position_rpeak(rpeaks)

        qon = my_markers['ECG_R_Onsets'][idx_rpeak]
        toff = my_markers['ECG_T_Offsets'][idx_rpeak]
    except TypeError:
        return None
    else:
        qt_interval = 1000*(toff - qon)/sampling_rate
        lower = qon
        upper = toff  + 1
        mask[lower:upper] = [1 for i in range (lower, upper)]
        return qt_interval, np.array(mask)

def position_rpeak(rpeaks):
    idx_peak = None
    for index, rpeak in enumerate (rpeaks):
        if rpeak>100 and rpeak<300:
            idx_peak = index
    return idx_peak

def convert_to_int(func):
    def str_to_int(filename, dir_path=''):
        labels = func(filename, dir_path='')
        for signal_id in labels.keys():
            if labels[signal_id] == ['']:
                labels[signal_id] = []
            else:
                labels[signal_id] = list(map(int, labels[signal_id]))
        return labels
    return str_to_int

@convert_to_int
def csv_to_dict(filename, dir_path=''):
    labels= dict()
    with open(dir_path + filename, mode='r') as inp:
        reader = csv.reader(inp)
        labels = {rows[0]:rows[1:] for rows in reader}
    return labels


def get_qrs_on(peak, qons, sampling_rate):
    for prev, nxt in zip(qons[:len(qons)-1], qons[1:]):
        if peak in range(prev,nxt) and convert_in_msec(peak-prev, sampling_rate)<200:
            return  prev

def get_t_off(peak, toffs, sampling_rate):
    for prev, nxt in zip(toffs[:len(toffs)-1], toffs[1:]):
        if peak in range(prev,nxt) and convert_in_msec(nxt-peak, sampling_rate)<800:
            return  nxt

def stat_difference(actual, predicted):
    diffs = [val1 - val2 for val1, val2 in zip(actual, predicted)] 
    mean_diff =  np.mean(diffs)
    std_diff = np.std(diffs)
    upper = mean_diff + 1.96*std_diff
    lower = mean_diff - 1.96*std_diff
    return diffs, mean_diff, upper, lower

def convert_in_msec(duration, sampling_rate):
    return 1000*duration/sampling_rate 
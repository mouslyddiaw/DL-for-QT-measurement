import torch
import torch.nn.functional as F
import numpy as np
import argparse
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
from model.data_loader import ECGDatasetStratified, normalize
from model.net import AttnCNNv2, KanResWide, UNET_1D, compute_qt, classify_qt, choose_model, DiceCoeff
from scipy.signal import resample
from tqdm import tqdm
import wfdb 
from compare.qrs_detector import get_final_rpeaks, create_template
from compare.utils import *

import warnings
warnings.filterwarnings("ignore") 

parser = argparse.ArgumentParser()
parser.add_argument('--db_name', default=None, help="Database name") 
parser.add_argument('--loss_weight', default=False, help="Save plots loss weights") 
parser.add_argument('--result_type', default=1, 
                    help="1: me, mae, accqt, boxplot perf leads; 2: qt std, qtd; 3: illustration qt delineation variability superimposed beats, 4: attn maps, masks unet, softmax probas, 5: save mean qts TQT study, 6: Analyze tqt study") 

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
 
### Datasets

class QTDataset(Dataset):
    def __init__(self, viz=False): 
        self.db = pd.read_csv(os.path.join('data/lu-qtdb/','qtdb.csv'))
        self.qrs_on_manual = csv_to_dict('data/lu-qtdb/QRSon_q1c.csv')
        self.qrs_peak_manual = csv_to_dict('data/lu-qtdb/QRSpeak_q1c.csv')
        self.t_off_manual = csv_to_dict('data/lu-qtdb/Toff_q1c.csv')
        self.ignored_ids = ['sel35_0','sel35_1', 'sel37_0', 'sel37_1'] # no T wave annotations
        self.sampling_rate  = 250 
        self.viz = viz

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
                original_beat = np.array(signal)[peak-200:peak+600-200]
                beat = data_loader.normalize(original_beat)
                resampled_beat =  resample(beat, 2*len(beat))[200:800]
                qon = get_qrs_on(peak, qons, sampling_rate=self.sampling_rate)
                toff = get_t_off(peak, toffs, sampling_rate=self.sampling_rate)
                qt =  1000*(toff - qon)/self.sampling_rate
            except TypeError:
                pass
            else:
                if self.viz:
                    summary = { }
                    summary['beat'] =  resample(original_beat, 2*len(original_beat))[200:800]
                    summary['normalized_beat'] =  resampled_beat
                    summary['qon_ref']  = 200 - (peak-qon)
                    summary['toff_ref']  = 200+ (toff-peak)
                    results.append(summary)
                else:
                    try :
                        pred_wvlt , _= get_qt_mask_wvlt(resampled_beat,  sampling_rate = 2*self.sampling_rate) 
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
    def __init__(self, viz=False):
        self.db = pd.read_csv('data/lu-qtdb/ludb.csv')
        self.qrs_on_manual = csv_to_dict('data/lu-qtdb/QRSon_manual.csv')
        self.qrs_peak_manual = csv_to_dict('data/lu-qtdb/QRSpeak_manual.csv')
        self.t_off_manual = csv_to_dict('data/lu-qtdb/Toff_manual.csv')  
        self.sampling_rate  = 500
        self.ignored_ids = ['104_ii', '104_iii', '104_avr', '104_avf', '112_v1', '112_v2', '38_v1', '38_v2', '7_v2'] #empty annotations (qon or toff) 
        self.viz = viz

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
                original_beat = np.array(signal)[peak-200:peak+600-200]
                beat = data_loader.normalize(original_beat)
                assert(len(beat)==600)
                qon = get_qrs_on(peak, qons, sampling_rate=self.sampling_rate)
                toff = get_t_off(peak, toffs, sampling_rate=self.sampling_rate) 
                qt =  1000*(toff - qon)/self.sampling_rate
                assert(str(qt)!='nan')
            except (TypeError, AssertionError): 
                pass
            else: 
                if self.viz:
                    summary = { }
                    summary['beat'] =  original_beat
                    summary['normalized_beat'] =  beat
                    summary['qon_ref']  = 200 - (peak-qon)
                    summary['toff_ref']  = 200+ (toff-peak)
                    results.append(summary) 
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
    def __init__(self, viz=False):
        self.data = pd.read_csv('data/TQTstudy/SCR-002.Clinical.Data.csv') 
        self.records = list(self.data.EGREFID)
        self.leads = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        self.sampling_rate  = 1000 
        self.viz = viz

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
            if self.viz:
                summary = { }
                summary['beat'] =  resample(template,  len(template)//2)
                summary['normalized_beat'] =  resampled_beat
                summary['qt_ref'] =  qt_ref
            else:
                summary = { } 
                summary['id'] = record_name + '_' + lead
                summary['qt_ref']  = qt_ref
                summary['qt_pred_wvlt']  = pred_wvlt
                summary['qt_pred_cnn']  = pred_cnn
                summary['qt_pred_resnet']  = pred_resnet
                summary['qt_pred_unet']  = pred_unet
        return summary

class PrivateDataset(Dataset):
    def __init__(self):
        self.data =  ECGDatasetStratified()
        self.patient_splits = json.load(open("data/private-database/patient_splits_global.json"))
        self.files_per_patient = json.load(open("data/private-database/files_per_patient.json"))
        self.idx_fname = self.data.idx_fname
        self.fname_idx = {item: key for key, item in self.idx_fname.items()}  
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
        except (TypeError, IndexError):
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

### Save results in json file 
def evaluate_model(db_name) : 
    if 'lu' in db_name:
        data = LUDataset() 
    else:
        data = QTDataset()  
    ids = [col for col in data.db.columns[1:] if col not in data.ignored_ids]
    
    predictions = {} 
    with tqdm(total=len(ids)) as pbar:
        for col in ids:
            predictions[col] =  get_predictions(data, col)
            try:
                write_json({col: predictions[col]}, os.path.join('data/outputs/qt-results', f"results_{db_name}.json") )
            except FileNotFoundError:
                save_dict_to_json({"predictions": [] }, os.path.join('data/outputs/qt-results', f"results_{db_name}.json") ) 
                write_json({col: predictions[col]}, os.path.join('data/outputs/qt-results', f"results_{db_name}.json") )
            pbar.update()  
    return predictions

def get_predictions(data, col): 
    results = data[col]  
    return {'ref': [summary['qt_ref'] for summary in results]  ,
            'wvlt': [summary['qt_pred_wvlt' ] for summary in results] , 
            'cnn':  [summary['qt_pred_cnn'] for summary in results] ,  
            'resnet': [summary['qt_pred_resnet' ] for summary in results] , 
            'unet':  [summary['qt_pred_unet'] for summary in results]} 

def evaluate_model_ptb_tqt(db_name='ptb'): 
    if 'ptb' in db_name:
        data = PTBDataset()   
    else:
        data = TQTDataset()
    predictions = {} 
    with tqdm(total=len(data)) as pbar:
        for record in data.records:
            for lead in data.leads: 
                summary = data[record, lead] 
                if summary: 
                    col = summary['id'] 
                    predictions[col] = {key.split('_')[-1]: [item] for key, item in summary.items() if 'id' not in key}  
                    try:
                        write_json({col: predictions[col]}, os.path.join(f'data/outputs/qt-results/results_{db_name}', f"{col}.json") )
                    except FileNotFoundError:
                        save_dict_to_json({"predictions": [] }, os.path.join(f'data/outputs/qt-results/results_{db_name}', f"{col}.json") ) 
                        write_json({col: predictions[col]}, os.path.join(f'data/outputs/qt-results/results_{db_name}', f"{col}.json") )
                    pbar.update()  
    return predictions

def evaluate_model_private(db_name='private'): 
    data = PrivateDataset()     
    predictions = {}  
    with tqdm(total=len(data)) as pbar:
        for fold, dic in data.patient_splits.items():
            fold = int(fold)
            val_recs = utils.flatten([data.files_per_patient[pid] for pid in dic['val']])  
            val_ids = np.array([data.fname_idx[rec] for rec in val_recs])
            for sample_idx in val_ids:
                for lead_idx in range(len(data.leads)):
                    summary = data[fold, sample_idx, lead_idx]  
                    if summary:
                        col = summary['id'] 
                        predictions[col] = {key.split('_')[-1]: [item] for key, item in summary.items() if 'id' not in key}  
                        try:
                            write_json({col: predictions[col]}, os.path.join(f'data/outputs/qt-results/results_{db_name}', f"{col}.json") )
                        except FileNotFoundError:
                            save_dict_to_json({"predictions": [] }, os.path.join(f'data/outputs/qt-results/results_{db_name}', f"{col}.json") ) 
                            write_json({col: predictions[col]}, os.path.join(f'data/outputs/qt-results/results_{db_name}', f"{col}.json") )
                        pbar.update()  
    return predictions
 
if __name__ == "__main__": 
    args = parser.parse_args() 
    db_name = args.db_name
    loss_weight = eval(str(args.loss_weight))
    result_type = int(args.result_type)

    if loss_weight:
        print('Plotting loss weights...')
        for model in ['cnn', 'resnet', 'unet']:
            output = get_loss_weights(log_path=f'experiments/{model}_strat/train.log')
            plot_loss_weights(output, model=model, filepath = f'data/outputs/loss-weights/{model}.png')
        print('-done')

     # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(230)
    
    print('Loading models ...')
    nfold = 5
    models_cnn = [load_model('experiments/cnn_strat', fold=fold)  for fold in range(nfold)] 
    models_resnet = [load_model('experiments/resnet_strat', fold=fold)  for fold in range(nfold)] 
    models_unet = [load_model('experiments/unet_strat', fold=fold)  for fold in range(nfold)] 
    print('-done')

    if not db_name:
        if result_type ==1:
            db_names = ['private', 'qt', 'lu', 'ptb', 'TQT']
            models =  ['cnn', 'resnet', 'unet', 'wvlt']
            qt_accuracy_metrics = {db_name: {model: {'nb_qts': None, 'me': None, 'me_std': None, 'mae': None, 'mae_std': None, 'accQT': None} for model in models} for db_name in db_names} 
            all_qts = {model: {'refs': [], 'preds': []} for model in models} 
            for db_name in db_names: 
                print(f'Loading qts in json file(s) ({db_name})')
                if db_name in ['lu', 'qt']:
                    predictions = read_json(db_name = db_name)
                else:
                    pathlist = Path(f'data/outputs/qt-results/results_{db_name}').glob('**/*.json')
                    predictions = [read_json(json_path=json_path)[0] for json_path in pathlist]  
                print('-done') 

                
                if db_name in ['lu', 'ptb']:
                    print(f'Boxplot performance across leads for {db_name}') 
                    if 'lu' in db_name: 
                        leads = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1','v2','v3','v4','v5','v6']
                        capitalized_leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1','V2','V3','V4','V5','V6'] 
                    else:
                        leads = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1','v2','v3','v4','v5','v6','vx','vy','vz']
                        capitalized_leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1','V2','V3','V4','V5','V6','Vx','Vy','Vz']   
                    summary = concatenate_dict(predictions, db_name) 
                    visualize_boxplot_by_lead(summary, leads=leads, capitalized_leads=capitalized_leads, filepath=os.path.join(f'data/outputs/qt-results/perf-across-leads', f"perf-across-leads-{db_name}.png"), metric = 'mae', ylabel = 'Absolute error (ms)', legend=True) 

                print(f'ME, MAE, accQT for {db_name}')
                for model in models: 
                    refs, preds, diffs = ba_plot(predictions, model=model, db_name=db_name, print_metric = True) 
                    # Compute metrics
                    assert(len(refs)==len(preds))
                    qt_accuracy_metrics[db_name][model]['nb_qts'] = len(refs)
                    qt_accuracy_metrics[db_name][model]['me'] = round(np.mean(diffs), 1)
                    qt_accuracy_metrics[db_name][model]['me_std'] = round(np.std(diffs), 1)
                    qt_accuracy_metrics[db_name][model]['mae'] = round(np.mean(np.abs(diffs)), 1)
                    qt_accuracy_metrics[db_name][model]['mae_std'] = round(np.std(np.abs(diffs)), 1)
                    qt_accuracy_metrics[db_name][model]['accQT'] = get_percent_accQT(refs, preds)
                    # Add qts from external databases to global list
                    if 'private' not in db_name:
                        all_qts[model]['refs'].extend(refs)
                        all_qts[model]['preds'].extend(preds)
                save_dict_to_json(qt_accuracy_metrics, os.path.join(f'data/outputs/qt-results', 'qt_accuracy_metrics.json') )
            
            print(f'Global ME, MAE, accQT')
            qt_accuracy_global_metrics = {model: {'nb_qts': None, 'me': None, 'me_std': None, 'mae': None, 'mae_std': None, 'accQT': None} for model in models}
            for model in models:
                all_refs, all_preds = all_qts[model]['refs'], all_qts[model]['preds']   
                diffs, _, _, _ = stat_difference(all_refs, all_preds)
                assert(len(all_refs)==len(all_preds))
                qt_accuracy_global_metrics[model]['nb_qts'] = len(all_refs)
                qt_accuracy_global_metrics[model]['me'] = round(np.mean(diffs), 1)
                qt_accuracy_global_metrics[model]['me_std'] = round(np.std(diffs), 1)
                qt_accuracy_global_metrics[model]['mae'] = round(np.mean(np.abs(diffs)), 1)
                qt_accuracy_global_metrics[model]['mae_std'] = round(np.std(np.abs(diffs)), 1)
                qt_accuracy_global_metrics[model]['accQT'] = get_percent_accQT(all_refs, all_preds)
            save_dict_to_json(qt_accuracy_global_metrics, os.path.join(f'data/outputs/qt-results', 'qt_accuracy_global_metrics.json') )
        elif result_type == 2:
            print('Compute QT std and QTd for QTDB and LUDB')
            db_names = ['qt', 'lu']
            annots = ['ref', 'cnn', 'resnet', 'unet', 'wvlt']
            metrics = ['std', 'QTd']
            output = {db_name: {metric: {annot: {'mean': None, 'std': None} for annot in annots} for metric in metrics} for db_name in db_names}
            for db_name in db_names:
                predictions = read_json(db_name = db_name)
                for metric in metrics:
                    for annot in annots: 
                        mn, sd = mean_sd(get_metric_all_leads(predictions, metric = metric)[annot])
                        output[db_name][metric][annot]['mean'] = mn
                        output[db_name][metric][annot]['std'] = sd
            save_dict_to_json(output, os.path.join(f'data/outputs/qt-results', 'qt_std_qtd.json') )
            print('-done')
        elif result_type == 3:
            data = QTDataset()
            for col in ['sel16786_0', 'sel33_0']:
                plot_qt_del_superimposed_beats(data, model= models_unet[0], col = col)
        elif result_type == 4:
            print('Visualize attention maps, U-Net mask and softmax probas...')
            print('Ex 1')
            data = LUDataset(viz=True)
            save_vis_maps_mask_probas(data, models_cnn, models_unet, filepath = 'data/outputs/qt-results/attn-maps-masks/ex1', col='34_i')
            data = QTDataset(viz=True)
            print('Ex 2')
            save_vis_maps_mask_probas(data, models_cnn, models_unet, filepath = 'data/outputs/qt-results/attn-maps-masks/ex2', col='sel100_0')
            print('Ex 3')
            save_vis_maps_mask_probas(data, models_cnn, models_unet, filepath = 'data/outputs/qt-results/attn-maps-masks/ex3', col='sel307_1')
            data = TQTDataset(viz=True)
            print('Ex 4')
            save_vis_maps_mask_probas(data, models_cnn, models_unet, filepath = 'data/outputs/qt-results/attn-maps-masks/ex4', col='c8cb0aaf-6209-4e04-81bb-4b611dd7f1eb', tqt=True)
            print('Ex 5')
            save_vis_maps_mask_probas(data, models_cnn, models_unet, filepath = 'data/outputs/qt-results/attn-maps-masks/ex5', col='b63ff6f6-03f4-4008-88a6-874982b369e0', tqt=True)
            print('-done')
        elif result_type == 5: 
            print('Saving qts auto for TQT analysis...')
            csv_path = 'data/TQTstudy/SCR-002.Clinical.Data.csv'
            raw_path = 'data/TQTstudy/raw/'  
            models = {'cnn': models_cnn, 'resnet': models_resnet, 'unet': models_unet}
            auto = AutoTQT(csv_path, raw_path, models = models)
            output = {method: {fname: None for fname in auto.data.EGREFID} for method in auto.methods}

            with tqdm(total=len(auto.data.EGREFID)) as pbar:
                for fname in auto.data.EGREFID: 
                    dic = auto[fname] 
                    for method in auto.methods:
                        output[method][fname] = dic[method]
                        save_dict_to_json(output[method], f'data/outputs/tqt-analysis/SCR-002.Clinical.Data.withQTauto-{method}.json') 
                    pbar.update() 
            print('-done')
        elif result_type == 6:
            patient_ids = list(range(1001, 1023))
            drugs = ['Dofetilide', 'Quinidine Sulph', 'Ranolazine', 'Verapamil HCL', 'Placebo'] 
            timepoints = [-0.5, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 12, 14, 24]
            data = pd.read_csv('data/TQTstudy/SCR-002.Clinical.Data.csv') 
            wvlt = json.load(open('data/outputs/tqt-analysis/SCR-002.Clinical.Data.withQTauto-wvlt.json', 'r'))
            cnn = json.load(open('data/outputs/tqt-analysis/SCR-002.Clinical.Data.withQTauto-cnn.json', 'r'))
            resnet = json.load(open('data/outputs/tqt-analysis/SCR-002.Clinical.Data.withQTauto-resnet.json', 'r'))
            unet = json.load(open('data/outputs/tqt-analysis/SCR-002.Clinical.Data.withQTauto-unet.json', 'r')) 
            data["QTwvlt"] = [val for _, val in wvlt.items()]  
            data["QTcnn"] = [val for _, val in cnn.items()] 
            data["QTresnet"] = [val for _, val in resnet.items()] 
            data["QTunet"] = [val for _, val in unet.items()] 
            data = data.dropna(subset=['QT'])

            print('Computing DDQTcmax (95% CI)...')
            ddqtc_max  = {drug: {method: {'max': None, 'lower': None, 'upper': None} for method in ['manual', 'wvlt', 'cnn', 'resnet', 'unet']} for drug in drugs[:-1]}
            for drug in drugs[:-1]:
                print(drug)
                for method in ['wvlt', 'cnn', 'resnet', 'unet']:
                    _, _, qt_prolng, qt_prolng_auto = DDqtc_trend(data, timepoints, patient_ids, drug=drug, show=False, auto=method)
                    if not ddqtc_max[drug]['manual']['max']:
                        ddqtc_max[drug]['manual']['max'] = qt_prolng['max_DDQTc']
                        ddqtc_max[drug]['manual']['lower'] = qt_prolng['lower_DDQTc']
                        ddqtc_max[drug]['manual']['upper'] = qt_prolng['upper_DDQTc']
                    if not ddqtc_max[drug][method]['max']:
                        ddqtc_max[drug][method]['max'] = qt_prolng_auto['max_DDQTc']
                        ddqtc_max[drug][method]['lower'] = qt_prolng_auto['lower_DDQTc']
                        ddqtc_max[drug][method]['upper'] = qt_prolng_auto['upper_DDQTc']
                    save_dict_to_json(ddqtc_max, 'data/outputs/tqt-analysis/ddqtc_max.json') 
            print('-done')

            print('Plotting time profiles')
            plot_time_profiles(data, timepoints, patient_ids, drugs)
            print('-done')

            print('BA plots and MGAE')
            methods = ["QTcnn", "QTresnet", "QTunet", "QTwvlt"]
            titles = ["AttnCNN", "KanResWide", "U-Net", "Wavelet"]
            mgae_all_methods = {method: {} for method in methods}
            for method, title in zip(methods, titles):
                print(method[2:])
                mgae = ba_plot_auto_tqt_study(data, auto = method, title=title)
                mgae_all_methods[method] = mgae
                save_dict_to_json(mgae_all_methods, f'data/outputs/tqt-analysis/mgae.json') 
            print('-done')
    else:
        print(f'Saving results for {db_name}')
        if db_name in ['qt', 'lu']:
            _ = evaluate_model(db_name) 
        elif db_name in ['ptb', 'TQT']:
            _ = evaluate_model_ptb_tqt(db_name) 
        elif 'private' in db_name:
            _ = evaluate_model_private(db_name=db_name)
        print('-done') 
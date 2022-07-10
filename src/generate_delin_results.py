import argparse 
import json 
import os
import numpy as np 
import pandas as pd
import csv
import wfdb
import torch
from torch.utils.data import Dataset
from pathlib import Path
from tqdm import tqdm 
from scipy.signal import resample
import model.data_loader as data_loader
from model.data_loader import ECGDataset, normalize 
from compare.wavedel.ecg_delineation import my_delineator
from compare.qrs_detector import get_final_rpeaks, create_template
from generate_qt_results import load_model
from compare.evaluate_delin import compare_annotations
from compare.utils import *

import warnings
warnings.filterwarnings("ignore") 

parser = argparse.ArgumentParser() 
parser.add_argument('--result_type', default=1, help="1: save unet masks; 2: sensitivity, mean error QT delineation, 3: save QTs U-Net asc and desc, , 4: compare U-Net asc vs desc") 

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
                qon  = get_qrs_on(peak, qons, sampling_rate=self.sampling_rate)
                toff  = get_t_off(peak, toffs, sampling_rate=self.sampling_rate) 
                if qon: 
                    qon_ref = 200 - (peak-qon)
                    qon_ref = resample_pos(qon_ref, 2*self.sampling_rate, self.sampling_rate)
                else:
                    qon_ref = qon
                if toff: 
                    toff_ref = 200+ (toff-peak)
                    toff_ref = resample_pos(toff_ref, 2*self.sampling_rate, self.sampling_rate)
                else:
                    toff_ref = toff  
            except TypeError:
                pass
            else:
                try :
                    qon_wvlt , toff_wvlt = get_qt_wvlt(resampled_beat,  sampling_rate = 2*self.sampling_rate) 
                except TypeError:  
                    continue
                else:  
                    masks_unet = get_unet_mask(resampled_beat)   
                    qts_unet = get_unet_qt(resampled_beat)
                    summary = { } 
                    summary['id'] = col
                    summary['position_rpeak'] = peak 
                    summary['ref']  = {'qon': qon_ref, 'toff': toff_ref} 
                    summary['wvlt']  = {'qon': qon_wvlt, 'toff': toff_wvlt} 
                    summary['qt_unet']  = qts_unet
                    summary['mask_unet']  = masks_unet
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
                qon  = get_qrs_on(peak, qons, sampling_rate=self.sampling_rate)
                toff  = get_t_off(peak, toffs, sampling_rate=self.sampling_rate) 
                if qon: 
                    qon_ref = 200 - (peak-qon) 
                else:
                    qon_ref = qon
                if toff: 
                    toff_ref = 200+ (toff-peak) 
                else:
                    toff_ref = toff 
            except (TypeError, AssertionError): 
                pass
            else:  
                try :
                    qon_wvlt , toff_wvlt = get_qt_wvlt(beat,  sampling_rate=self.sampling_rate)
                except TypeError : #returns None 
                    continue
                else:   
                    masks_unet = get_unet_mask(beat)  
                    qts_unet = get_unet_qt(beat)
                    summary = { } 
                    summary['id'] = col
                    summary['position_rpeak'] = peak 
                    summary['ref']  = {'qon': qon_ref, 'toff': toff_ref} 
                    summary['wvlt']  = {'qon': qon_wvlt, 'toff': toff_wvlt} 
                    summary['qt_unet']  = qts_unet
                    summary['unet']  = masks_unet
                    results.append(summary)  
        return results

class PTBDataset(Dataset):
    def __init__(self):
        self.records = [str(path) for path in  Path('data/ptb/').glob('**/*.dat')] 
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

        sig = signals[lead][:30*fs] 
        final_rpeaks, rpeaks, _, _ = get_final_rpeaks(sig, sampling_rate = self.sampling_rate)
        template = np.array(create_template(sig, rpeaks, final_rpeaks, sampling_rate = self.sampling_rate))
        resampled_beat = normalize(resample(template,  len(template)//2))  
        try:
            qon_wvlt , toff_wvlt  = get_qt_wvlt(resampled_beat,  sampling_rate = fs//2)  
            masks_unet = get_unet_mask(resampled_beat)  
        except TypeError:
            summary = { } 
            summary['lead'] = lead
            summary['wvlt']  = None
            summary['unet']  = None 
        else: 
            summary = { } 
            summary['lead'] = lead 
            summary['wvlt']  = {'qon': qon_wvlt, 'toff': toff_wvlt} 
            summary['unet']  = masks_unet
        return summary

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
            qon_wvlt , toff_wvlt  = get_qt_wvlt(resampled_beat,  sampling_rate = fs//2)  
            masks_unet = get_unet_mask(resampled_beat)
        except (ValueError, TypeError):
            summary = { } 
            summary['lead'] = lead
            summary['wvlt']  = None
            summary['unet']  = None    
        else:
            summary = { } 
            summary['lead'] = lead 
            summary['wvlt']  = {'qon': qon_wvlt, 'toff': toff_wvlt} 
            summary['unet']  = masks_unet
        return summary        

def get_unet_mask(beat):
    masks = [[float(val) for val in model(torch.tensor(beat).float().unsqueeze(0).cuda())[2].cpu().detach().numpy()[0][0]]
                for model in models_unet] 
    return masks

def get_unet_qt(beat):
    preds = [compute_qt_nn(beat, model) for model in models_unet]
    return preds

def get_qt_wvlt(ecg_signal, sampling_rate):
    my_markers =  my_delineator(ecg_signal, sampling_rate) 
    try:
        rpeaks = my_markers['ECG_R_Peaks']
        idx_rpeak = position_rpeak(rpeaks)

        qon = my_markers['ECG_R_Onsets'][idx_rpeak]
        toff = my_markers['ECG_T_Offsets'][idx_rpeak]
    except TypeError:
        return None
    else: 
        return int(qon), int(toff)

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
            results = data[col]
            try:
                write_json(results, os.path.join("data/outputs/","masks",db_name,f"{col}.json") ) 
            except FileNotFoundError:
                save_dict_to_json({"predictions":[]}, os.path.join("data/outputs/","masks",db_name,f"{col}.json") ) 
                write_json(results, os.path.join("data/outputs/","masks",db_name,f"{col}.json") )
            pbar.update()   
    return predictions 

def evaluate_model_bis(db_name='ptb'):
    if 'ptb' in db_name: 
        data = PTBDataset()   
    if 'TQT' in db_name:
        data = TQTDataset()
    predictions = {}  
    with tqdm(total=len(data.records)) as pbar:
        for record in data.records:
            results = [] 
            id = record.split('.')[0].split('/')[-1] 
            for lead in data.leads: 
                summary = data[record, lead]  
                results.append(summary) 
            try:
                write_json(results, os.path.join("data/outputs/","masks",db_name,f"{id}.json") ) 
            except FileNotFoundError:
                save_dict_to_json({"predictions":[]}, os.path.join("data/outputs/","masks",db_name,f"{id}.json") ) 
                write_json(results, os.path.join("data/outputs/","masks",db_name,f"{id}.json") )
            pbar.update()  
    return predictions

def get_all_markers(out, db, sampling_rate):   
    markers = {}
    markers['ref'] = out['ref']
    markers['wvlt'] = out['wvlt']
    if db=='lu':
        markers_unet = [qt_pos(mask) for mask in out['unet']]
    else:
        markers_unet = [qt_pos(mask) for mask in out['mask_unet']]
    qons_mask = [val[0] for val in markers_unet if val[0]]
    qon_unet = int(np.mean(qons_mask))
    toff_unet = int(np.mean([val[1] for val in markers_unet if val[1]]))
    markers['unet'] = {'qon': qon_unet, 'toff': toff_unet} 
    qt_unet = np.mean(out['qt_unet'])
    markers['unet_direct'] = {'qon': qon_unet, 'toff': qon_unet + convert_from_msec_to_samples(qt_unet, sampling_rate=sampling_rate)}  
    return markers

def get_results(pathlist, nb_files, pos, sampling_rate, db='not_lu'):
    methods =  ['wvlt', 'unet', 'unet_direct']
    res = {method: {'tp': 0, 'fp': 0, 'fn':0, 'n_test':0, 'diff': [], 'diff_tp': []} for method in methods}
    with tqdm(total=nb_files) as pbar:
        for path in pathlist:
            outs = json.load(open (path, "r"))['predictions'][0] 
            for out in outs:  
                markers = get_all_markers(out, db, sampling_rate)
                if markers['ref'][pos] and markers['wvlt'][pos] and markers['unet'][pos] and markers['unet_direct'][pos]:
                    for method in methods:
                        comparitor = evaluate_method(markers, method = method, pos=pos, sampling_rate = sampling_rate)
                        diff = compute_diff_ms([markers[method][pos]], [markers['ref'][pos]], sampling_rate = sampling_rate)
                        diff_tp = compute_diff_ms(comparitor.matched_test_sample, comparitor.matched_ref_sample, sampling_rate = sampling_rate)                                                                                                                                                                                                                                                                    
                        res[method]['tp'] += comparitor.tp
                        res[method]['fp'] += comparitor.fp
                        res[method]['fn'] += comparitor.fn
                        res[method]['n_test'] += comparitor.n_test
                        res[method]['diff'].extend(diff) 
                        res[method]['diff_tp'].extend(diff_tp) 
                    pbar.update()
    return res
        
def evaluate_method(markers, method = 'unet', pos = 'qon', sampling_rate = 500, window = 0.15):
    auto = markers[method][pos]
    ref = markers['ref'][pos] 
    comparitor = compare_annotations(ref_sample = np.array(sorted([auto])),
                                    test_sample = np.array(sorted([ref])),
                                    window_width = int(window * sampling_rate)) 
    return comparitor

def compute_diff_ms(lst1, lst2, sampling_rate):
    diffs = [1000*(val1-val2)/sampling_rate for val1, val2 in zip(lst1, lst2)]
    return diffs 

def compute_sens_pp(tp, fn, n_test):
    sensitivity = float(tp) / float(tp + fn)
    positive_predictivity = float(tp) / n_test
    return sensitivity, positive_predictivity

def get_qt_mean_mask(fname, db_name, directory='data/outputs/masks'):
    outs = json.load(open (f'{directory}/{db_name}/{fname}.json', "r"))['predictions']
    qts_unet_mask  = [get_all_qts(out, db_name)['unet'] for out in outs[0]] 
    qts_unet_mask = [val for val in qts_unet_mask if val] 
    return np.mean(qts_unet_mask)

def get_all_qts(out, db_name): 
    if db_name == 'qt':
        markers_unet = [qt_pos(mask) for mask in out['mask_unet']] 
    else:
        markers_unet = [qt_pos(mask) for mask in out['unet']]
    qons_mask = [val[0] for val in markers_unet if val[0]] 
    qon_unet = int(np.mean(qons_mask))
    toff_unet = int(np.mean([val[1] for val in markers_unet if val[1]])) 
    qt_wvlt = compute_qt_ms(out['wvlt']['qon'], out['wvlt']['toff'])
    qt_unet = compute_qt_ms(qon_unet, toff_unet)

    if db_name in ['qt', 'lu']:
        qt_ref = compute_qt_ms(out['ref']['qon'], out['ref']['toff'])
        if qt_ref:
            return {'ref' : qt_ref,
                    'wvlt' : qt_wvlt,
                    'unet' : qt_unet}
        else:
            return {'ref' : None, 'wvlt' : None, 'unet' : None}
    else:
        if qt_wvlt and qt_unet:
            return {'wvlt' : qt_wvlt,
                    'unet' : qt_unet}
        else:
            return {'wvlt' : None, 'unet' : None} 

def compute_qt_ms(qon, toff, fs = 500):
    if qon and toff:
        return 1000*(toff-qon)/fs
    else:
        return None

def get_index_fname(fname, results_keys):
    for idx, key in enumerate(results_keys):
        if fname==key:
            return idx

if __name__ == '__main__': 
    args = parser.parse_args()  
    result_type = int(args.result_type)
    
    if result_type == 1:
        # Set the random seed for reproducible experiments
        torch.manual_seed(230)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(230)
        
        print('Loading models ...')
        nfold = 5
        models_unet = [load_model('experiments/unet_strat', fold=fold)  for fold in range(nfold)] 
        print('-done')

        for db_name in ['qt', 'lu', 'ptb','TQT']:
            print(f'Evaluating models on {db_name}')
            if db_name in ['qt', 'lu']:
                _ = evaluate_model(db_name) 
            elif db_name in ['ptb','TQT']:
                _ = evaluate_model_bis(db_name=db_name) 
    elif result_type ==2:
        nb_files, sampling_rate = 20000, 500
        positions = 'qon', 'toff'
        methods = ['wvlt', 'unet', 'unet_direct']
        db_names = ['qt', 'lu']

        output  = {db_name: {pos: {method: {'me': None, 'me_std': None, 'sens': None, 'me_tp': None, 'me_tp_std': None} for method in methods} for pos in positions} for db_name in db_names}

        for db_name in db_names: 
            for pos in positions:
                print(db_name, pos)
                pathlist = Path(f'data/outputs/masks/{db_name}').glob('**/*.json')
                if db_name =='lu':
                    res = get_results(pathlist, nb_files, pos, sampling_rate, db=db_name)
                else:
                    res = get_results(pathlist, nb_files, pos, sampling_rate)
                for method in methods:
                    sensitivity, _ = compute_sens_pp(res[method]['tp'], res[method]['fn'], res[method]['n_test'])
                    output[db_name][pos][method]['sens'] = round(100*sensitivity, 2) 

                    mean_diff, std_diff = mean_sd(res[method]['diff'])
                    output[db_name][pos][method]['me'] = round(mean_diff, 1) 
                    output[db_name][pos][method]['me_std'] = round(std_diff, 1)  

                    mean_diff_tp, std_diff_tp = mean_sd(res[method]['diff_tp']) 
                    output[db_name][pos][method]['me_tp'] = round(mean_diff_tp, 1) 
                    output[db_name][pos][method]['me_tp_std'] = round(std_diff_tp, 1)   
                    save_dict_to_json(output, os.path.join(f'data/outputs/masks', 'sens_me.json') )
    elif result_type ==3:
        db_names = ['qt', 'lu', 'ptb', 'TQT']
        qts_all_db = {db_name: {'ref': [], 'unet': [], 'unet_mask': []} for db_name in db_names}

        for db_name in db_names[:2]:
            print(db_name)
            results = json.load(open (f'data/outputs/qt-results/results_{db_name}.json', "r"))['predictions']
            qts = {'ref': [], 'unet': [], 'unet_mask': []}
            with tqdm(total=len(results)) as pbar:
                for dic in results:
                    fname = list(dic.keys())[0] 
                    qt_unet = np.mean(dic[fname]['unet'])
                    qt_ref = np.mean(dic[fname]['ref'])
                    qt_unet_mask = get_qt_mean_mask(fname, db_name=db_name)
                    qts_all_db[db_name]['ref'].append(qt_ref)
                    qts_all_db[db_name]['unet'].append(qt_unet)
                    qts_all_db[db_name]['unet_mask'].append(qt_unet_mask) 
                    pbar.update() 
            print(len(qts_all_db[db_name]['unet_mask']))
            save_dict_to_json(qts_all_db, os.path.join(f'data/outputs/masks', 'qt_unet_asc_desc.json') )
        for db_name in db_names[2:]:
            print(db_name)
            if db_name == 'ptb':
                leads = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'vx', 'vy', 'vz']
            else:
                leads = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
            pathlist_qts = Path(f'data/outputs/qt-results/results_{db_name}').glob('**/*.json') 
            results = [read_json(json_path=json_path)[0] for json_path in pathlist_qts]   
            pathlist = Path(f'data/outputs/masks/{db_name}').glob('**/*.json')
            
            nb_files = len(leads)*len([path for path in pathlist]) 
            results_keys = [list(dic.keys())[0] for dic in results] 
            with tqdm(total=nb_files) as pbar:
                pathlist = Path(f'data/outputs/masks/{db_name}').glob('**/*.json')
                for path in pathlist: 
                    outs = json.load(open (path, "r"))['predictions'][0]
                    for out in outs:  
                        if out['unet']:
                            qt_unet_mask  = get_all_qts(out, db_name)['unet'] 
                            lead = out['lead'] 
                            fname = str(path).split('/')[-1].split('.')[0] +'_'+lead 
                            dic = results[get_index_fname(fname, results_keys)]
                            assert(list(dic.keys())[0]==fname)
                            qt_unet = np.mean(dic[fname]['unet'])
                            qt_ref = np.mean(dic[fname]['ref'])
                            fname_wo_lead = '_'.join(fname.split('_')[:-1])  
                            if qt_unet_mask:
                                qts_all_db[db_name]['ref'].append(qt_ref)
                                qts_all_db[db_name]['unet'].append(qt_unet)
                                qts_all_db[db_name]['unet_mask'].append(qt_unet_mask) 
                                pbar.update()
                    save_dict_to_json(qts_all_db, os.path.join(f'data/outputs/masks', 'qt_unet_asc_desc.json') )
            print(len(qts_all_db[db_name]['unet_mask']))
    elif result_type ==4:
        out = json.load(open ("data/outputs/masks/qt_unet_asc_desc.json", "r"))
        qts_wo_nan = {key: [] for key in ['ref', 'unet', 'unet_mask']}
        for db_name, qts in out.items():
            for ref, unet, unet_mask in zip(qts['ref'], qts['unet'], qts['unet_mask']):
                if str(ref)!='nan' and str(unet)!='nan' and str(unet_mask)!='nan': 
                    qts_wo_nan['ref'].append(ref)
                    qts_wo_nan['unet'].append(unet)
                    qts_wo_nan['unet_mask'].append(unet_mask)
        print(len(qts_wo_nan['ref']))

        diffs, mean_diff, upper, lower = stat_difference(qts_wo_nan['unet'], qts_wo_nan['unet_mask'])  
        refs = qts_wo_nan['ref']
        fig, ax = plt.subplots(1, 2, figsize = (25, 12), dpi=400)
        ax = ax.flatten()
        fontsize = 25
        ax[0].scatter(refs, diffs, s = 50)
        ax[0].axhline(y = mean_diff, color='k', linestyle='-', linewidth=3)
        ax[0].axhline(y = upper, color='r', linestyle='-.', linewidth=3)
        ax[0].axhline(y = lower, color='r', linestyle='-.', linewidth=3)
        ax[0].tick_params(labelsize=fontsize)
        ax[0].set_title(f'Mean (95% LOA): {round(mean_diff)} ({round(lower)}; {round(upper)})', fontsize = fontsize)
        ax[0].set_xlabel('Manual (ms)', fontsize = fontsize)
        ax[0].set_ylabel('QT U-Net↓ - U-Net↑ (ms)', fontsize = fontsize) 

        ax[1].scatter(refs, qts_wo_nan['unet_mask'], label = 'U-Net↑', s = 50, facecolors='none', edgecolors='silver')
        ax[1].scatter(refs, qts_wo_nan['unet'], label = 'U-Net↓', s = 50)
        ax[1].plot(refs, refs, color='k')
        ax[1].tick_params(labelsize=fontsize)
        ax[1].set_ylabel('Automatic (ms)', fontsize = fontsize)
        ax[1].set_xlabel('Manual (ms)', fontsize = fontsize)
        ax[1].legend(fontsize = fontsize) 
        plt.savefig('data/outputs/masks/ba_scatter_unet_asc_vs_desc.png')

        good, bad = [], []
        for ref, unet, unet_mask in zip(qts_wo_nan['ref'], qts_wo_nan['unet'], qts_wo_nan['unet_mask']): 
            if abs(ref-unet)>15:
                bad.append(unet-unet_mask)
            else:
                good.append(unet-unet_mask)

        fontsize = 15
        fig, ax = plt.subplots(figsize = (7, 5), dpi=400)
        bp = ax.boxplot([good, bad], widths = 0.7, showfliers=False) 
        plt.setp(bp["medians"], color='k', linewidth=2*1.5)
        plt.xticks([1, 2],['Accurate \n (71%)', 'Inaccurate \n (29%)'], fontsize = fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.ylabel('QT U-Net↓ - U-Net↑ (ms)', fontsize = fontsize)
        plt.savefig('data/outputs/masks/boxplot_unet_asc_desc_accuracy.png')
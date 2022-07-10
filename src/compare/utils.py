import csv
import os
import numpy as np
import pandas as pd
import torch 
import json
import matplotlib.pyplot as plt
from scipy.signal import resample
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.preprocessing import  StandardScaler
import wfdb
from .wavedel.ecg_delineation import my_delineator  
from .qrs_detector import convert_from_msec_to_samples, convert_from_samples_to_sec, get_final_rpeaks, create_template

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
 
def get_loss_weights(log_path, nfold=5):
    compt=0
    output = {f'fold_{fold}': [] for fold in range(1, nfold+1)}
    with open(log_path) as f:
        f = f.readlines()
        for line in f:
            if 'requires_grad=True' in line:
                weights_str = line.split(', requires_grad')[0].split('tensor(')[-1].strip('][').split(', ')
                weights = [float(val) for val in weights_str]
                compt += 1
                if compt in range(1, 31):
                    output['fold_1'].append(weights)
                if compt in range(31, 61):
                    output['fold_2'].append(weights)
                if compt in range(61, 91):
                    output['fold_3'].append(weights)
                if compt in range(91, 121):
                    output['fold_4'].append(weights)
                if compt in range(121, 151):
                    output['fold_5'].append(weights)
    assert(compt==150)
    return output

### Weight losses
def plot_loss_weights(output, model, filepath):
    linewidth = 1.5
    fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(6,8), dpi=400)
    if 'unet' in model:
        ax2 = ax[0].twinx() 
    compt=0
    for key in output.keys():
        y1 = [weights[0] for weights in output[key]]
        y2 = [weights[1] for weights in output[key]]
        if 'unet' in model:
            y3 = [weights[2] for weights in output[key]]
        
        ax[0].plot(np.exp(-np.array(y1)), label= 'MSE', linewidth=linewidth, color = 'dimgray')
        ax[0].plot(np.exp(-np.array(y2)), label= 'CE', linewidth=linewidth, color ='tab:blue' )  
        ax[0].set_ylabel('Loss weight exp(-γ)', fontsize=15)
        ax[0].tick_params(axis='y', labelsize=15)
        if 'unet' in model:
            ax2.plot(np.exp(-np.array(y3)), label= 'dice', linewidth=linewidth, color = 'y')
            ax2.set_ylabel('Dice weight', fontsize=15, color ='y')
            ax2.tick_params(axis='y', labelsize=15)

        ax[1].plot(np.array(y1), linewidth=linewidth, color = 'dimgray', label = 'MSE')
        ax[1].plot(np.array(y2), linewidth=linewidth, color = 'tab:blue', label = 'CE')
        if 'unet' in model:
            ax[1].plot(np.array(y3), linewidth=linewidth, color = 'y', label = 'Dice')  
        ax[1].set_xlabel('Epoch', fontsize=15)
        ax[1].set_ylabel('bias γ', fontsize=15)
        ax[1].tick_params(axis='x', labelsize=15)
        ax[1].tick_params(axis='y', labelsize=15)
        
        if compt == 0:
            ax[1].legend(frameon=False, fontsize=11) 
        compt+=1 
    plt.savefig(filepath)


### Bland-Altmann plot
def ba_plot(predictions, model='cnn', db_name='qt', print_metric=True, show=False) : 
    output = get_metric_by_lead(predictions, db_name, metric = 'mean')

    refs_w_nan = flatten([item for _,item in output['ref'].items()])
    preds_w_nan = flatten([item for _,item in output[model].items()])
    refs = [val1 for val1, val2 in zip(refs_w_nan, preds_w_nan) if str(val1)!='nan' and str(val2)!='nan'] 
    preds = [val2 for val1, val2 in zip(refs_w_nan, preds_w_nan) if str(val1)!='nan' and str(val2)!='nan'] 
    diffs, mean_diff, upper, lower = stat_difference(refs, preds) 
    if show: 
        plot_results(refs, preds, diffs, mean_diff, upper, lower, title = model) 
    return refs, preds, diffs

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

### Boxplot performance accross leads

def visualize_boxplot_by_lead(summary, leads, capitalized_leads, filepath, metric = 'mean', ylabel = 'QT mean (ms)', xlabel='Lead',legend=False, fontsize=20): 
    if 'mae' in metric:
        colors = ['silver', 'white', 'k', 'tab:cyan']
    else:
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'grey', 'tab:cyan']
    
    methods = ['ref', 'resnet','wvlt']
    figsize = (12, 8)
    fig, ax = plt.subplots(figsize = figsize, dpi=400) #
    for index, lead in enumerate(leads): 
        data = [val for key, val in summary[metric].items() if key.split('_')[-1] == lead 
                                           and key.split('_')[0] in methods] 
        
        labels =  ['Wavelet', 'KanResWide'] 
   
        if 'mae' in metric:
            positions = [(i+1) + 4*index  for i in range(len(methods)-1)] 
        else:
            positions = [(i+1) + 4*index for i in range(len(methods))] 
         
        bp = ax.boxplot(data, positions=positions, patch_artist=True, showfliers=False, widths=0.6)
        for box, color in zip(bp["boxes"], colors):
            box.set_facecolor(color)
        plt.setp(bp["medians"], color='k', linewidth=2*1.5)
            
    if 'mae' in metric:
        position_labels = [len(methods)/2 +  index*4  for index in range (len(leads))]
    else:
        position_labels = [(len(methods)+1)/2 +  index*4  for index in range (len(leads))] 
        
    if legend:
        ax.legend(bp["boxes"], labels, loc='upper right', frameon= False, fontsize = fontsize) 
    ax.set_ylabel(ylabel ,fontsize = fontsize) 
    ax.set_xlabel(xlabel ,fontsize = fontsize) 
    ax.set_xticks(position_labels)
    ax.set_xticklabels(capitalized_leads) 
    ax.tick_params( labelsize = fontsize) 
    plt.savefig(filepath) 

### Analyze variability     
def get_metric_all_leads(predictions, metric = 'mean', lead = 'all'):
    keys = ['ref', 'wvlt','cnn','resnet','unet']
    if 'mae' in metric:
        output = {key : [compute_metric(results[key], metric, ref = results['ref'])  for col, results in get_results(predictions)] 
                  for key in keys if 'ref' not in key} 
    elif 'QTd' in metric:
        output = {key : flatten([res[key] for res in  get_qtd_record(predictions)]) for key in keys} 
    else:
        output = {key : [compute_metric(results[key], metric)  for col, results in get_results(predictions)] for key in keys} 
    return output
 
def get_metric_by_lead(predictions, db_name, metric = 'mean'):
    keys = ['ref', 'wvlt','cnn','resnet','unet']
    if 'qt' in db_name:
        leads = ['0', '1']
    elif 'ptb' in db_name:
        leads = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'vx', 'vy', 'vz']
    elif 'TQT' in db_name:
        leads = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    else:
        leads = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1','v2','v3','v4','v5','v6']
    output = {}
    for key in keys:
        if 'mae' in metric:
            if 'ref' in key:
                continue
            result_leads = {lead: [compute_metric(results[key], metric, ref = results['ref'])  for col, results in get_results(predictions)
                                    if lead in col.split('_')[-1] and len(lead)==len(col.split('_')[-1])] for lead in leads }
            output[key] = result_leads 
        else:   
            result_leads = {lead: [compute_metric(results[key], metric)  for col, results in get_results(predictions) 
                            if lead in col.split('_')[-1] and len(lead)==len(col.split('_')[-1])] for lead in leads }
            output[key] = result_leads 
    return output


def concatenate_dict(predictions, db_name):
    summary = {}
    metrics = [ 'mae', 'mean','std','QTd'] 
    for metric in metrics:
        metric_by_lead = get_metric_by_lead(predictions, db_name, metric  = metric)
        d = {}
        for key1, dict1 in metric_by_lead.items():
            for key2, val in dict1.items():
                d[key1 + '_' + key2] = val 
        summary[metric] = d
    return summary

def compute_metric(val, metric, ref = None):
    if 'mean' in metric:
        return np.mean(val)
    if 'std' in metric:
        return np.std(val)
    # if 'QTd' in metric:
    #     return np.max(val)-np.min(val)
    if 'mae' in metric: 
        return np.abs(np.mean(ref)-np.mean(val))

def get_results(predictions):
    for d in predictions:
        col, results = list(d.items())[0]
        yield col, results

def get_qtd_record(predictions): 
    keys = ['ref', 'wvlt','cnn','resnet','unet']
    concat = concat_results_by_record(predictions)
    for patient_id, item in concat.items():
        temp = {key: list(zip(*[lst[key] for lst in item]))for key in keys} 
        res = {key: [np.max(vals) - np.min(vals) for vals in temp[key]] for key in keys}
        yield res
    
def concat_results_by_record(predictions):
    records = set([list(d.keys())[0].split('_')[0] for d in predictions])
    concat = {rec: [] for rec in records}
    for rec in records: 
        for d in predictions: 
            col, results = list(d.items())[0]  
            if rec in col and len(rec)==len(col.split('_')[0]): 
                mean_results = {key: [np.mean(item)] for key, item in results.items()}
                concat[rec].append(mean_results) 
    return concat

def convert_in_msec(duration, sampling_rate):
    return 1000*duration/sampling_rate  

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

def flatten(lst):
    flat_lst = []
    for val in lst:
        flat_lst.extend(val)
    return flat_lst   

def save_dict_to_json(d, json_path): 
    with open(json_path, 'w') as f: 
        json.dump(d, f, indent = 6) 

def write_json(new_data, json_path):
    with open(json_path,'r+') as f: 
        file_data = json.load(f)
        file_data["predictions"].append(new_data) 
        f.seek(0) 
        json.dump(file_data, f, indent = 6)

def read_json(db_name = None, json_path = None):
    if not json_path:
        json_path = os.path.join('data/outputs/qt-results', f"results_{db_name}.json")
    with open(json_path,'r') as f: 
        file_data = json.load(f)
        predictions = file_data["predictions"] 
    return predictions

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

def get_percent_accQT(refs, preds):
    acc = [abs(ref-pred)<=15 for ref, pred in zip(refs, preds)]
    nb_acc = len([val for val in acc if val])
    return round(100*nb_acc/len(refs), 0)

def mean_sd(values): 
    return round(np.mean(values), 0), round(np.std(values), 0)

def compute_qt_nn(beat, model):
    logits_mdl = model(torch.tensor(beat).float().unsqueeze(0).cuda())[0].cpu().detach()
    pred = compute_qt(classify_qt(logits_mdl, dim=1)).item()
    return pred

def compute_qt(qt_class):    
     intervals = range(150, 752, 2) 
     qt_value = 0 
     for index, (lower, upper) in enumerate(zip(intervals[:-1],intervals[1:] )): 
         if qt_class == index:
             qt_value = (lower+upper)/2
             break
     return torch.tensor(qt_value)

def classify_qt(outputs, dim=0):   
    m = torch.nn.Softmax(dim=dim)
    qt_class = torch.argmax(m(outputs)).item()    
    return qt_class 

def get_unet_mask(beat, model):
    mask = model(torch.tensor(beat).float().unsqueeze(0).cuda())[2].cpu().detach() 
    return mask

def qt_pos(mask):
    processed = [int(val>max(mask)/2) for val in mask] 
    processed[:50] = [0 for _ in range(50)]
    processed[len(processed)-50:] = [0 for _ in range(50)]
    qt = np.where(np.array(processed) == 1)[0] 
    if len(qt)==0:  
        return None, None
    qpos, tpos = qt[0], qt[-1]
    return qpos, tpos

def resample_pos(pos, new_fs, old_fs): 
    new_pos = new_fs*(pos-100)//old_fs #100 relates to rpeak ?
    return int(new_pos)

def normalize(signal):
    signal_normalized = StandardScaler().fit_transform(np.array(signal).reshape(-1, 1) ).reshape(len(signal))  
    return signal_normalized

### Plot delineation 28 beats qtdb

def plot_qt_del_superimposed_beats(data, col, model, fontsize = 15):  
    signal = data.db[col]
    plt.figure(figsize=(7, 5), dpi = 400)
    qts, qts_unet = [], []
    for d in data[col]:
        qts.append(d['qt_ref'])
        qts_unet.append(d['qt_pred_unet'])
        peak = d['position_rpeak']
        beat = np.array(signal)[peak-200:peak+600-200]
        resampled_beat = resample(beat, 2*len(beat))[200:800]
        mask_unet = get_unet_mask(resampled_beat, model).detach().numpy()[0][0] 
        qon, toff = qt_pos(mask_unet) 
        time =  [2*i for i in range(600)]
        plt.plot(time, resampled_beat, color = 'k')
        plt.plot((2*qon, 2*qon), (0.4,0.4),'|', ms=20, color ='y')
        plt.plot((2*toff, 2*toff), (0.4,0.4),'|', ms=20, color ='y')
        qon, toff = d['qon_ref'], d['toff_ref'] 
        plt.plot(((1000//250)*qon - 400,(1000//250)*qon - 400) , (0.1,0.1),'-|', ms=20, color ='k') #400 = (1000//250)*100
        plt.plot(((1000//250)*toff - 400,(1000//250)*toff - 400) , (0.1,0.1),'-|', ms=20,color ='k') 
    plt.xlabel('Time (ms)', fontsize=fontsize)
    plt.ylabel('Amplitude (mV)', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    filepath = f'data/outputs/qt-results/qt_del_superimposed_beats/{col}-std_ref_{round(np.std(qts),0)}-std_unet_{round(np.std(qts_unet),0)}.png'
    plt.savefig(filepath) 

### Visualize attention maps, mask unet, softmax probas

def get_attention_maps(model, normalized_beat, beat, filepath, show = True):
    outputs = model(torch.tensor(normalized_beat).float().unsqueeze(0).cuda())
    a1 = process_map(outputs[2].cpu())  
    a2 = process_map(outputs[3].cpu())  
    a3 = process_map(outputs[4].cpu()) 
    if show == True:
        show_map(beat, a1, title='Map 1', filepath=filepath, xlabel= True, filename='map1')
        show_map(beat, a2, title='Map 2', filepath=filepath, xlabel= True, filename='map2')
        show_map(beat, a3, title='Map 3', filepath=filepath, xlabel= True, filename='map3')
    return a1, a2, a3

def process_map(a, length=600): 
    a = F.interpolate(a, scale_factor= length/a.size()[-1], mode='linear', align_corners=False)
    a = a.view(a.size()[0], a.size()[1]*a.size()[2])  
    cam = np.maximum(a.detach().numpy()[0], 0) 
    heatmap = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-10)
    return heatmap   

def show_map(beat, a, title, filepath, fontsize=15, xlabel=False, filename='map1'): 
    x, y = beat, a 
    plt.rcParams["figure.figsize"] = 10, 3 
    fig, ax = plt.subplots(nrows=1, sharex=True, dpi=400) 
    time = np.array([2*i for i in range(600)])
    extent = [0, 1200, min(x).item() ,max(x).item()]
    img = ax.imshow(y[np.newaxis,:], cmap="plasma", aspect="auto", extent=extent)
    #ax.set_yticks([])
    #ax.set_xticks([]) 
    ax.set_xlim(extent[0], extent[1]) 
    ax.plot(time, x, color='w', linewidth=5)
    ax.set_title(title, fontsize=fontsize)
    ax.set_ylabel('Amplitude (mV)')
    if xlabel:
        ax.set_xlabel('Time (ms)') 
    plt.colorbar(img, ax=ax, fraction=0.46, pad=0.04)
    plt.tight_layout() 
    plt.savefig(os.path.join(filepath, f'{filename}.png'), bbox_inches='tight')

def get_probas(beat, model):
    mdl_output = model(torch.tensor(beat).float().unsqueeze(0).cuda())
    logits  = mdl_output[0].cpu().detach()
    m = torch.nn.Softmax(dim=1)
    probas = m(logits)
    return probas.detach().numpy()[0]

def process_mask(mask, qt_unet, tqt = False): 
    processed = [int(val>max(mask)/2) for val in mask] 
    processed[:50] = [0 for _ in range(50)]
    processed[len(processed)-50:] = [0 for _ in range(50)]
    qt = np.where(np.array(processed) == 1)[0] 
    if len(qt)==0:  
        return None, None
    qpos = qt[0] 
    tpos = int(qpos + convert_from_msec_to_samples(qt_unet, sampling_rate=500))
    new_mask = np.array([0 for _ in range(len(mask))])
    new_mask[qpos:tpos+1]  = 1
    if not tqt:
        return new_mask
    else:
        return qpos, new_mask 

def save_vis_maps_mask_probas(data, models_cnn, models_unet, filepath = 'data/outputs/qt-results/attn-maps-masks/ex1', col='34_i', tqt=False, fontsize = 15): 
    if not tqt:
        outs = data[col]
    else:
        record, lead = col , 'V4'
        id = record.split('.')[0].split('/')[-1] 
        outs = [data[record, lead]] 
    for out in outs:
        beat = out['beat']
        normalized_beat = out['normalized_beat']
        if not tqt:
            qon , toff = out['qon_ref'] , out['toff_ref'] 
        else:
            qt_ref = out['qt_ref']
            if qt_ref:  
                qon, toff = True, True
        if qon and toff:
            if not tqt:
                qt_ref = convert_from_samples_to_sec(toff-qon, data.sampling_rate, milli=True)
            i = 0   
            model_unet = models_unet[i]
            qt_unet = compute_qt_nn(normalized_beat, model_unet)  
            if col in ['34_i', 'sel307_1']:
                show = abs(qt_ref - qt_unet)>200 
            elif col == 'sel100_0':
                show = abs(qt_ref - qt_unet)<5 
            elif tqt:
                show = True
            if show:
                a1, a2, a3 = get_attention_maps(models_cnn[i], normalized_beat, beat, filepath) 
                mask_unet = [float(val) for val in model_unet(torch.tensor(normalized_beat).float().unsqueeze(0).cuda())[2].cpu().detach().numpy()[0][0]] 
                intervals = [compute_qt(i) for i in range(250)]
                probas = get_probas(normalized_beat, model_unet)
                plt.rcParams["figure.figsize"] = 7, 3
                fig, ax = plt.subplots(nrows=1, dpi=400) 
                ax.plot(intervals[:-1], probas[:-1], linewidth=3)  
                ax.set_xlabel('QT interval (ms)', fontsize = fontsize)
                ax.set_ylabel('Softmax probability', fontsize = fontsize)
                ax.axvline(qt_ref, ls='-.', color='k', label ='Manual: ' + str(qt_ref) + 'ms', linewidth=2) 
                ax.axvline(qt_unet, ls='-.', color='y', label ='U-Net: ' + str(qt_unet) + 'ms', linewidth=2)
                ax.tick_params(labelsize = fontsize) 
                ax.legend(frameon=False, loc = 'upper left', fontsize =fontsize-2)
                plt.savefig(os.path.join(filepath, 'probas.png'), bbox_inches='tight')

                plt.rcParams["figure.figsize"] = 10, 3
                fig, ax = plt.subplots(nrows=1, dpi=400)
                y = np.array(process_mask(mask_unet, qt_unet))
                x = beat 
                time = np.array([2*i for i in range(600)])
                extent = [0, 1200, min(x).item() ,max(x).item()]
                ax.plot(time, x, color='thistle', linewidth=5) 
                img = ax.imshow(y[np.newaxis,:], cmap="Blues", aspect="auto", extent=extent) 
                if col =='34_i':
                    ax.plot(2*qon, x[time==2*qon][0], 'X', color='k', markersize=10)
                    ax.plot(2*toff, x[time==2*toff][0], 'X', color='k', markersize=10)
                elif col in ['sel100_0', 'sel307_1']:
                    toff_ref_ms = 2*resample_pos(toff, 500, 250)
                    ampli_toff_ref  = x[time==toff_ref_ms ][0]
                    ax.plot(toff_ref_ms,ampli_toff_ref, 'X', color='k', markersize=10) 
                    qon_ref_ms = 2*resample_pos(qon, 500, 250)
                    ampli_qon_ref  = x[time==qon_ref_ms ][0]
                    ax.plot(qon_ref_ms,ampli_qon_ref, 'X', color='k', markersize=10) 
                elif tqt:
                    qon, mask = process_mask(mask_unet, qt_unet, tqt=True)
                    toff = qon + convert_from_msec_to_samples(qt_ref, sampling_rate=500)
                    ax.plot(2*qon, x[time==2*qon][0], 'X', color='k', markersize=10)
                    ax.plot(2*toff, x[time==2*toff][0], 'X', color='k', markersize=10)
                ax.tick_params(labelsize=fontsize)
                ax.set_xlim(extent[0], extent[1])
                plt.colorbar(img, ax=ax, fraction=0.46, pad=0.04)
                ax.set_xlabel('Time (ms)', fontsize=fontsize) 
                ax.set_ylabel('Amplitude (mV)', fontsize=fontsize) 
                plt.tight_layout() 
                plt.savefig(os.path.join(filepath, 'mask.png'), bbox_inches='tight')
                break

#### Analyze TQT study

class AutoTQT(Dataset):
    def __init__(self, csv_path, raw_path, models):
        self.data = pd.read_csv(csv_path) 
        self.raw_path = raw_path
        self.models = models
        self.methods = ['wvlt', 'cnn', 'resnet', 'unet']

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, fname): 
        'Generates one sample of data'
        patient_id = list(self.data[self.data.EGREFID==fname].RANDID)[0]
        record = wfdb.rdrecord(self.raw_path+str(patient_id)+'/'+fname) 
        signals, record_name  = get_signal(record)  
        fs  = record.fs   

        qts_leads = {method: [] for method in self.methods}
        for lead in signals.columns:
            sig = signals[lead] 
            try:
                final_rpeaks, rpeaks, _, _ = get_final_rpeaks(sig, sampling_rate = fs)
                template = np.array(create_template(sig, rpeaks, final_rpeaks, sampling_rate = fs))
                resampled_beat = normalize(resample(template,  len(template)//2))   
                pred = {method: None for method in self.methods}
                for method in pred.keys():
                    if method in ['cnn', 'resnet', 'unet']:
                        pred[method]= np.mean([compute_qt_nn(resampled_beat, model) for model in self.models[method]]) 
                    else:
                        pred[method] = get_qt_mask_wvlt(resampled_beat,  sampling_rate = fs//2)[0] 
            except (ValueError, TypeError):
                pass
            else: 
                for method in qts_leads.keys():
                    qts_leads[method].append(pred[method]) 

        mean_qt = {method: None for method in self.methods}
        for method, qts in qts_leads.items():
            if len(qts)>0:
                mean_qt[method]  = np.mean(qts)
            else:
                mean_qt[method]  = np.nan
        return mean_qt

def ba_plot_auto_tqt_study(data, auto = "QTwvlt", title='Wavelet', fontsize=30): 
    mgae = {key: {} for key in ['all', 'Dofetilide', 'Quinidine Sulph', 'Ranolazine','Verapamil HCL', 'Placebo']}
    fig, ax = plt.subplots(1, 2, figsize = (2*12, 12), dpi=400 ) 
    refs = [val1 for val1, val2 in zip(data["QT"], data[auto]) if str(val1)!='nan' and str(val2)!='nan' ]
    preds = [val2 for val1, val2 in zip(data["QT"], data[auto]) if str(val1)!='nan' and str(val2)!='nan']
    diffs, mean_diff, upper, lower = stat_difference(refs, preds) 
    mgae['all'] = {'nb_qts': len(refs), 'mae': round(np.mean(np.abs(diffs)), 1), 'mae_std': round(np.std(np.abs(diffs)), 1)}
    ax[0].axhline(y = mean_diff, color='k', linestyle='-', linewidth=3)
    ax[0].axhline(y = upper, color='r', linestyle='-.', linewidth=3)
    ax[0].axhline(y = lower, color='r', linestyle='-.', linewidth=3)
    ax[1].plot(refs, refs, color='k')
    ax[0].set_title(f'Mean (95% LOA): {round(mean_diff)} ({round(lower)}; {round(upper)})', fontsize = fontsize)
    dic = ba_plot(data, ax, pred = auto, cdrug='Ranolazine', ymin=300, ymax=510)
    mgae['Ranolazine'] = dic
    dic = ba_plot(data, ax, pred = auto, cdrug='Verapamil HCL', ymin=300, ymax=510)
    mgae['Verapamil HCL'] = dic
    dic = ba_plot(data, ax, pred = auto, cdrug='Placebo', ymin=300, ymax=510)
    mgae['Placebo'] = dic
    dic = ba_plot(data, ax, pred = auto, cdrug='Dofetilide', ymin=300, ymax=510, hue='silver', style='none')  
    mgae['Dofetilide'] = dic
    dic = ba_plot(data, ax, pred = auto, cdrug='Quinidine Sulph', ymin=300, ymax=510, hue='silver', style='none')
    mgae['Quinidine Sulph'] = dic
    ax[0].tick_params(labelsize=fontsize) 
    ax[0].set_xlabel('Manual (ms)', fontsize = fontsize)
    ax[0].set_ylabel('Manual - Automatic (ms)', fontsize = fontsize) 
    ax[0].set_ylim(-50, 225) 
    ax[1].tick_params(labelsize=fontsize) 
    ax[1].set_xlabel('Manual (ms)', fontsize = fontsize)
    ax[1].set_ylabel('Automatic (ms)', fontsize = fontsize) 
    ax[1].set_title(title, fontsize = fontsize)
    ax[1].set_ylim(320, 490)  
    plt.savefig(f'data/outputs/tqt-analysis/ba-plots/{auto}.png', bbox_inches='tight')
    return mgae

def ba_plot(data, ax, style='', pred = "QTwvlt", cdrug='Dofetilide', ymin=320, ymax=470, hue='tab:blue'):  
    refs = [val1 for drug, val1, val2 in zip(data["EXTRT"], data["QT"], data[pred]) if str(val1)!='nan' and str(val2)!='nan'
                                                                       and drug == cdrug]
    preds = [val2 for drug, val1, val2 in zip(data["EXTRT"], data["QT"], data[pred]) if str(val1)!='nan' and str(val2)!='nan'
                                                                       and drug == cdrug]
    diffs, mean_diff, upper, lower = stat_difference(refs, preds) 
    #print(f'MAE: {round(np.mean(np.abs(diffs)), 4)} ± {round(np.std(np.abs(diffs)), 4)}')
    #print(f'Mean (95% LOA): {round(mean_diff)} ({round(lower)}; {round(upper)})') 
    fontsize=30
    if style!='':
        ax[0].scatter(refs, diffs, s = 50,facecolors=style, edgecolors=hue) 
        ax[1].scatter(refs, preds, s = 50,facecolors=style, edgecolors=hue) 
    else:
        ax[0].scatter(refs, diffs, s = 50, c=hue) 
        ax[1].scatter(refs, preds, s = 50, c=hue)   
    return {'nb_qts': len(refs), 'mae': round(np.mean(np.abs(diffs)), 1), 'mae_std': round(np.std(np.abs(diffs)), 1)}

def plot_time_profiles(data, timepoints, patient_ids, drugs, step = 1/20, fontsize=20): 
    d = {'manual': {'shift': -2*step, 'color': 'k', 'style':'_-', 'label': 'Manual'},
        'cnn': {'shift': -step, 'color': 'tab:orange', 'style':'_-', 'label': 'AttnCNN'},
        'res': {'shift': 0, 'color': 'tab:blue', 'style':'_-', 'label': 'KanResWide'},
        'unet': {'shift': step, 'color': 'y', 'style':'_-', 'label': 'U-Net'},
        'wvlt': {'shift': 2*step, 'color': 'purple', 'style':'_-', 'label': 'Wavelet'}} 
    for drug in drugs[:-1]:
        print(drug)
        metrics, _ = get_mean_CI(data, timepoints, patient_ids, drug, typeAnnot='manual')
        metrics_cnn, _ = get_mean_CI(data, timepoints, patient_ids, drug, typeAnnot='cnn')
        metrics_res, _ = get_mean_CI(data, timepoints, patient_ids, drug, typeAnnot='resnet')
        metrics_unet, _ = get_mean_CI(data, timepoints, patient_ids, drug, typeAnnot='unet')
        metrics_wvlt, _ = get_mean_CI(data, timepoints, patient_ids, drug, typeAnnot='wvlt')
        plt.figure(figsize = (14, 8), dpi=400) 
        plot_profile(metrics, 'manual', d, timepoints)
        plot_profile(metrics_cnn, 'cnn', d, timepoints)
        plot_profile(metrics_res, 'res', d, timepoints)
        plot_profile(metrics_unet, 'unet', d, timepoints)
        plot_profile(metrics_wvlt, 'wvlt', d, timepoints)
        plt.axhline(10, color='k', ls='--')
        plt.title(drug, fontsize=fontsize) 
        plt.ylabel('ΔΔ ± 95% CI (ms)', fontsize=fontsize)
        plt.xlabel('Time postdose (h)', fontsize=fontsize)
        plt.xticks(fontsize=fontsize); plt.yticks(fontsize=fontsize)
        plt.xlim(0, 8.5) 
        plt.grid()
        plt.savefig(f'data/outputs/tqt-analysis/time-profiles/{drug}.png', bbox_inches='tight')

def plot_profile(metrics_auto, auto, d, timepoints):
    shift, color, style, label =d[auto]['shift'], d[auto]['color'] , d[auto]['style'], d[auto]['label'] 
    tpts = [val+shift for val in timepoints[1:]]
    plt.plot(tpts, metrics_auto['DDQTc_mean'], "x", color=color, label =label) 
    for x, lower, upper in zip(tpts, metrics_auto['lower_DDQTc'], metrics_auto['upper_DDQTc']):
        plt.plot((x, x), (lower,upper),style, color=color)

def DDqtc_trend(data, timepoints, patient_ids, drug='Ranolazine', auto='resnet', show=True, fontsize=20):
    metrics, qt_prolng = get_mean_CI(data, timepoints, patient_ids, drug, typeAnnot='manual')
    metrics_auto, qt_prolng_auto = get_mean_CI(data, timepoints, patient_ids, drug, typeAnnot=auto) 
        
    if show:
        plt.figure(figsize = (15, 8), dpi=400)
        #Manual
        plt.plot(timepoints[1:], metrics['DDQTc_mean'], "-o", color='tab:blue', label ='manual') 
        for x, lower, upper in zip(timepoints[1:], metrics['lower_DDQTc'], metrics['upper_DDQTc']):
            plt.plot((x, x), (lower,upper),'_-', color='tab:blue')
        #Auto
        plt.plot(timepoints[1:], metrics_auto['DDQTc_mean'], "-x", color='tab:orange', label ='auto') 
        for x, lower, upper in zip(timepoints[1:], metrics_auto['lower_DDQTc'], metrics_auto['upper_DDQTc']):
            plt.plot((x, x), (lower,upper),'_-', color='tab:orange')
        plt.axhline(10, color='r', ls='--')
        plt.title(drug, fontsize=fontsize) 
        plt.ylabel('ΔΔ ± 95% CI (ms)', fontsize=fontsize)
        plt.xlabel('Time postdose (h)', fontsize=fontsize)
        plt.xticks(fontsize=fontsize); plt.yticks(fontsize=fontsize)
        #plt.legend(fontsize=fontsize) 
        plt.grid()
        plt.show()
    return metrics, metrics_auto, qt_prolng, qt_prolng_auto

def get_mean_CI(data, timepoints, patient_ids, drug, typeAnnot):
    DDQTcs = {tpt: [val-placebo for val, placebo in 
             zip(get_DQTc_at_tpt(tpt, drug, typeAnnot, data, patient_ids), get_DQTc_at_tpt(tpt, 'Placebo', typeAnnot, data, patient_ids))] 
             for tpt in timepoints[1:] }
    metrics = {'DDQTc_mean': [np.mean(item) for _, item in DDQTcs.items()],
               'lower_DDQTc': [conf_intervals(item)[0] for _, item in DDQTcs.items()],
               'upper_DDQTc': [conf_intervals(item)[1]  for _, item in DDQTcs.items()]
              }
    idx_max = np.argmax(metrics['DDQTc_mean'])
    qt_prolongation = {'max_DDQTc': round(metrics['DDQTc_mean'][idx_max],1),
                       'lower_DDQTc': round(metrics['lower_DDQTc'][idx_max], 1),
                       'upper_DDQTc': round(metrics['upper_DDQTc'][idx_max], 1)}
    return metrics, qt_prolongation

def get_DQTc_at_tpt(tpt, drug, typeAnnot, data, patient_ids): 
    qtcs_at_tpt = []
    for patient_id in patient_ids:
        baseline = get_QTc(data, typeAnnot, patient_id=patient_id, tpt=-0.5, drug=drug)
        qt = get_QTc(data, typeAnnot, patient_id=patient_id, tpt=tpt, drug=drug)
        if baseline and qt:
            qtcs_at_tpt.append(qt-baseline) 
    return qtcs_at_tpt

def get_QTc(data, typeAnnot, patient_id, tpt, drug):
    rows = data[(data.RANDID==patient_id) & (data.EXTRT==drug) & (data.TPT==tpt)]
    rrs = remove_nan(rows.RR)
    if 'manual' in typeAnnot:
        qts = remove_nan(rows.QT)
    if 'wvlt' in typeAnnot:
        qts = remove_nan(rows.QTwvlt)
    if 'cnn' in typeAnnot:
        qts = remove_nan(rows.QTcnn)
    if 'resnet' in typeAnnot:
        qts = remove_nan(rows.QTresnet)
    if 'unet' in typeAnnot:
        qts = remove_nan(rows.QTunet) 
    if len(qts)>0 and len(rrs)>0:
        num = np.mean(qts)
        denom = (np.mean(rrs)/1000)**(1/3)
        return num/denom
    else:
        return None 
    
def conf_intervals(x, confidence = 0.95):
    np.random.seed(13)
    values = [np.random.choice(x,size=len(x),replace=True).mean() for i in range(1000)] 
    results = np.percentile(values,[100*(1-confidence)/2, 100*(1-(1-confidence)/2)]) 
    return results

def remove_nan(lst):
    lst_wo_nan = [val for val in lst if str(val)!='nan']
    return lst_wo_nan
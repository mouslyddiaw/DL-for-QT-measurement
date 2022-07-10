import xml.etree.ElementTree as ET 
from pathlib import Path
import numpy as np 
import math
import csv
import os 
import XMLparser.utils as ut 

ignored_files = ['BR-UPS1001-2020_104_Admission_20201104171238_2',
                'BR-UPS1001-2020_104_Admission_20201104171301_3',
                'BR-UPS1001-2020_104_Admission_20201104174151_1',
                'BR-UPS1001-2020_104_Admission_20201104181736_1',
                'BR-UPS1001-2020_104_Admission_20201104191912_3',
                'BR-UPS1001-2020_104_Admission_20201104201413_3',
                'BR-UPS1001-2020_104_Admission_20201104212140_1',
                'BR-UPS1001-2020_104_Admission_20201104212203_2',
                'BR-UPS1001-2020_104_Day7_20201111212826_3',
                'BR-UPS1001-2020_104_Day7_20201112170718_1',
                'BR-UPS1001-2020_105_Admission_20201105172529_1',
                'BR-UPS1001-2020_105_Admission_20201105172716_2',
                'BR-UPS1001-2020_105_Admission_20201105172916_3',
                'BR-UPS1001-2020_102_Admission_20201105164247_1',
                'BR-UPS1001-2020_102_Admission_20201105164612_2',
                'BR-UPS1001-2020_102_Admission_20201105165154_3',
                'BR-UPS1001-2020_102_Day1_20201105172709_3',
                'BR-UPS1001-2020_102_Day1_20201105173623_1',
                'BR-UPS1001-2020_102_Day1_20201105173911_2',
                'BR-UPS1001-2020_102_Day1_20201105174025_3',
                'BR-UPS1001-2020_102_Day1_20201105175648_1',
                'IP-07_21-0076L__20121102222039_1',
                'IP-07_21-0130L__20121129135955_1',
                'IP-07_21-0137L__20121205191607_1', 
                'IP-07_21-0139L__20121207141510_1' ,
                'IP-07_21-0141L__20121206112736_1',
                'IP-07_21-0145L__20121212161347_1',
                'IP-07_21-0146L__20121212162608_1',
                'IP-07_21-0147L__20121216193259_1',
                'IP-07_21-0148L__20121215104911_1',
                'IP-07_21-0153L__20121226150441_1',
                'IP-07_21-0154L__20121229143902_1',
                'IP-07_21-0161L__20130115112633_1', 
                'IP-07_21-0162L__20130115125022_1',
                'IP-07_21-0163L__20130121105859_1',
                'IP-07_21-0164L__20130123121738_1',
                'IP-07_21-0164L__20130125145236_1']

def convert_to_ms(position, sampling_rate):
    '''Converts duration from sample units to msec'''
    return 1000*position/sampling_rate

pathlist = Path('XML_BRUPS_IP').glob('**/*.xml')


directory='final_data/'
header_labels = ['id', 'filename', 'qpos', 'tpos']
with open("final_data/labels.csv", "w", newline='') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(header_labels) 
    
compt = 0
for path in pathlist:
    path_in_str = str(path)
    if any(ignored_file in path_in_str for ignored_file in ignored_files):
        pass
    else: 
        root = ET.parse(path_in_str).getroot()
        filename = os.path.split(path_in_str)[1].split('.')[0] 
        sampling_rate = ut.find_sampling_rate(root)
        annotations = ut.find_annotations(root, prefix="{urn:hl7-org:v3}") 
        ecg_leads = ut.find_leads(root)
        try:
            reference_q = annotations['Q_Onset']
            reference_t = annotations['T_Offset'] 
        except KeyError:
            pass
        else:
            compt+=1 
            file_id = 'id-' + str(compt)

            with open("final_data/labels.csv", "a", newline='') as f:
                writer = csv.writer(f, delimiter=',')
                l = [file_id, filename, reference_q, reference_t]
                writer.writerow(l) 
             
            list_leads = [ecg_leads[lead] for lead in ecg_leads.keys()]
        
            with open('final_data/train/' + file_id + '.csv', 'w', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(ecg_leads.keys())
                writer.writerows(map(lambda x: x, list(zip(*list_leads))))
           
            if compt%100==0:
                print('Iteration',compt)




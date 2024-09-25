from csv import DictReader
import json
import numpy as np
from matplotlib import patches as patches
import sys
import re
import csv
import datetime as datetime
import pickle

path = '/home/mbauer/Data/SSW_Data/'

with open(path+"solar-stormwatch-ii-classifications.csv", 'r') as f:
    dict_reader = DictReader(f)
    classifications = list(dict_reader)

f.close()

date_dict = {}
n = 0

for entry in classifications:

    for num_ann, ann in enumerate(json.loads(entry["annotations"])):
        subj_data = json.loads(entry["subject_data"])
        subj_data = subj_data[list(subj_data.keys())[0]]

        if(ann['value']!='Yes'):

            img_times = []
            img_sc = []
            img_annotator = []
            img_class_id = []
            subj_id = []

            for k in list(subj_data.keys()):
                
                if re.search('asset', k):
                    if re.search('sta', subj_data[k]):
                        if re.search('\AHCME', subj_data[k]):

                            temp_time = subj_data[k].split('_')[7:9]
                            temp_time = '_'.join(temp_time).split('.')[0]

                            img_times.append(temp_time)
                            img_sc.append(subj_data[k].split('_')[5])

                        elif re.search('\Assw', subj_data[k]):
                            
                            if re.search('swpc', subj_data[k]):
                                temp_time = subj_data[k].split('_')[6:8]
                                temp_time = '_'.join(temp_time).split('.')[0]

                                img_times.append(temp_time)
                                img_sc.append(subj_data[k].split('_')[4])
                            
                            else:
                                temp_time = subj_data[k].split('_')[4:6]
                                temp_time = '_'.join(temp_time).split('.')[0]

                                img_times.append(temp_time)
                                img_sc.append(subj_data[k].split('_')[2])

                        else:
                            print('not defined for ' + str(subj_data[k]))
                            sys.exit()

                        # print(subj_data[k])
                        img_annotator.append(entry["user_name"])
                        img_class_id.append(entry["classification_id"])
                        subj_id.append(entry['subject_ids'])

            polys = [[] for i in range(0, 3)]

            for val_num, val in enumerate(ann['value']):
                pts = val["points"]
                ind = val["frame"]
                polygon = []

                for p in pts:
                    polygon.append([p["x"],p["y"]])
                
                polygon = np.array(polygon).astype(np.int32)

                polys[ind].append(polygon)


            for i in range(0,len(img_times)):

                if not len(img_times[i]) == 0:

                    if img_times[i] in date_dict.keys():

                        date_dict[img_times[i]]['classification_id'].append(img_class_id[i])
                        date_dict[img_times[i]]['subject_id'].append(subj_id[i])
                        date_dict[img_times[i]]['sc'].append(img_sc[i])
                        date_dict[img_times[i]]['user_name'].append(img_annotator[i])
                        date_dict[img_times[i]]['masks'].append(polys[i])

                    else:
                        
                        date_dict[img_times[i]] = {"classification_id":[], "subject_id":[], "sc": [], "user_name": [], "masks": []}

                        date_dict[img_times[i]]['classification_id'].append(img_class_id[i])
                        date_dict[img_times[i]]['subject_id'].append(subj_id[i])
                        date_dict[img_times[i]]['sc'].append(img_sc[i])
                        date_dict[img_times[i]]['user_name'].append(img_annotator[i])
                        date_dict[img_times[i]]['masks'].append(polys[i])

    n = n+1
    print('Processed {} out of {}'.format(n, len(classifications)))
          
with open(path+'ssw_classification.pkl', 'wb') as f:
    pickle.dump(date_dict, f)
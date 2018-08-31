# coding: utf-8
# this will download audioset youtube samples
# 
# balanced
# unbalanced
# eval

import numpy as np
import pandas as pd
import re
import csv
import datetime
import os
import wave


## Lees csv-file in
labels = pd.read_csv('csv_files/unbalanced_train_segments.csv', skiprows=3, 
                          quotechar='"', skipinitialspace = True, header=None, 
                          names = ["YTID", "start_seconds", "end_seconds", "positive_labels"])


# bal_labels = pd.read_csv('csv_files/balanced_train_segments.csv', skiprows=3, 
#                          quotechar='"', skipinitialspace = True, header=None, 
#                          names = ["YTID", "start_seconds", "end_seconds", "positive_labels"])

# eval_labels = pd.read_csv('csv_files/eval_segments.csv', skiprows=3, 
#                          quotechar='"', skipinitialspace = True, header=None, 
#                          names = ["YTID", "start_seconds", "end_seconds", "positive_labels"])


# In[5]:


## Hierin worden de wav-files gegooid
file_path = "wav_files"


# In[4]:


# ## Maak folder aan als deze nog niet bestaat
# if not os.path.isdir(file_path):
#     os.makedirs(file_path)
    
# ## Maak ook folders aan voor bal/unbal/eval (om later makkelijk labels weer op te halen)
# if not os.path.isdir(file_path + "/bal"):
#     os.makedirs(file_path + "/bal")
# if not os.path.isdir(file_path + "/unbal"):
#     os.makedirs(file_path + "/unbal")
# if not os.path.isdir(file_path + "/eval"):
#     os.makedirs(file_path + "/eval")


# In[6]:


## Geef aan of de tf-records voor jungle of urban moeten worden gedownload (ik neem aan dat we alleen jungle doen)
target = "jungle" #"urban"
#mid_to_label = pd.read_csv("csv_files/class_labels_indices_" + target + ".csv", sep=";")
mid_to_label = pd.read_csv("csv_files/class_labels_indices_jungle.csv", sep=";")


# In[6]:


mid_to_label


# In[7]:


def getLabels(mid_str):
    ## Maak lijst van m-id's
    mid_list = mid_str.split(',')
    labels = []
    
    ## Voor elk m-id, vind labels, (if any labels: add to label list)
    for mid in mid_list:
        if (mid_to_label.loc[mid_to_label["mid"] == mid, "mid"].any()):
            labels.append(mid_to_label.loc[mid_to_label["mid"] == mid, "index"].values[0])
    
    ## Return unique set of labels
    return set(labels)


# In[8]:


# add progressbar
from tqdm import tqdm

## Download wav-files van youtube
def downloadWav(vid_to_mid, folder):
#    for i in tqdm(range(vid_to_mid.shape[0])):
    i = 0 # altijd deeerste
    mids = vid_to_mid["positive_labels"]
    labels = getLabels(mids)

    if(len(labels)>0):
        url = vid_to_mid["YTID"]
        start_sec = vid_to_mid["start_seconds"]
        start_time = str(datetime.timedelta(seconds=start_sec)) + '.00'

        ## titel = [url] + '_^_' + [starttime]
        file_path = 'wav_files/' + folder + "vid" + url #+ '_^_' + str(start_sec) + '00'
#        print(file_path)
#        cmd = 'youtube-dl -f "bestaudio" -o "' + file_path + '.%(ext)s" --extract-audio --audio-format wav --postprocessor-args "-ss ' + start_time + ' -t 00:00:10.00" "https://www.youtube.com/v/' + url + '"'
#        cmd = 'youtube-dl -f "bestaudio" -o "' + file_path + '.%(ext)s" --extract-audio  --postprocessor-args "-ss ' + start_time + ' -t 00:00:10.00 -acodec pcm_u8 -ar 16k" "https://www.youtube.com/v/' + url + '"'
        cmd = 'youtube-dl -f "bestaudio" -o "' + file_path + '.%(ext)s" --extract-audio  --postprocessor-args "-ss ' + start_time + ' -t 00:00:10.00" "https://www.youtube.com/v/' + url + '"'
        #print("CMD="+cmd)
        os.system(cmd)

#         # check file (length)
#         f = wave.open(file_path + '.wav', 'r')
#         frames = f.getnframes()
#         rate = f.getframerate()
#         duration = frames / float(rate)
#         f.close()

#         if(duration < 10):
#             print("Warning - duration of video '" + url + "' is only " + str(duration) + " second(s)!")


# In[9]:


# TODO (optional) - add code to clear folder(s) before a batch run
#folder = "bal/"
#folder = "eval/"
folder = "unbal/"


# In[ ]:


from joblib import Parallel, delayed
import multiprocessing


num_cores = multiprocessing.cpu_count()

# parralel
#results = Parallel(n_jobs=num_cores)(delayed(downloadWav)(eval_labels.loc[i], folder) for i in tqdm(range(eval_labels.shape[0])))
#results = Parallel(n_jobs=num_cores)(delayed(downloadWav)(bal_labels.loc[i], folder) for i in tqdm(range(bal_labels.shape[0])))
results = Parallel(n_jobs=num_cores)(delayed(downloadWav)(unbal_labels.loc[i], folder) for i in tqdm(range(unbal_labels.shape[0])))

#for i in tqdm(range(eval_labels.shape[0])):
#    downloadWav(eval_labels.loc[i], folder)


# In[10]:



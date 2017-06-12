# -*- coding: utf-8 -*-

import sys, csv, os, shutil

print("start... \n")

AUDIO_LIST_FILE = "dataset.csv"
src = "smb://pcloud/hugo/git/ESC-50/audio"
dst = "audio"

wav_list = []

with open(AUDIO_LIST_FILE, 'r') as f:
    reader = csv.reader(f)
    next(reader)
    try:
        for row in reader:
            wav_list.append(row[0])
    except csv.Error as e:
        sys.exit('file %s, line %d: %s' % (AUDIO_LIST_FILE, reader.line_num, e))


for w in wav_list:
    # lees wav check of die er is zo ja copier naar desitination
    src_file_path = os.path.normcase("%s/%s" % (src,w))

    dst_file_path = "%s\%s" % (dst,w)

    shutil.copyfile(src_file_path,dst_file_path)

print("done!\n")

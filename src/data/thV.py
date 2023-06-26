import numpy as np
import csv
import os
import pandas as pd

f = open("results_TM.csv","r")
reader = csv.reader(f)
ccnt = 0
gcnt = 0
for row in reader:
    if row[0] == 'label':
        continue
    if row[3] == '0':
        cstr = row[0] + '_4.png'
        gstr = row[0] + '_2.jpg'
        cfd = 'data1_TM2_Threshold/Color/' + cstr
        gfd = 'data1_TM2_Threshold/Gray/' + gstr
        if os.path.isfile(gfd):
            os.remove(gfd)
            gcnt += 1
        if os.path.isfile(cfd):
            os.remove(cfd)
            ccnt += 1
print(ccnt,gcnt)
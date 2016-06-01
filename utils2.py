#!/usr/bin/python
import sys, cPickle as pickle, time
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

import time
import cPickle as pickle
import gzip
import os.path
import gc

import code
import sys
# from http://vjethava.blogspot.de/2010/11/matlabs-keyboard-command-in-python.html
def keyboard(banner=None):
    ''' Function that mimics the matlab keyboard command '''
    # use exception trick to pick up the current frame
    try:
        raise None
    except:
        frame = sys.exc_info()[2].tb_frame.f_back
    print "# Use quit() to exit :) Happy debugging!"
    # evaluate commands in current namespace
    namespace = frame.f_globals.copy()
    namespace.update(frame.f_locals)
    try:
        code.interact(banner=banner, local=namespace)
    except SystemExit:
        return 
        
def progress_printer(current_number, total_number):
    """
    This function does nothing but displaying a fancy progress bar :)
    """
    
    global anim_state, pp_last_print_time
    if not 'anim_state' in globals():
        anim_state = 0
    
    # Chose if we have to print again?
    if 'pp_last_print_time' not in globals():
        printme = True
    elif time.time() - pp_last_print_time > 0.1:
        printme = True
    else:
        printme = False

    # What to print
    if printme:
        anim=["[*     ]","[ *    ]","[  *   ]","[   *  ]","[    * ]","[     *]","[    * ]","[   *  ]","[  *   ]","[ *    ]"]
        if total_number != None and total_number != 0:
            progress = str(int((float(current_number)/total_number)*100)) + "%"
        else:
            progress = " *working hard* (" + str("{:,}".format(current_number)) + " elements processed)"
        print "\r" + anim[anim_state] + " " + progress,
        anim_state = (anim_state + 1) % len(anim)
        pp_last_print_time = time.time()
        sys.stdout.flush()

## Functions for loading and preparing data
# load data
def load_data(filename,interval=60,dropnan=True):
    print "loading %s"%filename,
    sys.stdout.flush()
    df = pd.read_csv(filename)
    df = convert_timestamp(df)
    if interval>0:
        df = resample(df,interval)
    else:
        df = df.set_index('ut_ms')
    if dropnan:
        df = df.dropna()
    print "done"
    return df

# convert timestamp 
def convert_timestamp(df,index='ut_ms'):
    df[index] = pd.to_datetime(df[index], unit='ms')
    return df

# resample data to arbitrary interval in minutes
def resample(df, interval=60):
    df = df.set_index('ut_ms')
    df = df.resample('%dT'%interval).mean()
    return df


def marsexpress_error(predictions,targets):
    diff = (targets - predictions) ** 2
    error = np.mean(diff.values) ** 0.5
    return error


# prepares data by reformatting columns, resampling etc
def prepare_data_ftl(data,index):
    colList = ['flagcomms',"ACROSS_TRACK","D1PVMC","D2PLND","D3POCM","D4PNPO","D5PPHB",\
    "D7PLTS","D8PLTP","D9PSPO","EARTH","INERTIAL","MAINTENANCE",\
    "NADIR","RADIO_SCIENCE","SLEW","SPECULAR","SPOT","WARMUP","NADIR_LANDER",]
        
    df = pd.DataFrame(index=index)
    
    for col in colList:
        df[col] = 0
    
    # format times in data
    data['utb_ms'] = pd.to_datetime(data['utb_ms'], unit='ms')
    data['ute_ms'] = pd.to_datetime(data['ute_ms'], unit='ms')
    for i in range(len(data)):
        if ((i+1) % 100) == 0:
            progress_printer(i,len(data))
        #if i<10:
        #    keyboard()
        begin = df.index > data.utb_ms[i]
        end = df.index < data.ute_ms[i]
        thisindex = df.index.searchsorted(data.utb_ms[i])-1
        
        df[data.type[i]][thisindex] = 1
        df[data.type[i]][begin * end] = 1

        if data.flagcomms[i]:
            df.flagcomms[thisindex] = 1
        
        if data.flagcomms[i]:
            df.flagcomms[begin * end] = 1
        """
        print "---",i
        print data.type[i]
        print data.utb_ms[i]
        print data.ute_ms[i]
        print df.index[thisindex],df[data.type[i]][thisindex]
        print df[data.type[i]][begin * end]
        print "flagcomms:",data.flagcomms[i]
        print df.index[thisindex],df.flagcomms[thisindex]
        print df.flagcomms[begin * end]
        """

    return df

def prepare_data_evtf(data):
    umbras = pd.get_dummies(data.loc[data.description.str.contains("^MAR(.)*UMBRA", regex=True),:])
    df = pd.DataFrame(index=data.index)
    df = df.join(umbras).fillna(0)
    return df




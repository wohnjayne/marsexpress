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
def convert_timestamp(df):
    df['ut_ms'] = pd.to_datetime(df['ut_ms'], unit='ms')
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

# from http://vjethava.blogspot.de/2010/11/matlabs-keyboard-command-in-python.html
import code
import sys
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

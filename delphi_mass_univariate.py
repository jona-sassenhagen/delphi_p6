import numpy as np

import seaborn as sns
sns.set_style("white")
sns.set_context("notebook")
sns.set_style("ticks")

import matplotlib.pyplot as plt

import mne
mne.set_log_level(False)
from os import listdir

mfile = 'standard.elp'

import pandas as pd

import warnings
warnings.filterwarnings('ignore')

sourceloc = "../for_mne/"
fnames = [sourceloc + x for x in listdir(sourceloc) if ".set" in x]

event_id = {"control":100, "syntax":200}

def get_raw(fname, lowpass=.1, highpass=20):
    raw = mne.io.read_raw_eeglab(
        fname, event_id=event_id, preload=True, verbose=False, montage=mfile).filter(
        lowpass, highpass, n_jobs='cuda', filter_length='auto',
        l_trans_bandwidth='auto', h_trans_bandwidth='auto')
    mne.io.set_eeg_reference(raw, ["FT10", "FT9"], copy=False)
    return raw

def get_epochs(fname, lowpass=.1, highpass=20, tmin=-.5, tmax=1.5):
    raw = get_raw(fname, lowpass, highpass)
    events = mne.find_events(raw)
    events[:, 0] += 30  # fix triggers
    return mne.Epochs(raw, events, event_id, tmin, tmax)

def get_evokeds(fname, lowpass=.1, highpass=20, tmin=-.5, tmax=1.5):
    epochs = get_epochs(fname, lowpass, highpass, tmin, tmax)
    return {cond:epochs[cond].average() for cond in event_id}


# tbd

import os
import cv2
import shutil
from time import gmtime, strftime
import numpy as np
from collections import OrderedDict
import logging
import nrrd
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pylab as plt
from PIL import Image
from tqdm import tnrange, trange
import seaborn as sns
import pandas as pd

logging.basicConfig(filename="logging_info_"+strftime("%Y-%m-%d %H:%M:%S", gmtime())+".log",level=logging.DEBUG, format='%(asctime)s %(message)s')

def update_mpl_settings():
    #Direct input
    plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath, lmodern}"]
    #Options
    fontsize = 22
    params = {'text.usetex' : True,
              'font.family' : 'lmodern',
              'text.latex.unicode': True,
              'text.color':'black',
              'xtick.labelsize': fontsize-2,
              'ytick.labelsize': fontsize-2,
              'axes.labelsize': fontsize,
              'axes.labelweight': 'bold',
              'axes.edgecolor': 'white',
              'axes.titlesize': fontsize,
              'axes.titleweight': 'bold',
              'pdf.fonttype' : 42,
              'ps.fonttype' : 42,
              'axes.grid':False,
              'axes.facecolor':'white',
              'lines.linewidth': 1,
              "figure.figsize": '5,4',
              }
    plt.rcParams.update(params)

update_mpl_settings()

pgf_with_pdflatex = {
    "pgf.texsystem": "lualatex",
    "pgf.preamble": [
        r'\usepackage{amsmath,lmodern}',
        r'\usepackage[scientific-notation=true]{siunitx}',
        ]
}

mpl.rcParams.update(pgf_with_pdflatex)
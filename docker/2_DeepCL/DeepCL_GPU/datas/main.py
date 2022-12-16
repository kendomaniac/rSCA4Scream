import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from keras_dec import DeepEmbeddingClustering
from keras.datasets import mnist
import numpy as np
import pandas as pd
import sklearn
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from keras.datasets import mnist
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
from random import sample 
from keras.models import Sequential
from keras.layers import Dense, Activation
import keras
from sklearn.model_selection import train_test_split
from keras.utils import plot_model
from matplotlib import pyplot
import os
import scipy.io
import pandas as pd
from sklearn.manifold import TSNE
import sys
from sklearn.cluster import KMeans
import subprocess
import tensorflow as tsf
import keras
import datetime
from keras.callbacks import EarlyStopping
from shutil import copyfile
import matplotlib.cm as cm    
import stat

os.chdir("/scratch")

if np.asarray(sys.argv).shape[0] ==1:
    bias="TF"
    finetune_iters=5
    nEpochs=10
    projectName="prova"
    matrix="setA2.csv"
    sep=","
    nCluster=5
    bName="cancer_related_immuno_signatures.csv"
else:
    matrix=sys.argv[1]
    sep=sys.argv[2]
    nCluster=int(sys.argv[3])
    bias=sys.argv[4]
    finetune_iters=float(sys.argv[5])
    nEpochs=float(sys.argv[6])
    projectName=sys.argv[7]
    random.seed(sys.argv[8])
    bName=sys.argv[9]
print("matrixName "+matrix)
print("sep "+sep)
print("nCluster "+str(nCluster))
print("bias "+bias)
print("finetune_iters "+str(finetune_iters))
print("nEpochs "+str(nEpochs))
print("projectName "+projectName)
print("bName "+bName)

try:
    os.mkdir("/scratch/Results/")
except OSError:
    print ("Creation of the directory %s failed" % "/scratch/Results/")
else:
    print ("Successfully created the directory %s " % "/scratch/Results/")
    
try:
    os.mkdir("/scratch/Results/"+projectName)
except OSError:
    print ("Creation of the directory %s failed" % "/scratch/Results/"+projectName)
    now = datetime.datetime.now()
    projectName=projectName+str(now.year)+"_"+str(now.month)+"_"+str(now.day)+"_"+str(now.hour)+"_"+str(now.minute)+"_"+str(now.second)
    os.mkdir("/scratch/Results/"+projectName)
else:
    print ("Successfully created the directory %s " % "/scratch/Results/"+projectName)
copyfile("/scratch/"+matrix,"/scratch/Results/"+projectName+"/"+matrix)
path="/scratch/Results/"+projectName+"/"




matrix="/scratch/"+matrix
def get_mnist():
    mat=pd.read_csv(matrix,index_col=0,sep=sep)
    mat=mat.drop(mat.index[np.where(mat.T.sum()<=10)])
    Atac=mat.T 
    return Atac, Atac
X, Y  = get_mnist()
c = DeepEmbeddingClustering(n_clusters=nCluster, input_dim=X.shape[1],Atac=X,bias=bias,bName=bName,sep=sep,path=path)
c.initialize(X, finetune_iters=finetune_iters, layerwise_pretrain_iters=nEpochs)
c.cluster(X, y=Y)
os.system("chmod -R 777 /scratch/")

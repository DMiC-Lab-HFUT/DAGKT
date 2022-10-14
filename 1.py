import numpy as np
import tensorflow as tf
import pandas as pd
import os

path = 'C:/Users/wsco38/Desktop/Study/AKT/GIKT-alldiff_GAT_tf/data/junyi_3'
train = pd.read_csv(path + '/a.csv',header=None)
index = int(train.shape[0]/5*4)-2
train_seqs = train[0:index][:]
test_seqs = train[index:train.shape[0]][:]
train_seqs.to_csv(path+'/junyi_3_train.csv')
test_seqs.to_csv(path+'/junyi_3_test.csv')
print('OK')

# with tf.Session() as sess:








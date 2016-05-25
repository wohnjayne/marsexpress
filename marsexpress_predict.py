# parts of this script were inspired by Alexander Bauers baseline skript at
#  https://github.com/alex-bauer/kelvin-power-challenge/blob/master/src/rf_baseline.py
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from utils2 import *
import time
import os.path
import gc
import sys

interval = 60 #minutes

path = {}
path = "/home/malte/data/mars-express-power-3years/"
data_sources = ["saaf","ltdata"]
do_validation = True
do_test = True
##load data 
# targets
targets_pickle_name="%s/prepro/targets-interval%d.pklz"%(path,interval)
if  os.path.isfile(targets_pickle_name):
    print "loading pickled targets"
    f = gzip.open(targets_pickle_name,'rb')
    targets = pickle.load(f)
    f.close()
else:
    print "loading targets"
    targets_train1 = load_data('%s/train_set/power--2008-08-22_2010-07-10.csv'%path,interval=interval)
    targets_train2 = load_data('%s/train_set/power--2010-07-10_2012-05-27.csv'%path,interval=interval)
    targets_train3 = load_data('%s/train_set/power--2012-05-27_2014-04-14.csv'%path,interval=interval)
    targets_test = load_data('%s/power-prediction-sample-2014-04-14_2016-03-01.csv'%path,dropnan=False)
    targets = pd.concat([targets_train1, targets_train2, targets_train3,targets_test])
    f = gzip.open(targets_pickle_name,'wb')
    pickle.dump(targets,f)
    f.close()

data_pickle_name="%s/prepro/data"%path
for ds in data_sources:
    data_pickle_name+="-"+ds
data_pickle_name += "-interval%d.pklz"%(interval)

if  os.path.isfile(data_pickle_name):
    print "loading pickled data"
    f = gzip.open(data_pickle_name,'rb')
    data = pickle.load(f)
    f.close()
else:
    # input data, saaf
    saaf = pd.DataFrame()
    if 'saaf' in data_sources:
        saaf_train1 = load_data('%s/train_set/context--2008-08-22_2010-07-10--saaf.csv'%path,interval=interval)
        saaf_train2 = load_data('%s/train_set/context--2010-07-10_2012-05-27--saaf.csv'%path,interval=interval)
        saaf_train3 = load_data('%s/train_set/context--2012-05-27_2014-04-14--saaf.csv'%path,interval=interval)
        saaf_test = load_data('%s/test_set/context--2014-04-14_2016-03-01--saaf.csv'%path,dropnan=False).interpolate()
        saaf = pd.concat([saaf_train1, saaf_train2, saaf_train3,saaf_test])
        #saaf = saaf.fillna(method='pad')

    # Load the ltdata files
    ltdata = pd.DataFrame()
    if 'ltdata' in data_sources:
        ltdata_train1 = load_data('%s/train_set/context--2008-08-22_2010-07-10--ltdata.csv'%path,interval=0)
        ltdata_train2 = load_data('%s/train_set/context--2010-07-10_2012-05-27--ltdata.csv'%path,interval=0)
        ltdata_train3 = load_data('%s/train_set/context--2012-05-27_2014-04-14--ltdata.csv'%path,interval=0)
        ltdata_test = load_data('%s/test_set/context--2014-04-14_2016-03-01--ltdata.csv'%path,interval=0,dropnan=False)
        ltdata = pd.concat([ltdata_train1, ltdata_train2, ltdata_train3, ltdata_test])
        
    saaf = saaf.reindex(targets.index, method='nearest')
    ltdata = ltdata.reindex(targets.index, method='nearest')

    # create data 
    data = saaf.join(ltdata)
   
    f = gzip.open(data_pickle_name,'wb')
    pickle.dump(data,f)
    f.close()
    del saaf,ltdata
    
target_cols = list(targets.columns)

# split training from test data
is_training_data = ~targets[target_cols[0]].isnull()
data_train,targets_train = data[is_training_data], targets[is_training_data]
data_test,targets_test = data[~is_training_data], targets[~is_training_data]

# split validation data
cv_split = data_train.index < '2012-05-27'
data_train_cv, targets_train_cv = data_train[cv_split], targets_train[cv_split]
data_validation_cv, targets_validation_cv = data_train[~cv_split], targets_train[~cv_split]

from sklearn.ensemble import RandomForestRegressor

def fitAndPredictRandomForest(data,targets,data_for_prediction,n_estimators=200, min_samples_leaf=300):
    rf = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, n_jobs=-1)
    rf.fit(data, targets)
    return rf.predict(data_for_prediction),rf
if do_validation:
    print "fitting model for validation set"
    [predictions_validation_cv, model]= fitAndPredictRandomForest(data_train_cv,targets_train_cv,data_validation_cv)
    error = marsexpress_error(predictions_validation_cv,targets_validation_cv)
    print "Validation error: {} ".format(error)

    print "Frature importance:"
    for feature, importance in sorted(zip(model.feature_importances_, data_train.columns), key=lambda x: x[0], reverse=True):
        print feature, importance

if do_test:
    print "fitting model for test set"
    [predictions_test, model]= fitAndPredictRandomForest(data_train,targets_train,data_test)

    #Converting the prediction matrix to a dataframe
    predictions_test=pd.DataFrame(predictions_test, index=data_test.index, columns=target_cols)
    # We need to convert the parsed datetime back to utc timestamp
    predictions_test['ut_ms'] = (predictions_test.index.astype(np.int64) * 1e-6).astype(int)
    # Writing the submission file as csv
    predictions_test[['ut_ms'] + target_cols].to_csv('/tmp/rf_baseline.csv', index=False)

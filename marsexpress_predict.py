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
data_sources = ["saaf","ltdata","ftl","dmop","evtf"]
#data_sources = ["evtf"]
do_validation = True
do_parameter_testing = False
do_test = False
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
        
        saaf = saaf.reindex(targets.index, method='nearest')

    # Load the ltdata files
    ltdata = pd.DataFrame()
    if 'ltdata' in data_sources:
        ltdata_train1 = load_data('%s/train_set/context--2008-08-22_2010-07-10--ltdata.csv'%path,interval=0)
        ltdata_train2 = load_data('%s/train_set/context--2010-07-10_2012-05-27--ltdata.csv'%path,interval=0)
        ltdata_train3 = load_data('%s/train_set/context--2012-05-27_2014-04-14--ltdata.csv'%path,interval=0)
        ltdata_test = load_data('%s/test_set/context--2014-04-14_2016-03-01--ltdata.csv'%path,interval=0,dropnan=False)
        ltdata = pd.concat([ltdata_train1, ltdata_train2, ltdata_train3, ltdata_test])
        ltdata["OneMinusSunmarsearthangle_deg"] = 1-ltdata["sunmarsearthangle_deg"]
        ltdata = ltdata.reindex(targets.index, method='nearest')
     
    # Load the ftl files
    ftl = pd.DataFrame()
    if 'ftl' in data_sources:
        print "loading ftl"
        ftl_train1 = pd.read_csv('%s/train_set/context--2008-08-22_2010-07-10--ftl.csv'%path)
        ftl_train2 = pd.read_csv('%s/train_set/context--2010-07-10_2012-05-27--ftl.csv'%path)
        ftl_train3 = pd.read_csv('%s/train_set/context--2012-05-27_2014-04-14--ftl.csv'%path)
        ftl_test = pd.read_csv('%s/test_set/context--2014-04-14_2016-03-01--ftl.csv'%path)
        ftl = pd.concat([ftl_train1, ftl_train2, ftl_train3, ftl_test],ignore_index=True)
        ftl = prepare_data_ftl(ftl,targets.index)
        ftl = ftl.reindex(targets.index,method='nearest')
        
    # Load the dmop files
    dmop = pd.DataFrame()
    if 'dmop' in data_sources:
        dmop_train1 = load_data('%s/train_set/context--2008-08-22_2010-07-10--dmop.csv'%path,interval=0)
        dmop_train2 = load_data('%s/train_set/context--2010-07-10_2012-05-27--dmop.csv'%path,interval=0)
        dmop_train3 = load_data('%s/train_set/context--2012-05-27_2014-04-14--dmop.csv'%path,interval=0)
        dmop_test = load_data('%s/test_set/context--2014-04-14_2016-03-01--dmop.csv'%path,dropnan=False,interval=0)
        dmop = pd.concat([dmop_train1, dmop_train2, dmop_train3, dmop_test])
        dmop.subsystem = dmop.subsystem.apply(lambda d: str.split(d,".")[0])
        dmop = dmop.join(pd.get_dummies(dmop))
        dmop = dmop.drop(["subsystem"], axis=1)
        dmop = dmop.reindex(targets.index,method='nearest')
        
    # Load the evtf files
    evtf = pd.DataFrame()
    if 'evtf' in data_sources:
        evtf_train1 = load_data('%s/train_set/context--2008-08-22_2010-07-10--evtf.csv'%path,interval=0)
        evtf_train2 = load_data('%s/train_set/context--2010-07-10_2012-05-27--evtf.csv'%path,interval=0)
        evtf_train3 = load_data('%s/train_set/context--2012-05-27_2014-04-14--evtf.csv'%path,interval=0)
        evtf_test = load_data('%s/test_set/context--2014-04-14_2016-03-01--evtf.csv'%path,dropnan=False,interval=0)
        evtf = pd.concat([evtf_train1, evtf_train2, evtf_train3, evtf_test])
        evtf = prepare_data_evtf(evtf)
        evtf = evtf.resample('%dT'%interval).max().fillna(0)
        #keyboard()
        
        


    # create data 
    data = pd.DataFrame(index=targets.index)
    data = data.join(saaf)
    data = data.join(ltdata)
    data = data.join(ftl)
    data = data.join(dmop)
    data = data.join(evtf)
    f = gzip.open(data_pickle_name,'wb')
    pickle.dump(data,f)
    f.close()
    del saaf,ltdata

#print data.head()
#keyboard()
target_cols = list(targets.columns)

# split training from test data
is_training_data = ~targets[target_cols[0]].isnull()
data_train,targets_train = data[is_training_data], targets[is_training_data]
data_test,targets_test = data[~is_training_data], targets[~is_training_data]

print "-- checking for null values---"
print "data - train ",data_train.isnull().values.any()
print "data - test ",data_test.isnull().values.any()
print "targets - train ",targets_train.isnull().values.any()
print "targets - test ",targets_test.isnull().values.any(), "<-OK here, not with others"
#exit(1)
# split validation data
cv_split = data_train.index < '2012-05-27'
data_train_cv, targets_train_cv = data_train[cv_split], targets_train[cv_split]
data_validation_cv, targets_validation_cv = data_train[~cv_split], targets_train[~cv_split]

from sklearn.ensemble import RandomForestRegressor

def fitAndPredictRandomForest(data,targets,data_for_prediction,n_estimators=100, min_samples_leaf=5):
    rf = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, n_jobs=-1)
    rf.fit(data, targets)
    return rf.predict(data_for_prediction),rf

def plot_predictions(predictions,targets,filename):
    sum_targets = pd.Series(targets.sum(axis=1), index=targets.index, name='real power consumption')
    sum_predictions = pd.Series(predictions.sum(axis=1),index=targets.index, name="predicted power consumption")
    pd.concat([sum_targets,sum_predictions], axis=1).plot(figsize=(20,5))
    plt.savefig(filename)

def plot_differences(predictions,targets):
    import glob
    for filename in glob.glob("./NPWD*jpg"):
        os.remove(filename)
    errors = {}
    for field in predictions.columns:
        errors[field] = marsexpress_error(predictions[field],targets[field])
    
    cols = sorted(predictions.columns, key=lambda x: errors[x], reverse=True)
    how_many = 6
    plt.close("all")
    x_pos = np.arange(how_many)
    largest_errors = [errors[col] for col in cols[:how_many]]
    print largest_errors
    plt.bar(x_pos,largest_errors)
    plt.xticks(x_pos,cols[:how_many])
    plt.title("%d power lines with largest prediction error"%(how_many))
    plt.savefig("powerline_errors.jpg")
    plt.close("all")
    for field in cols[:how_many]:
        
        bigInterval = interval
        pred = predictions[field].resample("%dT"%bigInterval).mean()
        targ = targets[field].resample("%dT"%bigInterval).mean()
        
        #f, ax = plt.subplots(nrows=2,ncols=1, sharex=True)
        plt.subplot(2,1,1)
        x = np.arange(len(pred))
        pd.concat([targ,pred], axis=1).plot(figsize=(20,8),ax=plt.gca())
        plt.legend(['Targets (hourly)','Predictions (hourly)'])
        plt.title("hourly %s predictions vs targets, validation set error: %.3f"%(field,errors[field]))
        bigInterval = 60 * 24 * 7
        pred = predictions[field].resample("%dT"%bigInterval).mean()
        targ = targets[field].resample("%dT"%bigInterval).mean()
        plt.subplot(2,1,2)
        pd.concat([targ,pred], axis=1).plot(figsize=(20,8),ax=plt.gca())
        plt.legend(['Targets (weekly)','Predictions (weekly)'])
        plt.title("weekly %s predictions vs targets"%(field))
        plt.savefig("%s.jpg"%field)
        plt.close("all")

def fitPredictValidate(data_train_cv,targets_train_cv,data_validation_cv,targets_validation_cv,min_samples_leaf=5):
    [predictions_validation_cv, model]= fitAndPredictRandomForest(data_train_cv,targets_train_cv,data_validation_cv,min_samples_leaf=5)
    
    predictions_validation_cv = pd.DataFrame(predictions_validation_cv,columns=targets_train_cv.columns,index=data_validation_cv.index)
    if interval<60:
        predictions_validation_cv = predictions_validation_cv.resample('60T').mean().fillna(method='pad')
        #predictions_validation_cv = predictions_validation_cv.as_matrix()
        targets_validation_cv = targets_validation_cv.resample('60T').mean().fillna(method='pad')

    error = marsexpress_error(predictions_validation_cv,targets_validation_cv)
    
    plot_predictions(predictions_validation_cv,targets_validation_cv,"prediction_validation.jpg")
    
    predictions_test = pd.DataFrame(model.predict(data_train_cv),columns=targets_train_cv.columns,index=data_train_cv.index).resample('60T').mean().fillna(method='pad')
    plot_predictions(predictions_test,targets_train_cv,"prediction_train.jpg")
    plot_differences(predictions_validation_cv,targets_validation_cv)
    return [predictions_validation_cv, model,error]

    
if do_validation:
    print "fitting model for validation set"
    [predictions_validation_cv, model,error]= fitPredictValidate(data_train_cv,targets_train_cv,data_validation_cv,targets_validation_cv)
    print "Validation error: {} ".format(error)

    print "Feature importance:"
    for importance,feature in sorted(zip(model.feature_importances_, data_train.columns), key=lambda x: x[0], reverse=True):
        if importance > 0.001:
             print "%.3f"%importance,feature

results=[]
if do_parameter_testing:
    print "running parameter testing"
    for i in range(30):
        print "run ",i
        min_samples_leaf = np.random.randint(1,200)
        [predictions_validation_cv, model,error]= fitPredictValidate(data_train_cv,targets_train_cv,data_validation_cv,targets_validation_cv,min_samples_leaf=min_samples_leaf)
        
        results.append([min_samples_leaf, error])
        results2 = np.array(results)
        plt.scatter(results2[:,0],results2[:,1])
        plt.title('min_samples_leaf')
        plt.savefig("crossval.jpg")
        
    print ""

if do_test:
    print "fitting model for test set"
    [predictions_test, model]= fitAndPredictRandomForest(data_train,targets_train,data_test)

    #Converting the prediction matrix to a dataframe
    predictions_test=pd.DataFrame(predictions_test, index=data_test.index, columns=target_cols)
    if interval<60:
        predictions_test = predictions_test.resample('60T').mean()
    # We need to convert the parsed datetime back to utc timestamp
    predictions_test['ut_ms'] = (predictions_test.index.astype(np.int64) * 1e-6).astype(int)
    # Writing the submission file as csv
    predictions_test[['ut_ms'] + target_cols].to_csv('/tmp/rf_baseline.csv', index=False)

print "\n\n == Continue playing around or press STRG-D to exit == \n\n"
keyboard()

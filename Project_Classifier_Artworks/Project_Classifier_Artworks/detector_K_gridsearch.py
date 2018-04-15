# Grid search over K, SVM params, and AdaBoost params.
# Use SIFT features generated

import K_grid_search as search
import numpy as np
import cv2
import argparse
import visual_bow as bow
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.externals import joblib
from glob import glob
import random
import warnings
import os
import pickle
import coremltools

class FileHelpers:

    def __init__(self):
        pass

    def getFiles(self, path):
        """
        - returns  a dictionary of all files 
        having key => value as  objectname => image path

        - returns total number of files.

        """
        imlist = {}
        count = 0
        for each in glob(path + "*"):
            word = each.split("/")[-1]
            print " #### Reading image category ", word, " ##### "
            imlist[word] = []
            for imagefile in glob(path+word+"/*"):
                print "Reading file ", imagefile
                im = cv2.imread(imagefile, cv2.IMREAD_GRAYSCALE)
                imlist[word].append(im)
                count +=1 

        return [imlist, count]


#scoring = 'recall_micro'
scoring = 'accuracy'
train_path = None
test_path = None
file_helper = FileHelpers()
images = None
trainImageCount = 0
train_labels = np.array([])
name_dict = {}
descriptor_list = []

parser = argparse.ArgumentParser(description=" Bag of visual words example")
parser.add_argument('--train_path', action="store", dest="train_path", required=True)
args =  vars(parser.parse_args())
print args

# set training paths
train_path = args['train_path'] 

 # read file. prepare file lists.
images, trainImageCount = file_helper.getFiles(train_path)
print trainImageCount
print images
img_descs, y, train_labels = bow.gen_sift_features(images)
# print y
# print train_labels
# print img_descs

joblib.dump(img_descs, 'pickles/img_descs/img_descs.pickle')
joblib.dump(train_labels, 'pickles/img_descs/train_labels.pickle')

# # load SIFT features (eg from panda_detector_more_data notebook)
img_descs = joblib.load('pickles/img_descs/img_descs.pickle')
train_labels = joblib.load('pickles/img_descs/train_labels.pickle')

print len(img_descs), len(train_labels)

# generate indexes for train/test/val split
training_idxs, test_idxs, val_idxs = search.bow.train_test_val_split_idxs(
    total_rows=len(img_descs),
    percent_test=0.15,
    percent_val=0.15
)

results = {}
K_vals = [50, 150, 300, 500]

for K in K_vals:
    X_train, X_test, X_val, y_train, y_test, y_val, cluster_model = search.cluster_and_split(
        img_descs, train_labels, training_idxs, test_idxs, val_idxs, K)

    print "\nInertia for clustering with K=%i is:" % K, cluster_model.inertia_

    print '\nSVM Scores: '
    svmGS, svm_score = search.run_svm(X_train, X_test, y_train, y_test, train_labels, scoring)
    print '\nSVM_Model Scores: '
    input_data, output_data = search.run_svm_model(X_train, X_test, y_train, y_test, train_labels, scoring)
    print '\nAdaBoost Scores: '
    adaGS, ada_score = search.run_ada(X_train, X_test, y_train, y_test, scoring)

    results[K] = dict(
        inertia = cluster_model.inertia_,
        svmGS=svmGS,
        adaGS=adaGS,
        cluster_model=cluster_model,
        svm_score=svm_score,
        ada_score=ada_score)

    print '\n*** K=%i DONE ***\n' % K

print '**************************'
print '***** FINISHED ALL K *****'
print '**************************\n'

# pickle for later analysis
###########################

feature_data_path = 'pickles/k_grid_feature_data/'
result_path = 'pickles/k_grid_result'

# # delete previous pickles
# for path in [feature_data_path, result_path]:
#     for f in glob.glob(path+'/*'):
#         os.remove(f)

print 'pickling X_train, X_test, X_val, y_train, y_test, y_val'

for obj, obj_name in zip( [X_train, X_test, X_val, y_train, y_test, y_val],
                         ['X_train', 'X_test', 'X_val', 'y_train', 'y_test', 'y_val'] ):
    joblib.dump(obj, '%s%s.pickle' % (feature_data_path, obj_name))

print 'pickling results'

exports = joblib.dump(results, '%s/result.pickle' % result_path)

print("Saving the trained model...")
SAFE_DIRECTORY_PATH = "/home/guru/Project_Classifier/pickles/svc"

filename = os.path.join(SAFE_DIRECTORY_PATH, 'svc.pickle')
model = pickle.load(open(filename,'rb'))
coreml_model = coremltools.converters.sklearn.convert(model)
print coreml_model
coreml_model.save('artwork.mlmodel')
print('Core ML Model saved')

print '\n* * *'
print 'Scored grid search with metric: "%s"' % scoring

K_vals = sorted(results.keys())
for K in K_vals:
    print 'For K = %i:\tSVM %f\tAdaBoost %f\tK-Means Inertia %f' % (
        K, results[K]['svm_score'], results[K]['ada_score'], results[K]['inertia']);

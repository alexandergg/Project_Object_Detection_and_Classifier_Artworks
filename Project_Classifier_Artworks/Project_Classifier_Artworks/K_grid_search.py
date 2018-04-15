# Convenience functions to run a grid search over the classiers and over K in KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import KFold
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
import glob
import random
import warnings
import coremltools
import numpy as np
import os
import pickle
import visual_bow as bow

def cluster_and_split(img_descs, y, training_idxs, test_idxs, val_idxs, K):
    """Cluster into K clusters, then split into train/test/val"""
    # MiniBatchKMeans annoyingly throws tons of deprecation warnings that fill up the notebook. Ignore them.
    warnings.filterwarnings('ignore')

    X, cluster_model = bow.cluster_features(
        img_descs,
        training_idxs=training_idxs,
        cluster_model=MiniBatchKMeans(n_clusters=K)
    )

    print("Saving the cluster model...")
    SAFE_DIRECTORY_PATH = "/home/guru/Project_Classifier/pickles/cluster_model"
    model_filename = os.path.join(SAFE_DIRECTORY_PATH , 'cluster_model.pickle')
    pickle.dump(cluster_model, open(model_filename, 'wb'))
    
    warnings.filterwarnings('default')

    X_train, X_test, X_val, y_train, y_test, y_val = bow.perform_data_split(X, y, training_idxs, test_idxs, val_idxs)

    return X_train, X_test, X_val, y_train, y_test, y_val, cluster_model

def run_svm(X_train, X_test, y_train, y_test, train_labels, scoring,
    c_vals=[1, 5, 10], gamma_vals=[0.1, 0.01, 0.0001, 0.00001]):

    param_grid = [
    #   {'C': c_vals, 'kernel': ['linear']},
      {'C': c_vals, 'gamma': gamma_vals, 'kernel': ['rbf']},
     ]

    svc = GridSearchCV(SVC(), param_grid, n_jobs=-1, scoring=scoring)
    svc.fit(X_train, y_train)
    
    y_pred = svc.predict(X_test)

    print classification_report(y_test, y_pred)
    print 'train score (%s):'%scoring, svc.score(X_train, y_train)
    test_score = svc.score(X_test, y_test)
    print 'test score (%s):'%scoring, test_score

    print svc.best_estimator_
    #joblib.dump(cluster_model, 'pickles/cluster_model/cluster_model.pickle')
    return svc, test_score

def run_svm_model(X_train, X_test, y_train, y_test, train_labels, scoring):

    svc = SVC(C=5, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3, gamma=0.0001, kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)
    svc.fit(X_train, y_train)
    
    y_pred = svc.predict(X_test)

    print classification_report(y_test, y_pred)
    print 'train score (%s):'%scoring, svc.score(X_train, y_train)
    test_score = svc.score(X_test, y_test)
    print 'test score (%s):'%scoring, test_score

    print("Saving the trained model...")
    SAFE_DIRECTORY_PATH = "/home/guru/Project_Classifier/pickles/svc"
    model_filename = os.path.join(SAFE_DIRECTORY_PATH , 'svc.pickle')
    pickle.dump(svc, open(model_filename, 'wb'))

    return X_train, y_train

def run_ada(X_train, X_test, y_train, y_test, scoring,
    n_estimators=[50, 100, 250], learning_rate=[1.0, 1.5]):

    ada_params={
        'n_estimators':n_estimators,
        'learning_rate':learning_rate
    }

    ada = GridSearchCV(AdaBoostClassifier(), ada_params, n_jobs=-1, scoring=scoring)
    ada.fit(X_train, y_train)

    print 'train score (%s):'%scoring, ada.score(X_train, y_train)
    test_score = ada.score(X_test, y_test)
    print 'test score (%s):'%scoring, test_score
    print ada.best_estimator_

    return ada, test_score

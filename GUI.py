import os
import pickle
import sys
import tempfile

import keras
import numpy as np
import pandas as pd
from PyQt5 import uic, QtWidgets
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import QFileDialog, QApplication
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

import elm_model
import gradient_descent
import kmeans
import k_neighbors
import linear_models
import log_reg
import naive
import neural_network
import svm
import tree_models
from boosting import adaboost, xgb, lightgbm
from ridge_classifier import ridge
from theilsen import theil_sen
from tree_models import rgf

qtCreatorFile = "demo.ui"

Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)


# to make keras model serializable
def make_keras_picklable():
    def __getstate__(self):
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            keras.models.save_model(self, fd.name, overwrite=True)
            model_str = fd.read()
        d = {'model_str': model_str}
        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state['model_str'])
            fd.flush()
            model = keras.models.load_model(fd.name)
        self.__dict__ = model.__dict__

    cls = keras.models.Model
    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__


def data():

    # remove language name feature
    df.drop('LanguageName', 1, inplace=True)

    # chane column types

    df['CountryName'] = df['CountryName'].astype('str')
    df['Gender'] = df['Gender'].astype('str')
    df['CountryName'] = df['CountryName'].str.replace(r"[b\'.COM]", '')
    df['Gender'] = df['Gender'].str.replace(r"[b\']", '')

    df.dropna()

    date_df = df2.sort_values('Date').groupby('UserID')['Date'].agg(['first', 'last']).reset_index()

    date_df['Duration'] = date_df['last'] - date_df['first']
    date_df['Duration_Days'] = date_df['Duration'].dt.days
    date_df.drop('first', 1, inplace=True)
    date_df.drop('last', 1, inplace=True)
    date_df.drop('Duration', 1, inplace=True)

    # merging of the datasets

    date_df = pd.merge(left=date_df, right=df2, left_on='UserID', right_on='UserID')

    date_df.drop('Date', 1, inplace=True)

    merged_inner = pd.merge(left=df, right=date_df, left_on='USERID', right_on='UserID')

    # creating our label

    df3['AtRisk'] = np.where(df3['RGsumevents'] != 0, 1, 0)

    df3.drop(df3.columns[2:6], axis=1, inplace=True)

    features = pd.merge(merged_inner, df3, on='UserID', how='left')
    features.loc[features.AtRisk != 1, 'AtRisk'] = 0
    features['AtRisk'] = features['AtRisk'].astype(np.int64)

    features.loc[features.RGsumevents.isnull(), 'RGsumevents'] = 0
    features['RGsumevents'] = features['RGsumevents'].astype(np.int64)

    features.drop(features.columns[5:8], axis=1, inplace=True)

    aggregation_functions = {'CountryName': 'first', 'Gender': 'first',
                             'YearofBirth': 'first', 'Turnover': 'sum', 'Hold': 'sum',
                             'NumberofBets': 'sum',
                             'Duration_Days': 'first', 'RGsumevents': 'first', 'AtRisk': 'first'}
    features = features.groupby(features['USERID'], as_index=False).aggregate(aggregation_functions)

    features['Profit'] = np.where((features['Turnover'] - features['Hold']) > 0, 1, 0)

    features = pd.get_dummies(features, columns=["CountryName", "Gender"])

    # fixing column names

    # Correct Country Names
    features.rename(columns={'CountryName_Bosnia and Herzego': 'CountryName_BosniaHerzegovina',
                             'CountryName_FYR acedonia': 'CountryName_Macedonia',
                             'CountryName_Leanon': 'CountryName_Lebanon',
                             'CountryName_Luxemourg': 'CountryName_Luxembourg',
                             'CountryName_New Zealand': 'CountryName_NewZealand',
                             'CountryName_Russian Federation': 'CountryName_RussianFederation',
                             'CountryName_alta': 'CountryName_Malta',
                             'CountryName_anada': 'CountryName_Canada',
                             'CountryName_exicoX': 'CountryName_Mexico',
                             'CountryName_orocco': 'CountryName_Morocco',
                             'CountryName_roatia': 'CountryName_Croatia',
                             'CountryName_yprus': 'CountryName_Cyprus',
                             'CountryName_zech Repulic': 'CountryName_CzechRepulic'}, inplace=True)

    # Reorder Columns
    cols = list(features.columns.values)
    cols.pop(cols.index('Profit'))
    cols.pop(cols.index('AtRisk'))
    features = features[cols + ['Profit', 'AtRisk']]

    features['USERID'] = features['USERID'].astype(np.int32)
    features['NumberofBets'] = features['NumberofBets'].astype(np.int32)

    features.drop('YearofBirth', 1, inplace=True)
    features.drop('RGsumevents', 1, inplace=True)

    features = features[features.NumberofBets != 0]

    # IMPORTANT TO PLAY WITH !!!!
    features = features[features.Duration_Days > 5]

    features.dropna(inplace=True)
    features.drop_duplicates(inplace=True)

    labels = np.array(features['AtRisk'])

    features.drop('AtRisk', axis=1, inplace=True)
    # features.drop('USERID', 1, inplace=True)

    df_list = list(features.columns)

    if 'USERID' in df_list: df_list.remove('USERID')

    # features = np.array(features)

    # data splitting

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    X_train.drop('USERID', 1, inplace=True)

    X_train = np.array(X_train)

    # to_predict = X_test.drop('USERID', 1)

    # predict_set = pd.DataFrame(to_predict, columns=df_list)

    # predict_set.to_csv('data/pred.csv') # save to file

    user_id_for_prediction = pd.DataFrame(np.array(X_test['USERID']))

    # user_id_for_prediction.to_csv('data/predict_ids.csv') # save to file

    user_id_for_prediction = np.array(X_test['USERID'])

    return X_train, y_train, user_id_for_prediction


class External(QThread):
    countChanged = pyqtSignal(int)
    textChanged = pyqtSignal(str)
    stateChanged = pyqtSignal(int)

    nn_auroc_cv, nn_accuracy_cv, nn_sensitivity_cv, nn_spec_cv = 0, 0, 0, 0
    elm_auroc_cv, elm_accuracy_cv, elm_sensitivity_cv, elm_spec_cv = 0, 0, 0, 0
    rf_auroc_cv, rf_accuracy_cv, rf_sensitivity_cv, rf_spec_cv = 0, 0, 0, 0
    extreme_rf_auroc_cv, extreme_rf_accuracy_cv, extreme_rf_sensitivity_cv, extreme_rf_spec_cv = 0, 0, 0, 0
    dtc_auroc_cv, dtc_accuracy_cv, dtc_sensitivity_cv, dtc_spec_cv = 0, 0, 0, 0

    rgf_auroc_cv, rgf_accuracy_cv, rgf_sensitivity_cv, rgf_spec_cv = 0, 0, 0, 0
    adaboost_auroc_cv, adaboost_accuracy_cv, adaboost_sensitivity_cv, adaboost_spec_cv = 0, 0, 0, 0
    xgb_auroc_cv, xgb_accuracy_cv, xgb_sensitivity_cv, xgb_spec_cv = 0, 0, 0, 0
    gbm_auroc_cv, gbm_accuracy_cv, gbm_sensitivity_cv, gbm_spec_cv = 0, 0, 0, 0

    k_auroc_cv, k_accuracy_cv, k_sensitivity_cv, k_spec_cv = 0, 0, 0, 0
    bagging_k_nearest_auroc_cv, bagging_k_nearest_accuracy_cv, bagging_k_nearest_sensitivity_cv, \
    bagging_k_nearest_spec_cv = 0, 0, 0, 0
    svm_auroc_cv, svm_accuracy_cv, svm_sensitivity_cv, svm_spec_cv = 0, 0, 0, 0
    bagging_svm_auroc_cv, bagging_svm_accuracy_cv, bagging_svm_sensitivity_cv, bagging_svm_spec_cv = 0, 0, 0, 0

    lsvc_auroc_cv, lsvc_accuracy_cv, lsvc_sensitivity_cv, lsvc_spec_cv = 0, 0, 0, 0
    naive_auroc_cv, naive_accuracy_cv, naive_sensitivity_cv, naive_spec_cv = 0, 0, 0, 0
    bagging_naive_auroc_cv, bagging_naive_accuracy_cv, bagging_naive_sensitivity_cv, bagging_naive_spec_cv = 0, 0, 0, 0
    bernoulli_auroc_cv, bernoulli_accuracy_cv, bernoulli_sensitivity_cv, bernoulli_spec_cv = 0, 0, 0, 0

    lr_auroc_cv, lr_accuracy_cv, lr_sensitivity_cv, lr_spec_cv = 0, 0, 0, 0
    lrr_auroc_cv, lrr_accuracy_cv, lrr_sensitivity_cv, lrr_spec_cv = 0, 0, 0, 0
    tsr_auroc_cv, tsr_accuracy_cv, tsr_sensitivity_cv, tsr_spec_cv = 0, 0, 0, 0

    lda_auroc_cv, lda_accuracy_cv, lda_sensitivity_cv, lda_spec_cv = 0, 0, 0, 0
    sgd_auroc_cv, sgd_accuracy_cv, sgd_sensitivity_cv, sgd_spec_cv = 0, 0, 0, 0
    rc_auroc_cv, rc_accuracy_cv, rc_sensitivity_cv, rc_spec_cv = 0, 0, 0, 0

    def run(self):
        count = 0

        self.stateChanged.emit(0)
        self.countChanged.emit(count)

        External.features, External.labels, External.predict_ids = data()

        k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        count += 15
        self.countChanged.emit(count)

        make_keras_picklable()

        for train_index, test_index in k_fold.split(External.features, External.labels):
            train_features, test_features = External.features[train_index], External.features[test_index]
            train_labels, test_labels = External.labels[train_index], External.labels[test_index]

            scale = StandardScaler()
            scale.fit(train_features)  # fitting of training data to be scaled
            train_features = scale.transform(train_features)
            test_features = scale.transform(test_features)

            count += 1
            self.countChanged.emit(count)

            # Neural Network

            External.nn_model, External.nn_auroc, External.nn_accuracy, External.nn_sensitivity, External.nn_spec = \
                neural_network.neural_model(train_features, train_labels, test_features, test_labels)
            External.nn_auroc_cv += External.nn_auroc
            External.nn_accuracy_cv += External.nn_accuracy
            External.nn_sensitivity_cv += External.nn_sensitivity
            External.nn_spec_cv += External.nn_spec
            count += 0.2
            self.countChanged.emit(count)
            text = "Neural Network Performance: \n-----\n" \
                   "AUROC: " + str(External.nn_auroc) + \
                   "\nAccuracy: " + str(External.nn_accuracy) + \
                   "%.\nSensitivity: " + str(External.nn_sensitivity) + \
                   "%. \nSpecificity: " + str(External.nn_spec) + "%. \n-----\n"
            self.textChanged.emit(text)

            # ELM

            External.elm_model, External.elm_auroc, External.elm_accuracy, External.elm_sensitivity, \
            External.elm_spec = elm_model.elm_model(train_features, train_labels, test_features, test_labels)
            External.elm_auroc_cv += External.elm_auroc
            External.elm_accuracy_cv += External.elm_accuracy
            External.elm_sensitivity_cv += External.elm_sensitivity
            External.elm_spec_cv += External.elm_spec
            count += 0.3
            self.countChanged.emit(count)
            text = "Extreme Learning Machine Performance: \n-----\n" \
                   "AUROC: " + str(External.elm_auroc) + \
                   "\nAccuracy: " + str(External.elm_accuracy) + \
                   "%.\nSensitivity: " + str(External.elm_sensitivity) + \
                   "%. \nSpecificity: " + str(External.elm_spec) + "%. \n-----\n"
            self.textChanged.emit(text)

            # Random Forest

            External.rf_model, External.rf_auroc, External.rf_accuracy, External.rf_sensitivity, External.rf_spec = \
                tree_models.random_forest(train_features, train_labels, test_features, test_labels)

            External.rf_auroc_cv += External.rf_auroc
            External.rf_accuracy_cv += External.rf_accuracy
            External.rf_sensitivity_cv += External.rf_sensitivity
            External.rf_spec_cv += External.rf_spec

            count += 0.5
            self.countChanged.emit(count)
            text = "Random Forest Performance: \n-----\n" \
                   "AUROC: " + str(External.rf_auroc) + \
                   "\nAccuracy: " + str(External.rf_accuracy) + \
                   "%.\nSensitivity: " + str(External.rf_sensitivity) + \
                   "%. \nSpecificity: " + str(External.rf_spec) + "%. \n-----\n"
            self.textChanged.emit(text)

            # Extreme RF

            External.extreme_rf_model, External.extreme_rf_auroc, External.extreme_rf_accuracy, \
            External.extreme_rf_sensitivity, External.extreme_rf_spec = \
                tree_models.extremely_random_trees(train_features, train_labels, test_features, test_labels)

            External.extreme_rf_auroc_cv += External.extreme_rf_auroc
            External.extreme_rf_accuracy_cv += External.extreme_rf_accuracy
            External.extreme_rf_sensitivity_cv += External.extreme_rf_sensitivity
            External.extreme_rf_spec_cv += External.extreme_rf_spec

            count += 0.5
            self.countChanged.emit(count)
            text = "Extremely Randomised Forest Performance: \n-----\n" \
                   "AUROC: " + str(External.extreme_rf_auroc) + \
                   "\nAccuracy: " + str(External.extreme_rf_accuracy) + \
                   "%.\nSensitivity: " + str(External.extreme_rf_sensitivity) + \
                   "%. \nSpecificity: " + str(External.extreme_rf_spec) + "%. \n-----\n"
            self.textChanged.emit(text)

            # DTC

            External.dtc_model, External.dtc_auroc, External.dtc_accuracy, External.dtc_sensitivity, External.dtc_spec = \
                tree_models.dtc(train_features, train_labels, test_features, test_labels)

            External.dtc_auroc_cv += External.dtc_auroc
            External.dtc_accuracy_cv += External.dtc_accuracy
            External.dtc_sensitivity_cv += External.dtc_sensitivity
            External.dtc_spec_cv += External.dtc_spec

            count += 0.2
            self.countChanged.emit(count)
            text = "Decision Tree Classifier Performance: \n-----\n" \
                   "AUROC: " + str(External.dtc_auroc) + \
                   "\nAccuracy: " + str(External.dtc_accuracy) + \
                   "%.\nSensitivity: " + str(External.dtc_sensitivity) + \
                   "%. \nSpecificity: " + str(External.dtc_spec) + "%. \n-----\n"
            self.textChanged.emit(text)

            # RGF

            External.rgf_model, External.rgf_auroc, External.rgf_accuracy, External.rgf_sensitivity, External.rgf_spec = \
                rgf(train_features, train_labels, test_features, test_labels)

            External.rgf_auroc_cv += External.rgf_auroc
            External.rgf_accuracy_cv += External.rgf_accuracy
            External.rgf_sensitivity_cv += External.rgf_sensitivity
            External.rgf_spec_cv += External.rgf_spec

            count += 0.3
            self.countChanged.emit(count)
            text = "Regularized Greedy Forests Performance: \n-----\n" \
                   "AUROC: " + str(External.rgf_auroc) + \
                   "\nAccuracy: " + str(External.rgf_accuracy) + \
                   "%.\nSensitivity: " + str(External.rgf_sensitivity) + \
                   "%. \nSpecificity: " + str(External.rgf_spec) + "%. \n-----\n"
            self.textChanged.emit(text)

            # ADABOOST

            External.adaboost_model, External.adaboost_auroc, \
            External.adaboost_accuracy, External.adaboost_sensitivity, External.adaboost_spec = \
                adaboost(train_features, train_labels, test_features, test_labels)

            External.adaboost_auroc_cv += External.adaboost_auroc
            External.adaboost_accuracy_cv += External.adaboost_accuracy
            External.adaboost_sensitivity_cv += External.adaboost_sensitivity
            External.adaboost_spec_cv += External.adaboost_spec

            count += 0.2
            self.countChanged.emit(count)
            text = "AdaBoost with Decision Tree Base Performance: \n-----\n" \
                   "AUROC: " + str(External.adaboost_auroc) + \
                   "\nAccuracy: " + str(External.adaboost_accuracy) + \
                   "%.\nSensitivity: " + str(External.adaboost_sensitivity) + \
                   "%. \nSpecificity: " + str(External.adaboost_spec) + "%. \n-----\n"
            self.textChanged.emit(text)

            # XGB

            External.xgb_model, External.xgb_auroc, \
            External.xgb_accuracy, External.xgb_sensitivity, External.xgb_spec = \
                xgb(train_features, train_labels, test_features, test_labels)

            External.xgb_auroc_cv += External.xgb_auroc
            External.xgb_accuracy_cv += External.xgb_accuracy
            External.xgb_sensitivity_cv += External.xgb_sensitivity
            External.xgb_spec_cv += External.xgb_spec

            count += 0.3
            self.countChanged.emit(count)
            text = "XGradient Boosting Performance: \n-----\n" \
                   "AUROC: " + str(External.xgb_auroc) + \
                   "\nAccuracy: " + str(External.xgb_accuracy) + \
                   "%.\nSensitivity: " + str(External.xgb_sensitivity) + \
                   "%. \nSpecificity: " + str(External.xgb_spec) + "%. \n-----\n"
            self.textChanged.emit(text)

            # LightGBM

            External.gbm_model, External.gbm_auroc, \
            External.gbm_accuracy, External.gbm_sensitivity, External.gbm_spec = \
                lightgbm(train_features, train_labels, test_features, test_labels)

            External.gbm_auroc_cv += External.gbm_auroc
            External.gbm_accuracy_cv += External.gbm_accuracy
            External.gbm_sensitivity_cv += External.gbm_sensitivity
            External.gbm_spec_cv += External.gbm_spec

            count += 0.3
            self.countChanged.emit(count)
            text = "Light Gradient Boosting Performance: \n-----\n" \
                   "AUROC: " + str(External.xgb_auroc) + \
                   "\nAccuracy: " + str(External.xgb_accuracy) + \
                   "%.\nSensitivity: " + str(External.xgb_sensitivity) + \
                   "%. \nSpecificity: " + str(External.xgb_spec) + "%. \n-----\n"
            self.textChanged.emit(text)

            # K-Nearest

            External.k_model, External.k_nearest_auroc, External.k_nearest_accuracy, \
            External.k_nearest_sensitivity, External.k_nearest_spec = \
                k_neighbors.k_nearest(train_features, train_labels, test_features, test_labels)

            External.k_auroc_cv += External.k_nearest_auroc
            External.k_accuracy_cv += External.k_nearest_accuracy
            External.k_sensitivity_cv += External.k_nearest_sensitivity
            External.k_spec_cv += External.k_nearest_spec

            count += 0.2
            self.countChanged.emit(count)
            text = "K-Nearest Neighbors Performance: \n-----\n" \
                   "AUROC: " + str(External.k_nearest_auroc) + \
                   "\nAccuracy: " + str(External.k_nearest_accuracy) + \
                   "%.\nSensitivity: " + str(External.k_nearest_sensitivity) + \
                   "%. \nSpecificity: " + str(External.k_nearest_spec) + "%. \n-----\n"
            self.textChanged.emit(text)

            # Bagging K-Nearest

            External.bagging_k_model, External.bagging_k_auroc, External.bagging_k_nearest_accuracy, \
            External.bagging_k_nearest_sensitivity, External.bagging_k_nearest_spec = \
                k_neighbors.bagging_k_nearest(train_features, train_labels, test_features, test_labels)

            External.bagging_k_nearest_auroc_cv += External.bagging_k_auroc
            External.bagging_k_nearest_accuracy_cv += External.bagging_k_nearest_accuracy
            External.bagging_k_nearest_sensitivity_cv += External.bagging_k_nearest_sensitivity
            External.bagging_k_nearest_spec_cv += External.bagging_k_nearest_spec

            count += 0.5
            self.countChanged.emit(count)
            text = "Bagging K-Nearest Neighbors Performance: \n-----\n" \
                   "AUROC: " + str(External.bagging_k_auroc) + \
                   "\nAccuracy: " + str(External.bagging_k_nearest_accuracy) + \
                   "%.\nSensitivity: " + str(External.bagging_k_nearest_sensitivity) + \
                   "%. \nSpecificity: " + str(External.bagging_k_nearest_spec) + "%. \n-----\n"
            self.textChanged.emit(text)

            # SVM

            External.svm_model, External.svm_auroc, External.svm_accuracy, \
            External.svm_sensitivity, External.svm_spec = \
                svm.svm(train_features, train_labels, test_features, test_labels)

            External.svm_auroc_cv += External.svm_auroc
            External.svm_accuracy_cv += External.svm_accuracy
            External.svm_sensitivity_cv += External.svm_sensitivity
            External.svm_spec_cv += External.svm_spec

            count += 0.5
            self.countChanged.emit(count)
            text = "Support Vector Machine Performance: \n-----\n" \
                   "AUROC: " + str(External.svm_auroc) + \
                   "\nAccuracy: " + str(External.svm_accuracy) + \
                   "%.\nSensitivity: " + str(External.svm_sensitivity) + \
                   "%. \nSpecificity: " + str(External.svm_spec) + "%. \n-----\n"
            self.textChanged.emit(text)

            # Bagging SVM

            External.bagging_svm_model, External.bagging_svm_auroc, External.bagging_svm_accuracy, \
            External.bagging_svm_sensitivity, External.bagging_svm_spec = \
                svm.bagging_svm(train_features, train_labels, test_features, test_labels)

            External.bagging_svm_auroc_cv += External.bagging_svm_auroc
            External.bagging_svm_accuracy_cv += External.bagging_svm_accuracy
            External.bagging_svm_sensitivity_cv += External.bagging_svm_sensitivity
            External.bagging_svm_spec_cv += External.bagging_svm_spec

            count += 0.2
            self.countChanged.emit(count)
            text = "Bagging Support Vector Machine Performance: \n-----\n" \
                   "AUROC: " + str(External.bagging_svm_auroc) + \
                   "\nAccuracy: " + str(External.bagging_svm_accuracy) + \
                   "%.\nSensitivity: " + str(External.bagging_svm_sensitivity) + \
                   "%. \nSpecificity: " + str(External.bagging_svm_spec) + "%. \n-----\n"
            self.textChanged.emit(text)

            # Linear SVC

            External.lsvc_model, External.lsvc_auroc, External.lsvc_accuracy, \
            External.lsvc_sensitivity, External.lsvc_spec = \
                svm.linear_svc(train_features, train_labels, test_features, test_labels)

            External.lsvc_auroc_cv += External.lsvc_auroc
            External.lsvc_accuracy_cv += External.lsvc_accuracy
            External.lsvc_sensitivity_cv += External.lsvc_sensitivity
            External.lsvc_spec_cv += External.lsvc_spec

            count += 0.3
            self.countChanged.emit(count)
            text = "Linear SVC Performance: \n-----\n" \
                   "AUROC: " + str(External.lsvc_auroc) + \
                   "\nAccuracy: " + str(External.lsvc_accuracy) + \
                   "%.\nSensitivity: " + str(External.lsvc_sensitivity) + \
                   "%. \nSpecificity: " + str(External.lsvc_spec) + "%. \n-----\n"
            self.textChanged.emit(text)

            # Naive Bayes

            External.naive_model, External.naive_auroc, External.naive_accuracy, \
            External.naive_sensitivity, External.naive_spec = \
                naive.naive(train_features, train_labels, test_features, test_labels)

            External.naive_auroc_cv += External.naive_auroc
            External.naive_accuracy_cv += External.naive_accuracy
            External.naive_sensitivity_cv += External.naive_sensitivity
            External.naive_spec_cv += External.naive_spec

            count += 0.5
            self.countChanged.emit(count)
            text = "Naive Bayes Performance: \n-----\n" \
                   "AUROC: " + str(External.naive_auroc) + \
                   "\nAccuracy: " + str(External.naive_accuracy) + \
                   "%.\nSensitivity: " + str(External.naive_sensitivity) + \
                   "%. \nSpecificity: " + str(External.naive_spec) + "%. \n-----\n"
            self.textChanged.emit(text)

            # Bagging NB

            External.bagging_naive_model, External.bagging_naive_auroc, \
            External.bagging_naive_accuracy, External.bagging_naive_sens, External.bagging_naive_spec = \
                naive.bagging_nb(train_features, train_labels, test_features, test_labels)

            External.bagging_naive_auroc_cv += External.bagging_naive_auroc
            External.bagging_naive_accuracy_cv += External.bagging_naive_accuracy
            External.bagging_naive_sensitivity_cv += External.bagging_naive_sens
            External.bagging_naive_spec_cv += External.bagging_naive_spec

            count += 0.5
            self.countChanged.emit(count)
            text = "Bagging Naive Bayes Performance: \n-----\n" \
                   "AUROC: " + str(External.bagging_naive_auroc) + \
                   "\nAccuracy: " + str(External.bagging_naive_accuracy) + \
                   "%.\nSensitivity: " + str(External.bagging_naive_sens) + \
                   "%. \nSpecificity: " + str(External.bagging_naive_spec) + "%. \n-----\n"
            self.textChanged.emit(text)

            # Bernoulli NB

            External.bernoulli_model, External.bernoulli_auroc, \
            External.bernoulli_accuracy, External.bernoulli_sensitivity, \
            External.bernoulli_spec = \
                naive.bernoulli_naive(train_features, train_labels, test_features, test_labels)

            External.bernoulli_auroc_cv += External.bernoulli_auroc
            External.bernoulli_accuracy_cv += External.bernoulli_accuracy
            External.bernoulli_sensitivity_cv += External.bernoulli_sensitivity
            External.bernoulli_spec_cv += External.bernoulli_spec

            count += 0.5
            self.countChanged.emit(count)
            text = "Bernoulli Naive Bayes Performance: \n-----\n" \
                   "AUROC: " + str(External.bernoulli_auroc) + \
                   "\nAccuracy: " + str(External.bernoulli_accuracy) + \
                   "%.\nSensitivity: " + str(External.bernoulli_sensitivity) + \
                   "%. \nSpecificity: " + str(External.bernoulli_spec) + "%. \n-----\n"
            self.textChanged.emit(text)

            # Logistic Regression

            External.lr_model, External.lr_auroc, External.lr_accuracy, External.lr_sensitivity, External.lr_spec = \
                log_reg.lr(train_features, train_labels, test_features, test_labels)

            External.lr_auroc_cv += External.lr_auroc
            External.lr_accuracy_cv += External.lr_accuracy
            External.lr_sensitivity_cv += External.lr_sensitivity
            External.lr_spec_cv += External.lr_spec

            count += 0.3
            self.countChanged.emit(count)
            text = "Logistic Regression Performance: \n-----\n" \
                   "AUROC: " + str(External.lr_auroc) + \
                   "\nAccuracy: " + str(External.lr_accuracy) + \
                   "%.\nSensitivity: " + str(External.lr_sensitivity) + \
                   "%. \nSpecificity: " + str(External.lr_spec) + "%. \n-----\n"
            self.textChanged.emit(text)

            # Linear Regression

            External.lrr_model, External.lrr_auroc, External.lrr_accuracy, External.lrr_sensitivity, External.lrr_spec = \
                linear_models.linear_regression(train_features, train_labels, test_features, test_labels)

            External.lrr_auroc_cv += External.lrr_auroc
            External.lrr_accuracy_cv += External.lrr_accuracy
            External.lrr_sensitivity_cv += External.lrr_sensitivity
            External.lrr_spec_cv += External.lrr_spec

            count += 0.2
            self.countChanged.emit(count)
            text = "Linear Regression Performance: \n-----\n" \
                   "AUROC: " + str(External.lrr_auroc) + \
                   "\nAccuracy: " + str(External.lrr_accuracy) + \
                   "%.\nSensitivity: " + str(External.lrr_sensitivity) + \
                   "%. \nSpecificity: " + str(External.lrr_spec) + "%. \n-----\n"
            self.textChanged.emit(text)

            # Theil Sen

            External.tsr_model, External.tsr_auroc, External.tsr_accuracy, External.tsr_sensitivity, External.tsr_spec = \
                theil_sen(train_features, train_labels, test_features, test_labels)

            External.tsr_auroc_cv += External.tsr_auroc
            External.tsr_accuracy_cv += External.tsr_accuracy
            External.tsr_sensitivity_cv += External.tsr_sensitivity
            External.tsr_spec_cv += External.tsr_spec

            count += 0.2
            self.countChanged.emit(count)
            text = "Theil-Sen Regressor Performance: \n-----\n" \
                   "AUROC: " + str(External.tsr_auroc) + \
                   "\nAccuracy: " + str(External.tsr_accuracy) + \
                   "%.\nSensitivity: " + str(External.tsr_sensitivity) + \
                   "%. \nSpecificity: " + str(External.tsr_spec) + "%. \n-----\n"
            self.textChanged.emit(text)

            # LDA

            External.lda_model, External.lda_auroc, External.lda_accuracy, External.lda_sensitivity, External.lda_spec = \
                linear_models.lda(train_features, train_labels, test_features, test_labels)

            External.lda_auroc_cv += External.lda_auroc
            External.lda_accuracy_cv += External.lda_accuracy
            External.lda_sensitivity_cv += External.lda_sensitivity
            External.lda_spec_cv += External.lda_spec

            count += 0.3
            self.countChanged.emit(count)
            text = "Linear Discriminant Analysis Performance: \n-----\n" \
                   "AUROC: " + str(External.lda_auroc) + \
                   "\nAccuracy: " + str(External.lda_accuracy) + \
                   "%.\nSensitivity: " + str(External.lda_sensitivity) + \
                   "%. \nSpecificity: " + str(External.lda_spec) + "%. \n-----\n"
            self.textChanged.emit(text)

            # SGD

            External.sgd_model, External.sgd_auroc, External.sgd_accuracy, External.sgd_sensitivity, External.sgd_spec = \
                gradient_descent.sgd(train_features, train_labels, test_features, test_labels)

            External.sgd_auroc_cv += External.sgd_auroc
            External.sgd_accuracy_cv += External.sgd_accuracy
            External.sgd_sensitivity_cv += External.sgd_sensitivity
            External.sgd_spec_cv += External.sgd_spec

            count += 0.2
            self.countChanged.emit(count)
            text = "Stochastic Gradient Descent Performance: \n-----\n" \
                   "AUROC: " + str(External.sgd_auroc) + \
                   "\nAccuracy: " + str(External.sgd_accuracy) + \
                   "%.\nSensitivity: " + str(External.sgd_sensitivity) + \
                   "%. \nSpecificity: " + str(External.sgd_spec) + "%. \n-----\n"
            self.textChanged.emit(text)

            # RIDGE

            External.rc_model, External.rc_auroc, External.rc_accuracy, External.rc_sensitivity, External.rc_spec = \
                ridge(train_features, train_labels, test_features, test_labels)

            External.rc_auroc_cv += External.rc_auroc
            External.rc_accuracy_cv += External.rc_accuracy
            External.rc_sensitivity_cv += External.rc_sensitivity
            External.rc_spec_cv += External.rc_spec

            count += 0.3
            self.countChanged.emit(count)
            text = "Ridge Classifier Performance: \n-----\n" \
                   "AUROC: " + str(External.rc_auroc) + \
                   "\nAccuracy: " + str(External.rc_accuracy) + \
                   "%.\nSensitivity: " + str(External.rc_sensitivity) + \
                   "%. \nSpecificity: " + str(External.rc_spec) + "%. \n-----\n"
            self.textChanged.emit(text)

        rc_object = [External.rc_model, External.rc_auroc_cv / 10, External.rc_accuracy_cv / 10,
                     External.rc_sensitivity_cv / 10, External.rc_spec_cv / 10]
        text = "Ridge Classifier Mean Performance: \n-----\n" \
               "AUROC: " + str(External.rc_auroc_cv / 10) + \
               "\nAccuracy: " + str(External.rc_accuracy_cv / 10) + \
               "%.\nSensitivity: " + str(External.rc_sensitivity_cv / 10) + \
               "%. \nSpecificity: " + str(External.rc_spec_cv / 10) + "%. \n-----\n"
        self.textChanged.emit("")  # clear result box
        self.textChanged.emit(text)

        sgd_object = [External.sgd_model, External.sgd_auroc_cv / 10,
                      External.sgd_accuracy_cv / 10, External.sgd_sensitivity_cv / 10, External.sgd_spec_cv / 10]

        text = "Stochastic Gradient Descent Mean Performance: \n-----\n" \
               "AUROC: " + str(External.sgd_auroc_cv / 10) + \
               "\nAccuracy: " + str(External.sgd_accuracy_cv / 10) + \
               "%.\nSensitivity: " + str(External.sgd_sensitivity_cv / 10) + \
               "%. \nSpecificity: " + str(External.sgd_spec_cv / 10) + "%. \n-----\n"
        self.textChanged.emit(text)

        lda_object = [External.lda_model, External.lda_auroc_cv / 10,
                      External.lda_accuracy_cv / 10, External.lda_sensitivity_cv / 10, External.lda_spec_cv / 10]

        text = "Linear Discriminant Analysis Mean Performance: \n-----\n" \
               "AUROC: " + str(External.lda_auroc_cv / 10) + \
               "\nAccuracy: " + str(External.lda_accuracy_cv / 10) + \
               "%.\nSensitivity: " + str(External.lda_sensitivity_cv / 10) + \
               "%. \nSpecificity: " + str(External.lda_spec_cv / 10) + "%. \n-----\n"
        self.textChanged.emit(text)

        tsr_object = [External.tsr_model, External.tsr_auroc_cv / 10, External.tsr_accuracy_cv / 10,
                      External.tsr_sensitivity_cv / 10, External.tsr_spec_cv / 10]

        text = "Theil-Sen Regressor Mean Performance: \n-----\n" \
               "AUROC: " + str(External.tsr_auroc_cv / 10) + \
               "\nAccuracy: " + str(External.tsr_accuracy_cv / 10) + \
               "%.\nSensitivity: " + str(External.tsr_sensitivity_cv / 10) + \
               "%. \nSpecificity: " + str(External.tsr_spec_cv / 10) + "%. \n-----\n"
        self.textChanged.emit(text)

        lrr_object = [External.lrr_model, External.lrr_auroc_cv / 10, External.lrr_accuracy_cv / 10,
                      External.lrr_sensitivity_cv / 10, External.lrr_spec_cv / 10]

        text = "Linear Regression Mean Performance: \n-----\n" \
               "AUROC: " + str(External.lrr_auroc_cv / 10) + \
               "\nAccuracy: " + str(External.lrr_accuracy_cv / 10) + \
               "%.\nSensitivity: " + str(External.lrr_sensitivity_cv / 10) + \
               "%. \nSpecificity: " + str(External.lrr_spec_cv / 10) + "%. \n-----\n"
        self.textChanged.emit(text)

        lr_object = [External.lr_model, External.lr_auroc_cv / 10, External.lr_accuracy_cv / 10,
                     External.lr_sensitivity_cv / 10, External.lr_spec_cv / 10]

        text = "Logistic Regression Mean Performance: \n-----\n" \
               "AUROC: " + str(External.lr_auroc_cv / 10) + \
               "\nAccuracy: " + str(External.lr_accuracy_cv / 10) + \
               "%.\nSensitivity: " + str(External.lr_sensitivity_cv / 10) + \
               "%. \nSpecificity: " + str(External.lr_spec_cv / 10) + "%. \n-----\n"
        self.textChanged.emit(text)

        bern_object = [External.bernoulli_model, External.bernoulli_auroc_cv / 10, External.bernoulli_accuracy_cv / 10,
                       External.bernoulli_sensitivity_cv / 10, External.bernoulli_spec_cv / 10]

        text = "Bernoulli Naive Bayes Mean Performance: \n-----\n" \
               "AUROC: " + str(External.bernoulli_auroc_cv / 10) + \
               "\nAccuracy: " + str(External.bernoulli_accuracy_cv / 10) + \
               "%.\nSensitivity: " + str(External.bernoulli_sensitivity_cv / 10) + \
               "%. \nSpecificity: " + str(External.bernoulli_spec_cv / 10) + "%. \n-----\n"
        self.textChanged.emit(text)

        bag_naive = [External.bagging_naive_model, External.bagging_naive_auroc_cv / 10,
                     External.bagging_naive_accuracy_cv / 10, External.bagging_naive_sensitivity_cv / 10,
                     External.bagging_naive_spec_cv / 10]

        text = "Bagging Naive Bayes Mean Performance: \n-----\n" \
               "AUROC: " + str(External.bagging_naive_auroc_cv / 10) + \
               "\nAccuracy: " + str(External.bagging_naive_accuracy_cv / 10) + \
               "%.\nSensitivity: " + str(External.bagging_naive_sensitivity_cv / 10) + \
               "%. \nSpecificity: " + str(External.bagging_naive_spec_cv / 10) + "%. \n-----\n"
        self.textChanged.emit(text)

        rgf_object = \
            [External.rgf_model, External.rgf_auroc_cv / 10,
             External.rgf_accuracy_cv / 10, External.rgf_sensitivity_cv / 10, External.rgf_spec_cv / 10]

        text = "Regularized Greedy Forests Mean Performance: \n-----\n" \
               "AUROC: " + str(External.rgf_auroc_cv / 10) + \
               "\nAccuracy: " + str(External.rgf_accuracy_cv / 10) + \
               "%.\nSensitivity: " + str(External.rgf_sensitivity_cv / 10) + \
               "%. \nSpecificity: " + str(External.rgf_spec_cv / 10) + "%. \n-----\n"
        self.textChanged.emit(text)

        extreme_rf_object = [External.extreme_rf_model, External.extreme_rf_auroc_cv / 10,
                             External.extreme_rf_accuracy_cv / 10,
                             External.extreme_rf_sensitivity_cv / 10, External.extreme_rf_spec_cv / 10]

        text = "Extremely Randomised Forest Mean Performance: \n-----\n" \
               "AUROC: " + str(External.extreme_rf_auroc_cv / 10) + \
               "\nAccuracy: " + str(External.extreme_rf_accuracy_cv / 10) + \
               "%.\nSensitivity: " + str(External.extreme_rf_sensitivity_cv / 10) + \
               "%. \nSpecificity: " + str(External.extreme_rf_spec_cv / 10) + "%. \n-----\n"
        self.textChanged.emit(text)

        rf_object = \
            [External.rf_model, External.rf_auroc_cv / 10, External.rf_accuracy_cv / 10,
             External.rf_sensitivity_cv / 10, External.rf_spec_cv / 10]

        text = "Random Forest Mean Performance: \n-----\n" \
               "AUROC: " + str(External.rf_auroc_cv / 10) + \
               "\nAccuracy: " + str(External.rf_accuracy_cv / 10) + \
               "%.\nSensitivity: " + str(External.rf_sensitivity_cv / 10) + \
               "%. \nSpecificity: " + str(External.rf_spec_cv / 10) + "%. \n-----\n"
        self.textChanged.emit(text)

        dtc_object = \
            [External.dtc_model, External.dtc_auroc_cv / 10,
             External.dtc_accuracy_cv / 10, External.dtc_sensitivity_cv / 10, External.dtc_spec_cv / 10]

        text = "Decision Tree Classifier Mean Performance: \n-----\n" \
               "AUROC: " + str(External.dtc_auroc_cv / 10) + \
               "\nAccuracy: " + str(External.dtc_accuracy_cv / 10) + \
               "%.\nSensitivity: " + str(External.dtc_sensitivity_cv / 10) + \
               "%. \nSpecificity: " + str(External.dtc_spec_cv / 10) + "%. \n-----\n"
        self.textChanged.emit(text)

        nn_object = \
            [External.nn_model, External.nn_auroc_cv / 10, External.nn_accuracy_cv / 10,
             External.nn_sensitivity_cv / 10, External.nn_spec_cv / 10]

        text = "Neural Network Mean Performance: \n-----\n" \
               "AUROC: " + str(External.nn_auroc_cv / 10) + \
               "\nAccuracy: " + str(External.nn_accuracy_cv / 10) + \
               "%.\nSensitivity: " + str(External.nn_sensitivity_cv / 10) + \
               "%. \nSpecificity: " + str(External.nn_spec_cv / 10) + "%. \n-----\n"
        self.textChanged.emit(text)

        elm_object = \
            [External.elm_model, External.elm_auroc_cv / 10, External.elm_accuracy_cv / 10,
             External.elm_sensitivity_cv / 10, External.elm_spec_cv / 10]

        text = "Extreme Learning Machine Mean Performance: \n-----\n" \
               "AUROC: " + str(External.elm_auroc_cv / 10) + \
               "\nAccuracy: " + str(External.elm_accuracy_cv / 10) + \
               "%.\nSensitivity: " + str(External.elm_sensitivity_cv / 10) + \
               "%. \nSpecificity: " + str(External.elm_spec_cv / 10) + "%. \n-----\n"
        self.textChanged.emit(text)

        naive_object = \
            [External.naive_model, External.naive_auroc_cv / 10, External.naive_accuracy_cv / 10,
             External.naive_sensitivity_cv / 10, External.naive_spec_cv / 10]

        text = "Naive Bayes Mean Performance: \n-----\n" \
               "AUROC: " + str(External.naive_auroc_cv / 10) + \
               "\nAccuracy: " + str(External.naive_accuracy_cv / 10) + \
               "%.\nSensitivity: " + str(External.naive_sensitivity_cv / 10) + \
               "%. \nSpecificity: " + str(External.naive_spec_cv / 10) + "%. \n-----\n"
        self.textChanged.emit(text)

        lsvc_object = [External.lsvc_model, External.lsvc_auroc_cv / 10, External.lsvc_accuracy_cv / 10,
                       External.lsvc_sensitivity_cv / 10, External.lsvc_spec_cv / 10]

        text = "Linear SVC Mean Performance: \n-----\n" \
               "AUROC: " + str(External.lsvc_auroc_cv / 10) + \
               "\nAccuracy: " + str(External.lsvc_accuracy_cv / 10) + \
               "%.\nSensitivity: " + str(External.lsvc_sensitivity_cv / 10) + \
               "%. \nSpecificity: " + str(External.lsvc_spec_cv / 10) + "%. \n-----\n"
        self.textChanged.emit(text)

        bag_svm_object = [External.bagging_svm_model, External.bagging_svm_auroc_cv / 10,
                          External.bagging_svm_accuracy_cv / 10,
                          External.bagging_svm_sensitivity_cv / 10, External.bagging_svm_spec_cv / 10]

        text = "Bagging Support Vector Machine Mean Performance: \n-----\n" \
               "AUROC: " + str(External.bagging_svm_auroc_cv / 10) + \
               "\nAccuracy: " + str(External.bagging_svm_accuracy_cv / 10) + \
               "%.\nSensitivity: " + str(External.bagging_svm_sensitivity_cv / 10) + \
               "%. \nSpecificity: " + str(External.bagging_svm_spec_cv / 10) + "%. \n-----\n"
        self.textChanged.emit(text)

        svm_object = [External.svm_model, External.svm_auroc_cv / 10, External.svm_accuracy_cv / 10,
                      External.svm_sensitivity_cv / 10, External.svm_spec_cv / 10]

        text = "Support Vector Machine Mean Performance: \n-----\n" \
               "AUROC: " + str(External.svm_auroc_cv / 10) + \
               "\nAccuracy: " + str(External.svm_accuracy_cv / 10) + \
               "%.\nSensitivity: " + str(External.svm_sensitivity_cv / 10) + \
               "%. \nSpecificity: " + str(External.svm_spec_cv / 10) + "%. \n-----\n"
        self.textChanged.emit(text)

        bag_k_object = [External.bagging_k_model, External.bagging_k_nearest_auroc_cv / 10,
                        External.bagging_k_nearest_accuracy_cv / 10,
                        External.bagging_k_nearest_sensitivity_cv / 10, External.bagging_k_nearest_spec_cv / 10]

        text = "Bagging K-Nearest Neighbors Mean Performance: \n-----\n" \
               "AUROC: " + str(External.bagging_k_nearest_auroc_cv / 10) + \
               "\nAccuracy: " + str(External.bagging_k_nearest_accuracy_cv / 10) + \
               "%.\nSensitivity: " + str(External.bagging_k_nearest_sensitivity_cv / 10) + \
               "%. \nSpecificity: " + str(External.bagging_k_nearest_spec_cv / 10) + "%. \n-----\n"
        self.textChanged.emit(text)

        k_object = [External.k_model, External.k_auroc_cv / 10, External.k_accuracy_cv / 10,
                    External.k_sensitivity_cv / 10,
                    External.k_spec_cv / 10]

        text = "K-Nearest Neighbors Mean Performance: \n-----\n" \
               "AUROC: " + str(External.k_auroc_cv / 10) + \
               "\nAccuracy: " + str(External.k_accuracy_cv / 10) + \
               "%.\nSensitivity: " + str(External.k_sensitivity_cv / 10) + \
               "%. \nSpecificity: " + str(External.k_spec_cv / 10) + "%. \n-----\n"
        self.textChanged.emit(text)

        gbm_object = [External.gbm_model, External.gbm_auroc_cv / 10, External.gbm_accuracy_cv / 10,
                      External.gbm_sensitivity_cv / 10, External.gbm_spec_cv / 10]

        text = "Light Gradient Boosting Mean Performance: \n-----\n" \
               "AUROC: " + str(External.xgb_auroc_cv / 10) + \
               "\nAccuracy: " + str(External.xgb_accuracy_cv / 10) + \
               "%.\nSensitivity: " + str(External.xgb_sensitivity_cv / 10) + \
               "%. \nSpecificity: " + str(External.xgb_spec_cv / 10) + "%. \n-----\n"
        self.textChanged.emit(text)

        ada_object = [External.adaboost_model, External.adaboost_auroc_cv / 10, External.adaboost_accuracy_cv / 10,
                      External.adaboost_sensitivity_cv / 10, External.adaboost_spec_cv / 10]

        text = "AdaBoost with Decision Tree Base Mean Performance: \n-----\n" \
               "AUROC: " + str(External.adaboost_auroc_cv / 10) + \
               "\nAccuracy: " + str(External.adaboost_accuracy_cv / 10) + \
               "%.\nSensitivity: " + str(External.adaboost_sensitivity_cv / 10) + \
               "%. \nSpecificity: " + str(External.adaboost_spec_cv / 10) + "%. \n-----\n"
        self.textChanged.emit(text)

        xgb_object = [External.xgb_model, External.xgb_auroc_cv / 10, External.xgb_accuracy_cv / 10,
                      External.xgb_sensitivity_cv / 10, External.xgb_spec_cv / 10]

        text = "XGradient Boosting Mean Performance: \n-----\n" \
               "AUROC: " + str(External.xgb_auroc_cv / 10) + \
               "\nAccuracy: " + str(External.xgb_accuracy_cv / 10) + \
               "%.\nSensitivity: " + str(External.xgb_sensitivity_cv / 10) + \
               "%. \nSpecificity: " + str(External.xgb_spec_cv / 10) + "%. \n-----\n"
        self.textChanged.emit(text)

        External.all_models = [nn_object, rf_object, extreme_rf_object, dtc_object, rgf_object, ada_object, xgb_object,
                               gbm_object, k_object, bag_k_object, svm_object, bag_svm_object, lsvc_object,
                               naive_object, bag_naive, bern_object, lr_object, lrr_object, tsr_object,
                               lda_object, sgd_object, rc_object]

        External.all_models = np.asarray(External.all_models)

        # re-enable the start process button and the result options combo box
        self.stateChanged.emit(1)

        # To display data visually, reverse one-hot and re-add dropped columns (store in a list numpy array)


# noinspection PyCallByClass
class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        flags = Qt.WindowTitleHint | Qt.WindowSystemMenuHint | Qt.WindowCloseButtonHint
        flags = flags | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint

        QtWidgets.QMainWindow.__init__(self, flags=flags)
        Ui_MainWindow.__init__(self)
        self.calc = External()
        self.setupUi(self)

        # Import Data Menu Item
        self.actionImport_Player_Activity_Data.setShortcut("CTRL+O")
        self.actionImport_Player_Activity_Data.triggered.connect(self.open_file_name_dialog)
        self.actionImport_Player_RG_Events_Data_Set.triggered.connect(self.open_file_name_dialog)
        self.actionImport_Player_Descriptions_Data_Set.triggered.connect(self.open_file_name_dialog)
        self.actionImport_Data_Set_to_Predict.triggered.connect(self.open_file_name_dialog)
        self.actionImport_IDs_Data_Set.triggered.connect(self.open_file_name_dialog)

        # Exit Menu Item
        self.actionExit_Application.setShortcut("CTRL+E")
        self.actionExit_Application.triggered.connect(self.close_application)

        # Theme Change Buttons
        self.actionClassic_Gray.triggered.connect(self.classic_gray)
        self.actionDark_Orange.triggered.connect(self.dark_orange)
        self.actionDark_Blue.triggered.connect(self.dark_orange_gradient)

        # File Path Label
        self.act_file_path_label.setText("No Data File Uploaded.")
        self.rg_file_path_label.setText("No Data File Uploaded.")
        self.desc_file_path_label.setText("No Data File Uploaded.")
        self.pred_file_path.setText("No Data File Uploaded.")
        self.id_file_path.setText("No Data File Uploaded.")

        # Change Data Set Buttons Click
        self.change_activity_data_btn.clicked.connect(self.open_file_name_dialog)
        self.change_rg_data_btn.clicked.connect(self.open_file_name_dialog)
        self.change_desc_data_btn.clicked.connect(self.open_file_name_dialog)
        self.pred_file_btn.clicked.connect(self.open_file_name_dialog)
        self.id_file_btn.clicked.connect(self.open_file_name_dialog)

        # Results Box Set to Read Only and Hidden
        self.results_box.close()
        self.results_box.setReadOnly(True)
        self.pred_results_box.close()
        self.pred_results_box.setReadOnly(True)

        # Results Box Show/Hide
        self.train_test_result_btn.setEnabled(True)
        self.train_test_result_btn.clicked.connect(self.show_train_results)
        self.show_results_btn.clicked.connect(self.show_prediction_results)

        self.start_train_process_btn.clicked.connect(self.start_process)
        self.start_train_process_btn.setEnabled(False)

        self.clear_results_btn.clicked.connect(self.clear_result)

        self.save_results_btn.clicked.connect(self.save_result)

        self.start_pred.clicked.connect(self.start_analyse)

        self.train_models.setChecked(True)
        self.train_models.clicked.connect(self.train_models_click)
        self.pretrained.clicked.connect(self.pre_trained_click)

    def pre_trained_click(self):
        self.change_desc_data_btn.setEnabled(False)
        self.change_activity_data_btn.setEnabled(False)
        self.change_rg_data_btn.setEnabled(False)
        self.start_train_process_btn.setEnabled(False)
        self.train_test_result_btn.setEnabled(False)
        self.result_type_combo.setEnabled(False)
        self.actionImport_Player_Descriptions_Data_Set.setEnabled(False)
        self.actionImport_Player_Activity_Data.setEnabled(False)
        self.actionImport_Player_RG_Events_Data_Set.setEnabled(False)

        self.pred_file_btn.setEnabled(True)
        self.actionImport_Data_Set_to_Predict.setEnabled(True)
        self.actionImport_IDs_Data_Set.setEnabled(True)
        self.id_file_btn.setEnabled(True)

    def train_models_click(self):
        self.change_desc_data_btn.setEnabled(True)
        self.change_activity_data_btn.setEnabled(True)
        self.change_rg_data_btn.setEnabled(True)
        self.train_test_result_btn.setEnabled(True)
        self.actionImport_Player_Descriptions_Data_Set.setEnabled(True)
        self.actionImport_Player_Activity_Data.setEnabled(True)
        self.actionImport_Player_RG_Events_Data_Set.setEnabled(True)

        self.pred_file_btn.setEnabled(False)
        self.start_pred.setEnabled(False)
        self.id_file_btn.setEnabled(False)
        self.actionImport_IDs_Data_Set.setEnabled(False)

    def start_analyse(self):
        max_auroc, max_acc, max_sen, max_spec, model_obj = 0, 0, 0, 0, []
        result_option = str(self.result_type_combo.currentText())
        predict = np.array(df4)

        print(predict)

        if self.pretrained.isChecked():
            results = []
            path = 'trained_models/'
            pre_trained_models = \
                [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

            make_keras_picklable()

            for model in pre_trained_models:
                path = 'trained_models/' + model
                loaded_model = pickle.load(open(path, 'rb'))
                pred = loaded_model.predict(predict)
                pred = (pred > 0.5) * 1
                predictions = pred.ravel()
                results.append(predictions)

            results = np.array(results)
            results.flatten()
            print(results)

            res = np.sum(results, axis=0).tolist()
            res_perc = [((i / len(pre_trained_models)) * 100) for i in res]

            at_risk = np.array(df5)
            at_risk = at_risk.reshape(-1)

            at_risk = pd.DataFrame({'UserID': at_risk,
                                    'At-Risk Probability Across All Models (%)': res_perc})

            at_risk = at_risk[['UserID', 'At-Risk Probability Across All Models (%)']]

            text = at_risk.to_string()
            self.pred_results_box.setText(text)
            self.save_results_btn.setEnabled(True)
        else:
            if result_option == "Best AUROC":
                for x in External.all_models:
                    if x[1] > max_auroc:
                        max_auroc = x[1]
                        model_obj = x
                predictions = model_obj[0].predict(predict)
                predictions = (predictions > 0.5) * 1

                at_risk = pd.DataFrame({'UserID': External.predict_ids, 'Prediction': predictions})
                at_risk = at_risk[at_risk.Prediction != 0]
                at_risk.drop('Prediction', 1, inplace=True)
                at_risk = np.array(at_risk)
                at_risk = at_risk.reshape(-1)
                text = "UserID of Potential At Risk Players from Uploaded Data Set:\n" \
                       "AUROC value for training of the Selected Model: " + str(max_auroc) + "\n--------\n"
                text += '\n'.join(map(str, at_risk))[1:-1]
                self.pred_results_box.setText(text)

            elif result_option == "Best Overall Accuracy":
                for x in External.all_models:
                    if x[2] > max_acc:
                        max_acc = x[2]
                        model_obj = x
                predictions = model_obj[0].predict(predict)
                predictions = (predictions > 0.5) * 1

                at_risk = pd.DataFrame({'UserID': External.predict_ids, 'Prediction': predictions})
                at_risk = at_risk[at_risk.Prediction != 0]
                at_risk.drop('Prediction', 1, inplace=True)
                at_risk = np.array(at_risk)
                at_risk = at_risk.reshape(-1)
                text = "UserID of Potential At Risk Players from Uploaded Data Set:\n" \
                       "Accuracy for training of Model Selected: " + str(max_acc) + "%.\n--------\n"
                text += '\n'.join(map(str, at_risk))[1:-1]
                self.pred_results_box.setText(text)

            elif result_option == "Best Sensitivity":
                for x in External.all_models:
                    if x[3] > max_sen:
                        max_sen = x[3]
                        model_obj = x
                predictions = model_obj[0].predict(predict)
                predictions = (predictions > 0.5) * 1

                at_risk = pd.DataFrame({'UserID': External.predict_ids, 'Prediction': predictions})
                at_risk = at_risk[at_risk.Prediction != 0]
                at_risk.drop('Prediction', 1, inplace=True)
                at_risk = np.array(at_risk)
                at_risk = at_risk.reshape(-1)
                text = "UserID of Potential At Risk Players from Uploaded Data Set:\n" \
                       "Sensitivity for training of Model Selected: " + str(max_sen) + "%.\n--------\n"
                text += '\n'.join(map(str, at_risk))[1:-1]
                self.pred_results_box.setText(text)

            elif result_option == "Best Specificity":
                for x in External.all_models:
                    if x[4] > max_spec:
                        max_spec = x[4]
                        model_obj = x
                predictions = model_obj[0].predict(predict)
                predictions = (predictions > 0.5) * 1

                at_risk = pd.DataFrame({'UserID': External.predict_ids, 'Prediction': predictions})
                at_risk = at_risk[at_risk.Prediction != 0]
                at_risk.drop('Prediction', 1, inplace=True)
                at_risk = np.array(at_risk)
                at_risk = at_risk.reshape(-1)
                text = "UserID of Potential At Risk Players from Uploaded Data Set:\n" \
                       "Specificity for training of Model Selected: " + str(max_spec) + "%.\n--------\n"
                text += '\n'.join(map(str, at_risk))[1:-1]
                self.pred_results_box.setText(text)

    def save_result(self):
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Results', os.getenv('HOME'))
        if file_name != "":
            with open(file_name, 'w') as file:
                text = self.pred_results_box.toPlainText()
                file.write(text)
                file.close()

    def clear_result(self):
        self.results_box.setText("")
        self.pred_results_box.setText("")

    # noinspection PyArgumentList
    def onCountChanged(self, value):
        self.process_progress_bar.setValue(value)
        if value == 100:
            QApplication.restoreOverrideCursor()
            self.save_results_btn.setEnabled(True)
            self.actionExport_Results.setEnabled(True)
            QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.information(self, 'Model Training',
                                              "All models were successfully trained and validated.\n"
                                              "Please see the results tab for further details.",
                                              QtWidgets.QMessageBox.Ok)

    def onTextChanged(self, value):
        self.results_box.setText(self.results_box.toPlainText() + value)

    def onStateChanged(self, value):
        # self.start_train_process_btn.setEnabled(value)
        self.pred_file_btn.setEnabled(value)
        self.actionImport_Data_Set_to_Predict.setEnabled(value)

    @staticmethod
    def classic_gray():
        app.setStyleSheet("""""")

    @staticmethod
    def dark_orange():
        app.setStyleSheet(
            """QToolTip{border:1px solid black;background-color:#ffa02f;padding:1px;border-radius:3px;opacity:100;}QWidget{color:#b1b1b1;background-color:#323232;}QWidget:item:hover{background-color:QLinearGradient( x1:0, y1:0, x2:0, y2:1, stop:0 #ffa02f, stop:1 #ca0619);color:#000000;}QWidget:item:selected{background-color:QLinearGradient( x1:0, y1:0, x2:0, y2:1, stop:0 #ffa02f, stop:1 #d7801a);}QMenuBar::item{background:transparent;}QMenuBar::item:selected{background:transparent;border:1px solid #ffaa00;}QMenuBar::item:pressed{background:#444;border:1px solid #000;background-color:QLinearGradient( x1:0, y1:0, x2:0, y2:1, stop:1 #212121, stop:0.4 #343434 );margin-bottom:-1px;padding-bottom:1px;}QMenu{border:1px solid #000;}QMenu::item{padding:2px 20px 2px 20px;}QMenu::item:selected{color:#000000;}QWidget:disabled{color:#404040;background-color:#323232;}QAbstractItemView{background-color:QLinearGradient( x1:0, y1:0, x2:0, y2:1, stop:0 #4d4d4d, stop:0.1 #646464, stop:1 #5d5d5d);}QWidget:focus{}QLineEdit{background-color:QLinearGradient( x1:0, y1:0, x2:0, y2:1, stop:0 #4d4d4d, stop:0 #646464, stop:1 #5d5d5d);padding:1px;border-style:solid;border:1px solid #1e1e1e;border-radius:5;}QPushButton{color:#b1b1b1;background-color:QLinearGradient( x1:0, y1:0, x2:0, y2:1, stop:0 #565656, stop:0.1 #525252, stop:0.5 #4e4e4e, stop:0.9 #4a4a4a, stop:1 #464646);border-width:1px;border-color:#1e1e1e;border-style:solid;border-radius:6;padding:3px;font-size:12px;padding-left:5px;padding-right:5px;}QPushButton:pressed{background-color:QLinearGradient( x1:0, y1:0, x2:0, y2:1, stop:0 #2d2d2d, stop:0.1 #2b2b2b, stop:0.5 #292929, stop:0.9 #282828, stop:1 #252525);}QComboBox{selection-background-color:#ffaa00;background-color:QLinearGradient( x1:0, y1:0, x2:0, y2:1, stop:0 #565656, stop:0.1 #525252, stop:0.5 #4e4e4e, stop:0.9 #4a4a4a, stop:1 #464646);border-style:solid;border:1px solid #1e1e1e;border-radius:5;}QComboBox:hover,QPushButton:hover{border:2px solid QLinearGradient( x1:0, y1:0, x2:0, y2:1, stop:0 #ffa02f, stop:1 #d7801a);}QComboBox:on{padding-top:3px;padding-left:4px;background-color:QLinearGradient( x1:0, y1:0, x2:0, y2:1, stop:0 #2d2d2d, stop:0.1 #2b2b2b, stop:0.5 #292929, stop:0.9 #282828, stop:1 #252525);selection-background-color:#ffaa00;}QComboBox QAbstractItemView{border:2px solid darkgray;selection-background-color:QLinearGradient( x1:0, y1:0, x2:0, y2:1, stop:0 #ffa02f, stop:1 #d7801a);}QComboBox::drop-down{subcontrol-origin:padding;subcontrol-position:top right;width:15px;border-left-width:0px;border-left-color:darkgray;border-left-style:solid;border-top-right-radius:3px;border-bottom-right-radius:3px;}QComboBox::down-arrow{image:url(:/down_arrow.png);}QGroupBox:focus{border:2px solid QLinearGradient( x1:0, y1:0, x2:0, y2:1, stop:0 #ffa02f, stop:1 #d7801a);}QTextEdit:focus{border:2px solid QLinearGradient( x1:0, y1:0, x2:0, y2:1, stop:0 #ffa02f, stop:1 #d7801a);}QScrollBar:horizontal{border:1px solid #222222;background:QLinearGradient( x1:0, y1:0, x2:0, y2:1, stop:0.0 #121212, stop:0.2 #282828, stop:1 #484848);height:7px;margin:0px 16px 0 16px;}QScrollBar::handle:horizontal{background:QLinearGradient( x1:0, y1:0, x2:1, y2:0, stop:0 #ffa02f, stop:0.5 #d7801a, stop:1 #ffa02f);min-height:20px;border-radius:2px;}QScrollBar::add-line:horizontal{border:1px solid #1b1b19;border-radius:2px;background:QLinearGradient( x1:0, y1:0, x2:1, y2:0, stop:0 #ffa02f, stop:1 #d7801a);width:14px;subcontrol-position:right;subcontrol-origin:margin;}QScrollBar::sub-line:horizontal{border:1px solid #1b1b19;border-radius:2px;background:QLinearGradient( x1:0, y1:0, x2:1, y2:0, stop:0 #ffa02f, stop:1 #d7801a);width:14px;subcontrol-position:left;subcontrol-origin:margin;}QScrollBar::right-arrow:horizontal, QScrollBar::left-arrow:horizontal{border:1px solid black;width:1px;height:1px;background:white;}QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal{background:none;}QScrollBar:vertical{background:QLinearGradient( x1:0, y1:0, x2:1, y2:0, stop:0.0 #121212, stop:0.2 #282828, stop:1 #484848);width:7px;margin:16px 0 16px 0;border:1px solid #222222;}QScrollBar::handle:vertical{background:QLinearGradient( x1:0, y1:0, x2:0, y2:1, stop:0 #ffa02f, stop:0.5 #d7801a, stop:1 #ffa02f);min-height:20px;border-radius:2px;}QScrollBar::add-line:vertical{border:1px solid #1b1b19;border-radius:2px;background:QLinearGradient( x1:0, y1:0, x2:0, y2:1, stop:0 #ffa02f, stop:1 #d7801a);height:14px;subcontrol-position:bottom;subcontrol-origin:margin;}QScrollBar::sub-line:vertical{border:1px solid #1b1b19;border-radius:2px;background:QLinearGradient( x1:0, y1:0, x2:0, y2:1, stop:0 #d7801a, stop:1 #ffa02f);height:14px;subcontrol-position:top;subcontrol-origin:margin;}QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical{border:1px solid black;width:1px;height:1px;background:white;}QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical{background:none;}QTextEdit{background-color:#242424;}QPlainTextEdit{background-color:#242424;}QHeaderView::section{background-color:QLinearGradient(x1:0, y1:0, x2:0, y2:1, stop:0 #616161, stop:0.5 #505050, stop:0.6 #434343, stop:1 #656565);color:white;padding-left:4px;border:1px solid #6c6c6c;}QCheckBox:disabled{color:#414141;}QDockWidget::title{text-align:center;spacing:3px;background-color:QLinearGradient(x1:0, y1:0, x2:0, y2:1, stop:0 #323232, stop:0.5 #242424, stop:1 #323232);}QDockWidget::close-button, QDockWidget::float-button{text-align:center;spacing:1px;background-color:QLinearGradient(x1:0, y1:0, x2:0, y2:1, stop:0 #323232, stop:0.5 #242424, stop:1 #323232);}QDockWidget::close-button:hover, QDockWidget::float-button:hover{background:#242424;}QDockWidget::close-button:pressed, QDockWidget::float-button:pressed{padding:1px -1px -1px 1px;}QMainWindow::separator{background-color:QLinearGradient(x1:0, y1:0, x2:0, y2:1, stop:0 #161616, stop:0.5 #151515, stop:0.6 #212121, stop:1 #343434);color:white;padding-left:4px;border:1px solid #4c4c4c;spacing:3px;}QMainWindow::separator:hover{background-color:QLinearGradient(x1:0, y1:0, x2:0, y2:1, stop:0 #d7801a, stop:0.5 #b56c17 stop:1 #ffa02f);color:white;padding-left:4px;border:1px solid #6c6c6c;spacing:3px;}QToolBar::handle{spacing:3px;background:url(:/images/handle.png);}QMenu::separator{height:2px;background-color:QLinearGradient(x1:0, y1:0, x2:0, y2:1, stop:0 #161616, stop:0.5 #151515, stop:0.6 #212121, stop:1 #343434);color:white;padding-left:4px;margin-left:10px;margin-right:5px;}QProgressBar{border:2px solid grey;border-radius:5px;text-align:center;}QProgressBar::chunk{background-color:#d7801a;width:2.15px;margin:0.5px;}QTabBar::tab{color:#b1b1b1;border:1px solid #444;border-bottom-style:none;background-color:#323232;padding-left:10px;padding-right:10px;padding-top:3px;padding-bottom:2px;margin-right:-1px;}QTabWidget::pane{border:1px solid #444;top:1px;}QTabBar::tab:last{margin-right:0;border-top-right-radius:3px;}QTabBar::tab:first:!selected{margin-left:0px;border-top-left-radius:3px;}QTabBar::tab:!selected{color:#b1b1b1;border-bottom-style:solid;margin-top:3px;background-color:QLinearGradient(x1:0, y1:0, x2:0, y2:1, stop:1 #212121, stop:.4 #343434);}QTabBar::tab:selected{border-top-left-radius:3px;border-top-right-radius:3px;margin-bottom:0px;}QTabBar::tab:!selected:hover{border-top-left-radius:3px;border-top-right-radius:3px;background-color:QLinearGradient(x1:0, y1:0, x2:0, y2:1, stop:1 #212121, stop:0.4 #343434, stop:0.2 #343434, stop:0.1 #ffaa00);}QRadioButton::indicator:checked, QRadioButton::indicator:unchecked{color:#b1b1b1;background-color:#323232;border:1px solid #b1b1b1;border-radius:6px;}QRadioButton::indicator:checked{background-color:qradialgradient( cx:0.5, cy:0.5, fx:0.5, fy:0.5, radius:1.0, stop:0.25 #ffaa00, stop:0.3 #323232 );}QCheckBox::indicator{color:#b1b1b1;background-color:#323232;border:1px solid #b1b1b1;width:9px;height:9px;}QRadioButton::indicator{border-radius:6px;}QRadioButton::indicator:hover, QCheckBox::indicator:hover{border:1px solid #ffaa00;}QCheckBox::indicator:checked{image:url(:/images/checkbox.png);}QCheckBox::indicator:disabled, QRadioButton::indicator:disabled{border:1px solid #444;}""")

    @staticmethod
    def dark_orange_gradient():
        app.setStyleSheet(
            """QToolTip{border:1px solid black;background-color:#436775 ;padding:1px;border-radius:3px;opacity:100;}QWidget{color:#b1b1b1;background-color:#323232;}QWidget:item:hover{background-color:QLinearGradient( x1:0, y1:0, x2:0, y2:1, stop:0 #436775 , stop:1 #ca0619);color:#000000;}QWidget:item:selected{background-color:QLinearGradient( x1:0, y1:0, x2:0, y2:1, stop:0 #436775 , stop:1 #d7801a);}QMenuBar::item{background:transparent;}QMenuBar::item:selected{background:transparent;border:1px solid #ffaa00;}QMenuBar::item:pressed{background:#444;border:1px solid #000;background-color:QLinearGradient( x1:0, y1:0, x2:0, y2:1, stop:1 #212121, stop:0.4 #343434 );margin-bottom:-1px;padding-bottom:1px;}QMenu{border:1px solid #000;}QMenu::item{padding:2px 20px 2px 20px;}QMenu::item:selected{color:#000000;}QWidget:disabled{color:#404040;background-color:#323232;}QAbstractItemView{background-color:QLinearGradient( x1:0, y1:0, x2:0, y2:1, stop:0 #4d4d4d, stop:0.1 #646464, stop:1 #5d5d5d);}QWidget:focus{}QLineEdit{background-color:QLinearGradient( x1:0, y1:0, x2:0, y2:1, stop:0 #4d4d4d, stop:0 #646464, stop:1 #5d5d5d);padding:1px;border-style:solid;border:1px solid #1e1e1e;border-radius:5;}QPushButton{color:#b1b1b1;background-color:QLinearGradient( x1:0, y1:0, x2:0, y2:1, stop:0 #565656, stop:0.1 #525252, stop:0.5 #4e4e4e, stop:0.9 #4a4a4a, stop:1 #464646);border-width:1px;border-color:#1e1e1e;border-style:solid;border-radius:6;padding:3px;font-size:12px;padding-left:5px;padding-right:5px;}QPushButton:pressed{background-color:QLinearGradient( x1:0, y1:0, x2:0, y2:1, stop:0 #2d2d2d, stop:0.1 #2b2b2b, stop:0.5 #292929, stop:0.9 #282828, stop:1 #252525);}QComboBox{selection-background-color:#ffaa00;background-color:QLinearGradient( x1:0, y1:0, x2:0, y2:1, stop:0 #565656, stop:0.1 #525252, stop:0.5 #4e4e4e, stop:0.9 #4a4a4a, stop:1 #464646);border-style:solid;border:1px solid #1e1e1e;border-radius:5;}QComboBox:hover,QPushButton:hover{border:2px solid QLinearGradient( x1:0, y1:0, x2:0, y2:1, stop:0 #436775 , stop:1 #d7801a);}QComboBox:on{padding-top:3px;padding-left:4px;background-color:QLinearGradient( x1:0, y1:0, x2:0, y2:1, stop:0 #2d2d2d, stop:0.1 #2b2b2b, stop:0.5 #292929, stop:0.9 #282828, stop:1 #252525);selection-background-color:#ffaa00;}QComboBox QAbstractItemView{border:2px solid darkgray;selection-background-color:QLinearGradient( x1:0, y1:0, x2:0, y2:1, stop:0 #436775 , stop:1 #d7801a);}QComboBox::drop-down{subcontrol-origin:padding;subcontrol-position:top right;width:15px;border-left-width:0px;border-left-color:darkgray;border-left-style:solid;border-top-right-radius:3px;border-bottom-right-radius:3px;}QComboBox::down-arrow{image:url(:/down_arrow.png);}QGroupBox:focus{border:2px solid QLinearGradient( x1:0, y1:0, x2:0, y2:1, stop:0 #436775 , stop:1 #d7801a);}QTextEdit:focus{border:2px solid QLinearGradient( x1:0, y1:0, x2:0, y2:1, stop:0 #436775 , stop:1 #d7801a);}QScrollBar:horizontal{border:1px solid #222222;background:QLinearGradient( x1:0, y1:0, x2:0, y2:1, stop:0.0 #121212, stop:0.2 #282828, stop:1 #484848);height:7px;margin:0px 16px 0 16px;}QScrollBar::handle:horizontal{background:QLinearGradient( x1:0, y1:0, x2:1, y2:0, stop:0 #436775 , stop:0.5 #d7801a, stop:1 #436775 );min-height:20px;border-radius:2px;}QScrollBar::add-line:horizontal{border:1px solid #1b1b19;border-radius:2px;background:QLinearGradient( x1:0, y1:0, x2:1, y2:0, stop:0 #436775 , stop:1 #d7801a);width:14px;subcontrol-position:right;subcontrol-origin:margin;}QScrollBar::sub-line:horizontal{border:1px solid #1b1b19;border-radius:2px;background:QLinearGradient( x1:0, y1:0, x2:1, y2:0, stop:0 #436775 , stop:1 #d7801a);width:14px;subcontrol-position:left;subcontrol-origin:margin;}QScrollBar::right-arrow:horizontal, QScrollBar::left-arrow:horizontal{border:1px solid black;width:1px;height:1px;background:white;}QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal{background:none;}QScrollBar:vertical{background:QLinearGradient( x1:0, y1:0, x2:1, y2:0, stop:0.0 #121212, stop:0.2 #282828, stop:1 #484848);width:7px;margin:16px 0 16px 0;border:1px solid #222222;}QScrollBar::handle:vertical{background:QLinearGradient( x1:0, y1:0, x2:0, y2:1, stop:0 #436775 , stop:0.5 #d7801a, stop:1 #436775 );min-height:20px;border-radius:2px;}QScrollBar::add-line:vertical{border:1px solid #1b1b19;border-radius:2px;background:QLinearGradient( x1:0, y1:0, x2:0, y2:1, stop:0 #436775 , stop:1 #d7801a);height:14px;subcontrol-position:bottom;subcontrol-origin:margin;}QScrollBar::sub-line:vertical{border:1px solid #1b1b19;border-radius:2px;background:QLinearGradient( x1:0, y1:0, x2:0, y2:1, stop:0 #d7801a, stop:1 #436775 );height:14px;subcontrol-position:top;subcontrol-origin:margin;}QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical{border:1px solid black;width:1px;height:1px;background:white;}QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical{background:none;}QTextEdit{background-color:#242424;}QPlainTextEdit{background-color:#242424;}QHeaderView::section{background-color:QLinearGradient(x1:0, y1:0, x2:0, y2:1, stop:0 #616161, stop:0.5 #505050, stop:0.6 #434343, stop:1 #656565);color:white;padding-left:4px;border:1px solid #6c6c6c;}QCheckBox:disabled{color:#414141;}QDockWidget::title{text-align:center;spacing:3px;background-color:QLinearGradient(x1:0, y1:0, x2:0, y2:1, stop:0 #323232, stop:0.5 #242424, stop:1 #323232);}QDockWidget::close-button, QDockWidget::float-button{text-align:center;spacing:1px;background-color:QLinearGradient(x1:0, y1:0, x2:0, y2:1, stop:0 #323232, stop:0.5 #242424, stop:1 #323232);}QDockWidget::close-button:hover, QDockWidget::float-button:hover{background:#242424;}QDockWidget::close-button:pressed, QDockWidget::float-button:pressed{padding:1px -1px -1px 1px;}QMainWindow::separator{background-color:QLinearGradient(x1:0, y1:0, x2:0, y2:1, stop:0 #161616, stop:0.5 #151515, stop:0.6 #212121, stop:1 #343434);color:white;padding-left:4px;border:1px solid #4c4c4c;spacing:3px;}QMainWindow::separator:hover{background-color:QLinearGradient(x1:0, y1:0, x2:0, y2:1, stop:0 #d7801a, stop:0.5 #b56c17 stop:1 #436775 );color:white;padding-left:4px;border:1px solid #6c6c6c;spacing:3px;}QToolBar::handle{spacing:3px;background:url(:/images/handle.png);}QMenu::separator{height:2px;background-color:QLinearGradient(x1:0, y1:0, x2:0, y2:1, stop:0 #161616, stop:0.5 #151515, stop:0.6 #212121, stop:1 #343434);color:white;padding-left:4px;margin-left:10px;margin-right:5px;}QProgressBar{border:2px solid grey;border-radius:5px;text-align:center;}QProgressBar::chunk{background-color:#d7801a;width:2.15px;margin:0.5px;}QTabBar::tab{color:#b1b1b1;border:1px solid #444;border-bottom-style:none;background-color:#323232;padding-left:10px;padding-right:10px;padding-top:3px;padding-bottom:2px;margin-right:-1px;}QTabWidget::pane{border:1px solid #444;top:1px;}QTabBar::tab:last{margin-right:0;border-top-right-radius:3px;}QTabBar::tab:first:!selected{margin-left:0px;border-top-left-radius:3px;}QTabBar::tab:!selected{color:#b1b1b1;border-bottom-style:solid;margin-top:3px;background-color:QLinearGradient(x1:0, y1:0, x2:0, y2:1, stop:1 #212121, stop:.4 #343434);}QTabBar::tab:selected{border-top-left-radius:3px;border-top-right-radius:3px;margin-bottom:0px;}QTabBar::tab:!selected:hover{border-top-left-radius:3px;border-top-right-radius:3px;background-color:QLinearGradient(x1:0, y1:0, x2:0, y2:1, stop:1 #212121, stop:0.4 #343434, stop:0.2 #343434, stop:0.1 #ffaa00);}QRadioButton::indicator:checked, QRadioButton::indicator:unchecked{color:#b1b1b1;background-color:#323232;border:1px solid #b1b1b1;border-radius:6px;}QRadioButton::indicator:checked{background-color:qradialgradient( cx:0.5, cy:0.5, fx:0.5, fy:0.5, radius:1.0, stop:0.25 #ffaa00, stop:0.3 #323232 );}QCheckBox::indicator{color:#b1b1b1;background-color:#323232;border:1px solid #b1b1b1;width:9px;height:9px;}QRadioButton::indicator{border-radius:6px;}QRadioButton::indicator:hover, QCheckBox::indicator:hover{border:1px solid #ffaa00;}QCheckBox::indicator:checked{image:url(:/images/checkbox.png);}QCheckBox::indicator:disabled, QRadioButton::indicator:disabled{border:1px solid #444;}""")

    # noinspection PyArgumentList
    def close_application(self):
        choice = QtWidgets.QMessageBox.question(self, 'Exiting...',
                                                "Are you sure you want to quit? Any unsaved data will be lost!",
                                                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if choice == QtWidgets.QMessageBox.Yes:
            print("Application Closed!")
            sys.exit()
        else:
            pass

    def show_prediction_results(self):
        self.pred_results_box.setEnabled(True)
        if self.show_results_btn.text() == 'Show Results':
            self.pred_results_box.show()
            self.clear_results_btn.setEnabled(True)
            self.show_results_btn.setText('Hide Results')
        elif self.show_results_btn.text() == 'Hide Results':
            self.pred_results_box.close()
            self.clear_results_btn.setEnabled(False)
            self.show_results_btn.setText('Show Results')

    def show_train_results(self):
        self.results_box.setEnabled(True)
        if self.train_test_result_btn.text() == 'Show Train/Test Results':
            self.results_box.show()
            self.train_test_result_btn.setText('Hide Train/Test Results')
        elif self.train_test_result_btn.text() == 'Hide Train/Test Results':
            self.results_box.close()
            self.train_test_result_btn.setText('Show Train/Test Results')

    # noinspection PyArgumentList
    def open_file_name_dialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self, "Import Data Set", "",
                                                   "SAS Files (*.sas7bdat);;CSV Files (*.csv)", options=options)

        QApplication.setOverrideCursor(Qt.WaitCursor)
        if file_name:
            global df, df2, df3, df4, df5
            if self.sender().text() == 'Change Activity Data Set' \
                    or self.sender().text() == 'Import Player Activity Data Set':
                # Set File Path
                self.act_file_path_label.setText(file_name)

                if os.path.basename(file_name).split('.')[1] == 'sas7bdat':
                    df2 = pd.read_sas(file_name)
                    QApplication.restoreOverrideCursor()
                    QtWidgets.QMessageBox.information(self, 'Data Set Upload', "Data Set Uploaded Successfully.",
                                                      QtWidgets.QMessageBox.Ok)
                elif os.path.basename(file_name).split('.')[1] == 'csv':
                    df2 = pd.read_csv(file_name)
                    QApplication.restoreOverrideCursor()
                    QtWidgets.QMessageBox.information(self, 'Data Set Upload', "Data Set Uploaded Successfully.",
                                                      QtWidgets.QMessageBox.Ok)

            elif self.sender().text() == 'Change RG Data Set' \
                    or self.sender().text() == 'Import Player RG Events Data Set':
                # Set File Path
                self.rg_file_path_label.setText(file_name)

                if os.path.basename(file_name).split('.')[1] == 'sas7bdat':
                    df3 = pd.read_sas(file_name)
                    QApplication.restoreOverrideCursor()
                    QtWidgets.QMessageBox.information(self, 'Data Set Upload',
                                                      "Data Set Uploaded Successfully.", QtWidgets.QMessageBox.Ok)
                elif os.path.basename(file_name).split('.')[1] == 'csv':
                    df3 = pd.read_csv(file_name)
                    QApplication.restoreOverrideCursor()
                    QtWidgets.QMessageBox.information(self, 'Data Set Upload', "Data Set Uploaded Successfully.",
                                                      QtWidgets.QMessageBox.Ok)
            elif self.sender().text() == 'Change Player Descriptions Data Set' \
                    or self.sender().text() == 'Import Player Descriptions Data Set':
                # Set File Path
                self.desc_file_path_label.setText(file_name)

                if os.path.basename(file_name).split('.')[1] == 'sas7bdat':
                    df = pd.read_sas(file_name)
                    QApplication.restoreOverrideCursor()
                    QtWidgets.QMessageBox.information(self, 'Data Set Upload', "Data Set Uploaded Successfully.",
                                                      QtWidgets.QMessageBox.Ok)
                elif os.path.basename(file_name).split('.')[1] == 'csv':
                    df = pd.read_csv(file_name)
                    QApplication.restoreOverrideCursor()
                    QtWidgets.QMessageBox.information(self, 'Data Set Upload', "Data Set Uploaded Successfully.",
                                                      QtWidgets.QMessageBox.Ok)

            elif self.sender().text() == 'Upload Data to Analyse' \
                    or self.sender().text() == 'Import Data Set to Analyse':
                # Set File Path
                self.pred_file_path.setText(file_name)

                if self.train_models.isChecked():
                    self.result_type_combo.setEnabled(True)
                    self.start_pred.setEnabled(True)
                elif self.pretrained.isChecked() and self.id_file_path.text() != 'No Data File Uploaded.':
                    self.start_pred.setEnabled(True)

                if os.path.basename(file_name).split('.')[1] == 'sas7bdat':
                    df4 = pd.read_sas(file_name)
                    QApplication.restoreOverrideCursor()
                    QtWidgets.QMessageBox.information(self, 'Data Set Upload', "Data Set Uploaded Successfully.",
                                                      QtWidgets.QMessageBox.Ok)
                elif os.path.basename(file_name).split('.')[1] == 'csv':
                    df4 = pd.read_csv(file_name, index_col=0)
                    QApplication.restoreOverrideCursor()
                    QtWidgets.QMessageBox.information(self, 'Data Set Upload', "Data Set Uploaded Successfully.",
                                                      QtWidgets.QMessageBox.Ok)

            elif self.sender().text() == 'Upload IDs Data' \
                    or self.sender().text() == 'Import IDs Data Set':
                # Set File Path
                self.id_file_path.setText(file_name)

                if self.pretrained.isChecked() and self.pred_file_path.text() != 'No Data File Uploaded.':
                    self.start_pred.setEnabled(True)

                if os.path.basename(file_name).split('.')[1] == 'sas7bdat':
                    df5 = pd.read_sas(file_name)
                    QApplication.restoreOverrideCursor()
                    QtWidgets.QMessageBox.information(self, 'Data Set Upload', "Data Set Uploaded Successfully.",
                                                      QtWidgets.QMessageBox.Ok)
                elif os.path.basename(file_name).split('.')[1] == 'csv':
                    df5 = pd.read_csv(file_name, index_col=0)
                    QApplication.restoreOverrideCursor()
                    QtWidgets.QMessageBox.information(self, 'Data Set Upload', "Data Set Uploaded Successfully.",
                                                      QtWidgets.QMessageBox.Ok)

            if self.act_file_path_label.text() != 'No Data File Uploaded.' \
                    and self.rg_file_path_label.text() != 'No Data File Uploaded.' \
                    and self.desc_file_path_label.text() != 'No Data File Uploaded.':
                # Enable the Start Button
                self.start_train_process_btn.setEnabled(True)

    def start_process(self):
        self.start_train_process_btn.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.calc.countChanged.connect(self.onCountChanged)
        self.calc.textChanged.connect(self.onTextChanged)
        self.calc.stateChanged.connect(self.onStateChanged)
        self.calc.start()


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    desired_width = 250
    pd.set_option('display.width', desired_width)
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("plastique")
    app.setApplicationName("RiskR")
    window = MyApp()
    window.show()
    sys.exit(app.exec_())

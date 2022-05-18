import os

import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

desired_width = 250
pd.set_option('display.width', desired_width)


def specificity(matrix):
    spec = matrix[0][0] / (matrix[0][0] + matrix[0][1])
    print("Specificity: ", round(spec, 2))  # TN/(TN + FP)


def dtc():
    model = DecisionTreeClassifier()

    model.fit(train_features, train_labels)

    predictions = model.predict(test_features)

    print("Decision Tree: ", round(accuracy_score(test_labels, predictions) * 100, 2), "%.")


def k_nearest():
    model = KNeighborsClassifier()

    model.fit(train_features, train_labels)

    predictions = model.predict(test_features)

    print("K-Nearest Neighbors: ", round(accuracy_score(test_labels, predictions) * 100, 2), "%.")


def bagging_k_nearest():
    rf = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)

    rf.fit(train_features, train_labels)

    predictions = rf.predict(test_features)

    # Performance Metrics
    acc_score = accuracy_score(test_labels, predictions, normalize=True) * 100

    print("Bagging K_Nearest: ", round(acc_score, 2), "%.")


def sgd(x_train, x_test):
    scaler = StandardScaler()
    scaler.fit(x_train)  # fitting of training data to be scaled
    train_features = scaler.transform(x_train)
    test_features = scaler.transform(x_test)

    model = SGDClassifier(loss="hinge", penalty="l2", learning_rate='optimal', n_jobs=1)

    model.fit(train_features, train_labels)

    predictions = model.predict(test_features)

    print("Stochastic gradient descent: ", round(accuracy_score(test_labels, predictions) * 100, 2), "%.")


def naive():
    model = GaussianNB()

    model.fit(train_features, train_labels)

    predictions = model.predict(test_features)

    print("Naive Bayes: ", round(accuracy_score(test_labels, predictions) * 100, 2), "%.")


def bagging_nb():
    rf = BaggingClassifier(GaussianNB(), max_samples=0.5, max_features=0.5)

    rf.fit(train_features, train_labels)

    predictions = rf.predict(test_features)

    # Performance Metrics
    acc_score = accuracy_score(test_labels, predictions, normalize=True) * 100

    print("Bagging Naive Bayes: ", round(acc_score, 2), "%.")


def bernoulli_naive():
    model = BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None)

    model.fit(train_features, train_labels)

    predictions = model.predict(test_features)

    print("Bernoulli Naive Bayes: ", round(accuracy_score(test_labels, predictions) * 100, 2), "%.")


def lda():
    model = LinearDiscriminantAnalysis()

    model.fit(train_features, train_labels)

    predictions = model.predict(test_features)

    print("Linear Discriminant Analysis: ", round(accuracy_score(test_labels, predictions) * 100, 2), "%.")


def lr():
    model = LogisticRegression()

    model.fit(train_features, train_labels)

    predictions = model.predict(test_features)

    print("Logistic Regression: ", round(accuracy_score(test_labels, predictions) * 100, 2), "%.")


def svm(x_train, x_test):
    scaler = StandardScaler()
    scaler.fit(x_train)  # fitting of training data to be scaled
    train_features = scaler.transform(x_train)
    test_features = scaler.transform(x_test)

    model = SVC(C=0.001, cache_size=200, class_weight=None, coef0=0.0,
                decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf',
                max_iter=-1, probability=False, random_state=None, shrinking=True,
                tol=0.001, verbose=False)

    model.fit(train_features, train_labels)

    predictions = model.predict(test_features)

    print("Support Vector Machine: ", round(accuracy_score(test_labels, predictions) * 100, 2), "%.")


def bagging_svm():
    rf = BaggingClassifier(SVC())

    rf.fit(train_features, train_labels)

    predictions = rf.predict(test_features)

    # Performance Metrics
    acc_score = accuracy_score(test_labels, predictions, normalize=True) * 100

    print("Bagging SVC: ", round(acc_score, 2), "%.")


def random_forest():
    rf = RandomForestClassifier(n_estimators=500, random_state=42, max_depth=5)

    rf.fit(train_features, train_labels)

    predictions = rf.predict(test_features)

    # Performance Metrics
    acc_score = accuracy_score(test_labels, predictions, normalize=True) * 100

    print("Random Forest: ", round(acc_score, 2), "%.")


def bagging_random_forest():
    rf = BaggingClassifier(RandomForestClassifier
                           (n_estimators=500, random_state=42, max_depth=5), max_samples=0.5, max_features=0.5)

    rf.fit(train_features, train_labels)

    predictions = rf.predict(test_features)

    # Performance Metrics
    acc_score = accuracy_score(test_labels, predictions, normalize=True) * 100

    print("Bagging Random Forest: ", round(acc_score, 2), "%.")


def extremely_random_trees():
    rf = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)

    rf.fit(train_features, train_labels)

    predictions = rf.predict(test_features)

    # Performance Metrics
    acc_score = accuracy_score(test_labels, predictions, normalize=True) * 100

    print("Extremely Randomized Trees: ", round(acc_score, 2), "%.")


def adaboost():
    rf = AdaBoostClassifier(n_estimators=100)

    rf.fit(train_features, train_labels)

    predictions = rf.predict(test_features)

    # Performance Metrics
    acc_score = accuracy_score(test_labels, predictions, normalize=True) * 100

    print("AdaBoost: ", round(acc_score, 2), "%.")


def neural_network():
    model = Sequential()

    # Adding the input layer and the first hidden layer
    model.add(Dense(units=56, kernel_initializer='uniform', activation='relu', input_dim=56))
    # Adding the second hidden layer
    model.add(Dense(units=23, kernel_initializer='uniform', activation='relu'))
    # Adding the third hidden layer
    model.add(Dense(units=11, kernel_initializer='uniform', activation='relu'))

    # Adding the output layer
    model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

    # Compiling Neural Network
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_features, train_labels, batch_size=5, epochs=10, verbose=0)
    predictions = model.predict(test_features)
    predictions = (predictions > 0.5)

    print("Neural Network: ", round(accuracy_score(test_labels, predictions) * 100, 2), "%.")


def data():
    df = pd.read_sas('data/data1.sas7bdat')
    df2 = pd.read_sas('data/data2.sas7bdat')
    df3 = pd.read_sas('data/data3.sas7bdat')

    df.drop('LanguageName', 1, inplace=True)

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

    date_df = pd.merge(left=date_df, right=df2, left_on='UserID', right_on='UserID')

    date_df.drop('Date', 1, inplace=True)

    merged_inner = pd.merge(left=df, right=date_df, left_on='USERID', right_on='UserID')

    df3['AtRisk'] = np.where(df3['RGsumevents'] != 0, 1, 0)

    df3.drop(df3.columns[2:6], axis=1, inplace=True)

    features = pd.merge(merged_inner, df3, on='UserID', how='left')
    features.loc[features.AtRisk != 1, 'AtRisk'] = 0
    features['AtRisk'] = features['AtRisk'].astype(np.int64)

    features.loc[features.RGsumevents.isnull(), 'RGsumevents'] = 0
    features['RGsumevents'] = features['RGsumevents'].astype(np.int64)

    features.drop(features.columns[5:8], axis=1, inplace=True)

    aggregation_functions = {'CountryName': 'first', 'Gender': 'first',
                             'YearofBirth': 'first', 'Turnover': 'sum', 'Hold': 'sum', 'NumberofBets': 'sum',
                             'Duration_Days': 'first', 'RGsumevents': 'first', 'AtRisk': 'first'}
    features = features.groupby(features['USERID'], as_index=False).aggregate(aggregation_functions)

    features['Profit'] = np.where((features['Turnover'] - features['Hold']) > 0, 1, 0)

    features = pd.get_dummies(features, columns=["CountryName", "Gender"])

    # Correct Country Names
    features.rename(columns={'CountryName_Bosnia and Herzego': 'CountryName_BosniaHerzegovina',
                             'CountryName_FYR acedonia': 'CountryName_Macedonia',
                             'CountryName_Leanon': 'CountryName_Lebanon',
                             'CountryName_Luxemourg': 'CountryName_Luxembourg',
                             'CountryName_New Zealand': 'CountryName_NewZealand',
                             'CountryName_Russian Federation': 'CountryName_RussianFederation',
                             'CountryName_alta': 'CountryName_Malta', 'CountryName_anada': 'CountryName_Canada',
                             'CountryName_exicoX': 'CountryName_Mexico', 'CountryName_orocco': 'CountryName_Morocco',
                             'CountryName_roatia': 'CountryName_Croatia', 'CountryName_yprus': 'CountryName_Cyprus',
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
    features.drop('USERID', 1, inplace=True)

    features = features[features.NumberofBets != 0]

    # IMPORTANT TO PLAY WITH !!!!
    # features = features[features.Duration_Days > 5]

    features.dropna(inplace=True)
    features.drop_duplicates(inplace=True)

    labels = np.array(features['AtRisk'])

    features.drop('AtRisk', axis=1, inplace=True)
    df_list = list(features.columns)

    features = np.array(features)

    return features, labels


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    features, labels = data()

    kFold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for train_index, test_index in kFold.split(features, labels):
        train_features, test_features = features[train_index], features[test_index]
        train_labels, test_labels = labels[train_index], labels[test_index]

    # neural_network()
    # random_forest()
    # bagging_random_forest()
    # extremely_random_trees()
    # dtc()
    # adaboost()
    # k_nearest()
    # bagging_k_nearest()
    # svm(train_features, test_features)
    # bagging_svm()
    # naive()
    # bagging_nb()
    # bernoulli_naive()
    ##lr()
    # lda()
    # sgd(train_features, test_features)

    # To display data visually, reverse one-hot and re-add dropped coloumns (store in a list numpy array)

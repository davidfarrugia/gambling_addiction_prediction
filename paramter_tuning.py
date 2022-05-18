import multiprocessing

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

import elm


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
                             'YearofBirth': 'first', 'Turnover': 'sum', 'Hold': 'sum',
                             'NumberofBets': 'sum',
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

    features = features[features.Duration_Days > 5]

    features.dropna(inplace=True)
    features.drop_duplicates(inplace=True)

    labels = np.array(features['AtRisk'])

    features.drop('AtRisk', axis=1, inplace=True)

    df_list = list(features.columns)

    if 'USERID' in df_list: df_list.remove('USERID')

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    X_train.drop('USERID', 1, inplace=True)

    X_train = np.array(X_train)

    to_predict = X_test.drop('USERID', 1)

    predict_set = pd.DataFrame(to_predict, columns=df_list)

    predict_set.to_csv('data/pred.csv')

    user_id_for_prediction = np.array(X_test['USERID'])

    return X_train, y_train


if __name__ == '__main__':
    X_train, y_train = data()

    scale = StandardScaler()
    scale.fit(X_train)  # fitting of training data to be scaled
    train_features = scale.transform(X_train)

    X_train, X_test, y_train, y_test = train_test_split(train_features, y_train,
                                                        train_size=0.75, test_size=0.25)

    param_grid = {    # Parameters to tune
        'n_clusters': [2]
    }

    model = KMeans() #Model to Tune

    model.fit(X_train, y_train)

    grid = GridSearchCV(model, param_grid, verbose=5, cv=10, n_jobs=multiprocessing.cpu_count(), scoring='roc_auc')
    grid.fit(X_train, y_train)

    print(grid.best_score_)
    print(grid.best_params_)

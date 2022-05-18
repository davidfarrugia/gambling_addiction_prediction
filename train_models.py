# Save Model Using Pickle
import pickle
import tempfile

import keras
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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


if __name__ == '__main__':
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

    features.dropna(inplace=True)
    features.drop_duplicates(inplace=True)

    labels = np.array(features['AtRisk'])

    # features.to_pickle('data.pkl')
    # data = pd.read_pickle('data.pkl')

    print(len(features))

    features.drop('AtRisk', axis=1, inplace=True)

    df_list = list(features.columns)

    if 'USERID' in df_list: df_list.remove('USERID')

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    X_train.drop('USERID', 1, inplace=True)

    X_train = np.array(X_train)

    X_train2, X_test2, y_train2, y_test2 = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    scale = StandardScaler()
    scale.fit(X_train2)  # fitting of training data to be scaled
    train_features = scale.transform(X_train2)

    make_keras_picklable()

    # model = RandomForestClassifier(bootstrap=True, criterion="gini", max_features=0.2, min_samples_leaf=1,
    # min_samples_split=20, n_estimators=100)

    # model.fit(train_features, y_train2)

    # predictions = model.predict(X_test2)

    # save the model to disk
    filename = 'trained_models/neural_network.sav'
    # pickle.dump(model, open(filename, 'wb'))

    # some time later...

    # load the model from disk

    loaded_model = pickle.load(open(filename, 'rb'))
    predictions = loaded_model.predict(X_test2)

    predictions = (predictions > 0.5) * 1

    x = predictions.ravel()

    print(predictions)

    print()
    print()

    # Performance Metrics
    acc_score = accuracy_score(y_test2, x, normalize=True) * 100
    auroc = roc_auc_score(y_test2, x)
    tn, fp, fn, tp = confusion_matrix(y_test2, x).ravel()
    sens = tp / (tp + fn) * 100
    spec = tn / (tn + fp) * 100

    print(classification_report(y_test2, x))

    print()

    print("AUROC: ", round(auroc, 4), "%.")
    print("Accuracy: ", round(acc_score, 4), "%.")
    print("Sensitivity: ", round(sens, 4), "%.")
    print("Specificity: ", round(spec, 4), "%.")

    # predictions = [int(elem) for elem in list(chain.from_iterable(predictions))]
    # predictions = list(chain.from_iterable(predictions))

    print(x)

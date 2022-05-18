import lightgbm as lgb
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def adaboost(train_features, train_labels, test_features, test_labels):
    model = AdaBoostClassifier(n_estimators=500, learning_rate=0.1, algorithm='SAMME.R')

    model.fit(train_features, train_labels)

    predictions = model.predict(test_features)

    # Performance Metrics
    acc_score = accuracy_score(test_labels, predictions, normalize=True) * 100
    auroc = roc_auc_score(test_labels, predictions)
    tn, fp, fn, tp = confusion_matrix(test_labels, predictions).ravel()
    sens = tp / (tp + fn) * 100
    spec = tn / (tn + fp) * 100

    print("AdaBoost Accuracy: ", round(acc_score, 2), "%.")
    print("AdaBoost Sensitivity: ", round(sens, 2), "%.")
    print("AdaBoost Spec: ", round(spec, 2), "%.")

    return model, round(auroc, 4), round(acc_score, 2), round(sens, 2), round(spec, 2)


def xgb(train_features, train_labels, test_features, test_labels):
    model = XGBClassifier(subsample=1, max_depth=3, learning_rate=0.1, gamma=0.1, colsample_bytree=1)

    model.fit(train_features, train_labels)

    predictions = model.predict(test_features)

    predictions = (predictions > 0.5)

    # Performance Metrics
    acc_score = accuracy_score(test_labels, predictions, normalize=True) * 100
    auroc = roc_auc_score(test_labels, predictions)
    tn, fp, fn, tp = confusion_matrix(test_labels, predictions).ravel()
    sens = tp / (tp + fn) * 100
    spec = tn / (tn + fp) * 100

    print("XGBClassifier AUROC: ", round(auroc, 2), "%.")
    print("XGBClassifier Accuracy: ", round(acc_score, 2), "%.")
    print("XGBClassifier Sensitivity: ", round(sens, 2), "%.")
    print("XGBClassifier Specificity: ", round(spec, 2), "%.")

    return model, round(auroc, 4), round(acc_score, 2), round(sens, 2), round(spec, 2)


def lightgbm(train_features, train_labels, test_features, test_labels):
    sc = StandardScaler()
    sc.fit(train_features)  # fitting of training data to be scaled
    train_features = sc.transform(train_features)
    test_features = sc.transform(test_features)

    d_train = lgb.Dataset(train_features, label=train_labels)

    params = {'learning_rate': 0.003, 'boosting_type': 'gbdt', 'objective': 'binary', 'metric': 'binary_logloss',
              'sub_feature': 0.5, 'num_leaves': 10, 'min_data': 50, 'max_depth': 10}

    model = lgb.train(params, d_train, 100)

    # Prediction
    predictions = model.predict(test_features)

    predictions = (predictions > 0.5)

    # Performance Metrics
    acc_score = accuracy_score(test_labels, predictions, normalize=True) * 100
    auroc = roc_auc_score(test_labels, predictions)
    tn, fp, fn, tp = confusion_matrix(test_labels, predictions).ravel()
    sens = tp / (tp + fn) * 100
    spec = tn / (tn + fp) * 100

    print("lgb AUROC: ", round(auroc, 2), "%.")
    print("lgb Accuracy: ", round(acc_score, 2), "%.")
    print("lgb Sensitivity: ", round(sens, 2), "%.")
    print("lgb Specificity: ", round(spec, 2), "%.")

    return model, round(auroc, 4), round(acc_score, 2), round(sens, 2), round(spec, 2)

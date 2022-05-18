from sklearn import linear_model
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

import elm


def elm_model(train_features, train_labels, test_features, test_labels):
    scale = StandardScaler()
    scale.fit(train_features)  # fitting of training data to be scaled
    train_features = scale.transform(train_features)
    test_features = scale.transform(test_features)

    model = elm.ELMClassifier(n_hidden=2000,
                              alpha=0.93,
                              activation_func='sigmoid',
                              regressor=linear_model.Ridge(),
                              random_state=42)

    model.fit(train_features, train_labels)

    predictions = model.predict(test_features)

    # Performance Metrics
    acc_score = accuracy_score(test_labels, predictions, normalize=True) * 100
    auroc = roc_auc_score(test_labels, predictions)
    tn, fp, fn, tp = confusion_matrix(test_labels, predictions).ravel()
    sens = tp / (tp + fn) * 100
    spec = tn / (tn + fp) * 100

    print("Extreme Learning Machine Accuracy: ", round(acc_score, 2), "%.")
    print("Extreme Learning Machine AUROC: ", round(auroc, 2), "%.")
    print("Extreme Learning Machine Sensitivity: ", round(sens, 2), "%.")
    print("Extreme Learning Machine Spec: ", round(spec, 2), "%.")

    return model, round(auroc, 4), round(acc_score, 2), round(sens, 2), round(spec, 2)

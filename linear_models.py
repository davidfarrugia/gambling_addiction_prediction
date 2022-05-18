from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score


def lda(train_features, train_labels, test_features, test_labels):
    model = LinearDiscriminantAnalysis()

    model.fit(train_features, train_labels)

    predictions = model.predict(test_features)

    # Performance Metrics
    acc_score = accuracy_score(test_labels, predictions, normalize=True) * 100
    auroc = roc_auc_score(test_labels, predictions)
    tn, fp, fn, tp = confusion_matrix(test_labels, predictions).ravel()
    sens = tp / (tp + fn) * 100
    spec = tn / (tn + fp) * 100

    print("Linear Discriminant Analysis AUROC: ", round(auroc, 2), "%.")
    print("Linear Discriminant Analysis Accuracy: ", round(acc_score, 2), "%.")
    print("Linear Discriminant Analysis Sensitivity: ", round(sens, 2), "%.")
    print("Linear Discriminant Analysis Spec: ", round(spec, 2), "%.")

    return model, round(auroc, 4), round(acc_score, 2), round(sens, 2), round(spec, 2)


def linear_regression(train_features, train_labels, test_features, test_labels):
    model = LinearRegression()

    model.fit(train_features, train_labels)

    predictions = model.predict(test_features)

    # Performance Metrics
    acc_score = accuracy_score(test_labels, predictions.round(), normalize=True) * 100
    auroc = roc_auc_score(test_labels, predictions)
    tn, fp, fn, tp = confusion_matrix(test_labels, predictions.round(), labels=[0, 1]).ravel()
    sens = tp / (tp + fn) * 100
    spec = tn / (tn + fp) * 100

    print("Linear Regression AUROC: ", round(auroc, 2), "%.")
    print("Linear Regression Accuracy: ", round(acc_score, 2), "%.")
    print("Linear Regression Sensitivity: ", round(sens, 2), "%.")
    print("Linear Regression Spec: ", round(spec, 2), "%.")

    return model, round(auroc, 4), round(acc_score, 2), round(sens, 2), round(spec, 2)

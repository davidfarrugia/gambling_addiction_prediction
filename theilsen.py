from sklearn.linear_model import TheilSenRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix


def theil_sen(train_features, train_labels, test_features, test_labels):
    model = TheilSenRegressor(max_iter=500, max_subpopulation=100, tol=0.01)

    model.fit(train_features, train_labels)

    predictions = model.predict(test_features)

    predictions = (predictions > 0.5)

    # Performance Metrics
    acc_score = accuracy_score(test_labels, predictions, normalize=True) * 100
    auroc = roc_auc_score(test_labels, predictions)
    tn, fp, fn, tp = confusion_matrix(test_labels, predictions).ravel()
    sens = tp / (tp + fn) * 100
    spec = tn / (tn + fp) * 100

    print("TheilSenRegressor AUROC: ", round(auroc, 2), "%.")
    print("TheilSenRegressor Accuracy: ", round(acc_score, 2), "%.")
    print("TheilSenRegressor Sensitivity: ", round(sens, 2), "%.")
    print("TheilSenRegressor Specificity: ", round(spec, 2), "%.")

    return model, round(auroc, 4), round(acc_score, 2), round(sens, 2), round(spec, 2)

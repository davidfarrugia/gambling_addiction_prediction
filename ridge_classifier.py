from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix


def ridge(train_features, train_labels, test_features, test_labels):
    model = RidgeClassifier(alpha=0.1, max_iter=50, solver='lsqr')

    model.fit(train_features, train_labels)

    predictions = model.predict(test_features)

    # Performance Metrics
    acc_score = accuracy_score(test_labels, predictions, normalize=True) * 100
    auroc = roc_auc_score(test_labels, predictions)
    tn, fp, fn, tp = confusion_matrix(test_labels, predictions).ravel()
    sens = tp / (tp + fn) * 100
    spec = tn / (tn + fp) * 100

    print("RidgeClassifier AUROC: ", round(auroc, 2), "%.")
    print("RidgeClassifier Accuracy: ", round(acc_score, 2), "%.")
    print("RidgeClassifier Sensitivity: ", round(sens, 2), "%.")
    print("RidgeClassifier Specificity: ", round(spec, 2), "%.")

    return model, round(auroc, 4), round(acc_score, 2), round(sens, 2), round(spec, 2)

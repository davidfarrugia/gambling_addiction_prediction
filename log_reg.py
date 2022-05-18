from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score


def lr(train_features, train_labels, test_features, test_labels):
    model = LogisticRegression(solver='newton-cg', max_iter=1000, penalty='l2')

    model.fit(train_features, train_labels)

    predictions = model.predict(test_features)

    # Performance Metrics
    acc_score = accuracy_score(test_labels, predictions, normalize=True) * 100
    auroc = roc_auc_score(test_labels, predictions)
    tn, fp, fn, tp = confusion_matrix(test_labels, predictions).ravel()
    sens = tp / (tp + fn) * 100
    spec = tn / (tn + fp) * 100

    print("Logistic Regression Accuracy: ", round(acc_score, 2), "%.")
    print("Logistic Regression Sensitivity: ", round(sens, 2), "%.")
    print("Logistic Regression Spec: ", round(spec, 2), "%.")

    return model, round(auroc, 4), round(acc_score, 2), round(sens, 2), round(spec, 2)

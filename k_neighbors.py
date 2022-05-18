from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier


def k_nearest(train_features, train_labels, test_features, test_labels):
    model = KNeighborsClassifier(n_neighbors=100, weights='uniform', algorithm='ball_tree')

    model.fit(train_features, train_labels)

    predictions = model.predict(test_features)

    # Performance Metrics
    acc_score = accuracy_score(test_labels, predictions, normalize=True) * 100
    auroc = roc_auc_score(test_labels, predictions)
    tn, fp, fn, tp = confusion_matrix(test_labels, predictions).ravel()
    sens = tp / (tp + fn) * 100
    spec = tn / (tn + fp) * 100

    print("K-Nearest Neighbors AUROC: ", round(auroc, 2), "%.")
    print("K-Nearest Neighbors Accuracy: ", round(acc_score, 2), "%.")
    print("K-Nearest Neighbors Sensitivity: ", round(sens, 2), "%.")
    print("K-Nearest Neighbors Spec: ", round(spec, 2), "%.")

    return model, round(auroc, 4), round(acc_score, 2), round(sens, 2), round(spec, 2)


def bagging_k_nearest(train_features, train_labels, test_features, test_labels):
    model = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)

    model.fit(train_features, train_labels)

    predictions = model.predict(test_features)

    # Performance Metrics
    acc_score = accuracy_score(test_labels, predictions, normalize=True) * 100
    auroc = roc_auc_score(test_labels, predictions)
    tn, fp, fn, tp = confusion_matrix(test_labels, predictions).ravel()
    sens = tp / (tp + fn) * 100
    spec = tn / (tn + fp) * 100

    print("Bagging K_Nearest AUROC: ", round(auroc, 2), "%.")
    print("Bagging K_Nearest Accuracy: ", round(acc_score, 2), "%.")
    print("Bagging K_Nearest Sensitivity: ", round(sens, 2), "%.")
    print("Bagging K_Nearest Spec: ", round(spec, 2), "%.")

    return model, round(auroc, 4), round(acc_score, 2), round(sens, 2), round(spec, 2)

from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix


def kmean(train_features, train_labels, test_features, test_labels):
    model = KMeans(n_clusters=2, init='k-means++')

    model.fit(train_features, train_labels)

    predictions = model.predict(test_features)

    # Performance Metrics
    acc_score = accuracy_score(test_labels, predictions, normalize=True) * 100
    auroc = roc_auc_score(test_labels, predictions)
    tn, fp, fn, tp = confusion_matrix(test_labels, predictions).ravel()
    sens = tp / (tp + fn) * 100
    spec = tn / (tn + fp) * 100

    print("K-Means Clustering AUROC: ", round(auroc, 2), "%.")
    print("K-Means Clustering Accuracy: ", round(acc_score, 2), "%.")
    print("K-Means Clustering Sensitivity: ", round(sens, 2), "%.")
    print("K-Means Clustering Spec: ", round(spec, 2), "%.")

    return model, round(auroc, 4), round(acc_score, 2), round(sens, 2), round(spec, 2)

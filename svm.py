from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC, OneClassSVM


def svm(train_features, train_labels, test_features, test_labels):
    scale = StandardScaler()
    scale.fit(train_features)  # fitting of training data to be scaled
    train_features = scale.transform(train_features)
    test_features = scale.transform(test_features)

    model = SVC(C=10, cache_size=25, gamma=0.001, kernel='linear')

    model.fit(train_features, train_labels)

    predictions = model.predict(test_features)

    # Performance Metrics
    acc_score = accuracy_score(test_labels, predictions, normalize=True) * 100
    auroc = roc_auc_score(test_labels, predictions)
    tn, fp, fn, tp = confusion_matrix(test_labels, predictions).ravel()
    sens = tp / (tp + fn) * 100
    spec = tn / (tn + fp) * 100

    print("Support Vector Machine Accuracy: ", round(acc_score, 2), "%.")
    print("Support Vector Machine Sensitivity: ", round(sens, 2), "%.")
    print("Support Vector Machine Spec: ", round(spec, 2), "%.")

    return model, round(auroc, 4), round(acc_score, 2), round(sens, 2), round(spec, 2)


def bagging_svm(train_features, train_labels, test_features, test_labels):
    scale = StandardScaler()
    scale.fit(train_features)  # fitting of training data to be scaled
    train_features = scale.transform(train_features)
    test_features = scale.transform(test_features)

    model = BaggingClassifier(SVC(), max_samples=0.5, max_features=0.5)

    model.fit(train_features, train_labels)

    predictions = model.predict(test_features)

    # Performance Metrics
    acc_score = accuracy_score(test_labels, predictions, normalize=True) * 100
    auroc = roc_auc_score(test_labels, predictions)
    tn, fp, fn, tp = confusion_matrix(test_labels, predictions).ravel()
    sens = tp / (tp + fn) * 100
    spec = tn / (tn + fp) * 100

    print("Bagging SVC Accuracy: ", round(acc_score, 2), "%.")
    print("Bagging SVC Sensitivity: ", round(sens, 2), "%.")
    print("Bagging SVC Spec: ", round(spec, 2), "%.")

    return model, round(auroc, 4), round(acc_score, 2), round(sens, 2), round(spec, 2)


def linear_svc(train_features, train_labels, test_features, test_labels):
    model = LinearSVC(C=1, loss='hinge', max_iter=2000, tol=0.0001)

    model.fit(train_features, train_labels)

    predictions = model.predict(test_features)

    # Performance Metrics
    acc_score = accuracy_score(test_labels, predictions, normalize=True) * 100
    auroc = roc_auc_score(test_labels, predictions)
    tn, fp, fn, tp = confusion_matrix(test_labels, predictions).ravel()
    sens = tp / (tp + fn) * 100
    spec = tn / (tn + fp) * 100

    print("LinearSVC AUROC: ", round(auroc, 2), "%.")
    print("LinearSVC Accuracy: ", round(acc_score, 2), "%.")
    print("LinearSVC Sensitivity: ", round(sens, 2), "%.")
    print("LinearSVC Specificity: ", round(spec, 2), "%.")

    return model, round(auroc, 4), round(acc_score, 2), round(sens, 2), round(spec, 2)


def one_class_svm(train_features, train_labels, test_features, test_labels):
    model = OneClassSVM()

    model.fit(train_features, train_labels)

    predictions = model.predict(test_features)

    # Performance Metrics
    acc_score = accuracy_score(test_labels, predictions, normalize=True) * 100
    auroc = roc_auc_score(test_labels, predictions)
    tn, fp, fn, tp = confusion_matrix(test_labels, predictions).ravel()
    sens = tp / (tp + fn) * 100
    spec = tn / (tn + fp) * 100

    print("OneClassSVM AUROC: ", round(auroc, 2), "%.")
    print("OneClassSVM Accuracy: ", round(acc_score, 2), "%.")
    print("OneClassSVM Sensitivity: ", round(sens, 2), "%.")
    print("OneClassSVM Specificity: ", round(spec, 2), "%.")

    return model, round(auroc, 4), round(acc_score, 2), round(sens, 2), round(spec, 2)

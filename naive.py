from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.naive_bayes import BernoulliNB, GaussianNB


def naive(train_features, train_labels, test_features, test_labels):
    model = GaussianNB()

    model.fit(train_features, train_labels)

    predictions = model.predict(test_features)

    # Performance Metrics
    acc_score = accuracy_score(test_labels, predictions, normalize=True) * 100
    auroc = roc_auc_score(test_labels, predictions)
    tn, fp, fn, tp = confusion_matrix(test_labels, predictions).ravel()
    sens = tp / (tp + fn) * 100
    spec = tn / (tn + fp) * 100

    print("Naive Bayes Accuracy: ", round(acc_score, 2), "%.")
    print("Naive Bayes Sensitivity: ", round(sens, 2), "%.")
    print("Naive Bayes Spec: ", round(spec, 2), "%.")

    return model, round(auroc, 4), round(acc_score, 2), round(sens, 2), round(spec, 2)


def bagging_nb(train_features, train_labels, test_features, test_labels):
    model = BaggingClassifier(GaussianNB(), max_samples=0.5, max_features=0.5)

    model.fit(train_features, train_labels)

    predictions = model.predict(test_features)

    # Performance Metrics
    acc_score = accuracy_score(test_labels, predictions, normalize=True) * 100
    auroc = roc_auc_score(test_labels, predictions)
    tn, fp, fn, tp = confusion_matrix(test_labels, predictions).ravel()
    sens = tp / (tp + fn) * 100
    spec = tn / (tn + fp) * 100

    print("Bagging Naive Bayes Accuracy: ", round(acc_score, 2), "%.")
    print("Bagging Naive Bayes Sensitivity: ", round(sens, 2), "%.")
    print("Bagging Naive Bayes Spec: ", round(spec, 2), "%.")

    return model, round(auroc, 4), round(acc_score, 2), round(sens, 2), round(spec, 2)


def bernoulli_naive(train_features, train_labels, test_features, test_labels):
    model = BernoulliNB(alpha=0.01, fit_prior=True)

    model.fit(train_features, train_labels)

    predictions = model.predict(test_features)

    # Performance Metrics
    acc_score = accuracy_score(test_labels, predictions, normalize=True) * 100
    auroc = roc_auc_score(test_labels, predictions)
    tn, fp, fn, tp = confusion_matrix(test_labels, predictions).ravel()
    sens = tp / (tp + fn) * 100
    spec = tn / (tn + fp) * 100

    print("Bernoulli Naive Bayes Accuracy: ", round(acc_score, 2), "%.")
    print("Bernoulli Naive Bayes Sensitivity: ", round(sens, 2), "%.")
    print("Bernoulli Naive Bayes Spec: ", round(spec, 2), "%.")

    return model, round(auroc, 4), round(acc_score, 2), round(sens, 2), round(spec, 2)

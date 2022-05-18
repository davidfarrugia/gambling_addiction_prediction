from rgf import RGFClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.tree import DecisionTreeClassifier


def dtc(train_features, train_labels, test_features, test_labels):
    model = DecisionTreeClassifier(max_depth=5, criterion='entropy')

    model.fit(train_features, train_labels)

    predictions = model.predict(test_features)

    # Performance Metrics
    acc_score = accuracy_score(test_labels, predictions, normalize=True) * 100
    auroc = roc_auc_score(test_labels, predictions)
    tn, fp, fn, tp = confusion_matrix(test_labels, predictions).ravel()
    sens = tp / (tp + fn) * 100
    spec = tn / (tn + fp) * 100

    print("Decision Tree AUROC: ", round(auroc, 2), "%.")
    print("Decision Tree Accuracy: ", round(acc_score, 2), "%.")
    print("Decision Tree Sensitivity: ", round(sens, 2), "%.")
    print("Decision Tree Specificity: ", round(spec, 2), "%.")

    return model, round(auroc, 4), round(acc_score, 2), round(sens, 2), round(spec, 2)


def random_forest(train_features, train_labels, test_features, test_labels):
    model = RandomForestClassifier(bootstrap=True, criterion="gini", max_features=0.2, min_samples_leaf=1,
                                   min_samples_split=20, n_estimators=100)

    model.fit(train_features, train_labels)

    predictions = model.predict(test_features)

    # Performance Metrics
    acc_score = accuracy_score(test_labels, predictions, normalize=True) * 100
    auroc = roc_auc_score(test_labels, predictions)
    tn, fp, fn, tp = confusion_matrix(test_labels, predictions).ravel()
    sens = tp / (tp + fn) * 100
    spec = tn / (tn + fp) * 100

    print("Random Forest AUROC: ", round(auroc, 2), "%.")
    print("Random Forest Accuracy: ", round(acc_score, 2), "%.")
    print("Random Forest Sensitivity: ", round(sens, 2), "%.")
    print("Random Forest Specificity: ", round(spec, 2), "%.")

    return model, round(auroc, 4), round(acc_score, 2), round(sens, 2), round(spec, 2)


def extremely_random_trees(train_features, train_labels, test_features, test_labels):
    model = ExtraTreesClassifier(bootstrap=False, criterion='entropy',
                                 max_features=0.95, min_samples_leaf=5, min_samples_split=14, n_estimators=100)

    model.fit(train_features, train_labels)

    predictions = model.predict(test_features)

    # Performance Metrics
    acc_score = accuracy_score(test_labels, predictions, normalize=True) * 100
    auroc = roc_auc_score(test_labels, predictions)
    tn, fp, fn, tp = confusion_matrix(test_labels, predictions).ravel()
    sens = tp / (tp + fn) * 100
    spec = tn / (tn + fp) * 100

    print("Extremely Randomized Trees AUROC: ", round(auroc, 2), "%.")
    print("Extremely Randomized Trees Accuracy: ", round(acc_score, 2), "%.")
    print("Extremely Randomized Trees Sensitivity: ", round(sens, 2), "%.")
    print("Extremely Randomized Trees Specificity: ", round(spec, 2), "%.")

    return model, round(auroc, 4), round(acc_score, 2), round(sens, 2), round(spec, 2)


def rgf(train_features, train_labels, test_features, test_labels):
    model = RGFClassifier(algorithm='RGF', l2=0.3, max_leaf=2000, min_samples_leaf=5)

    model.fit(train_features, train_labels)

    predictions = model.predict(test_features)

    predictions = (predictions > 0.5)

    # Performance Metrics
    acc_score = accuracy_score(test_labels, predictions, normalize=True) * 100
    auroc = roc_auc_score(test_labels, predictions)
    tn, fp, fn, tp = confusion_matrix(test_labels, predictions).ravel()
    sens = tp / (tp + fn) * 100
    spec = tn / (tn + fp) * 100

    print("RGFClassifier AUROC: ", round(auroc, 2), "%.")
    print("RGFClassifier Accuracy: ", round(acc_score, 2), "%.")
    print("RGFClassifier Sensitivity: ", round(sens, 2), "%.")
    print("RGFClassifier Specificity: ", round(spec, 2), "%.")

    return model, round(auroc, 4), round(acc_score, 2), round(sens, 2), round(spec, 2)

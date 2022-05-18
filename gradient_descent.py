from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler


def sgd(train_features, train_labels, test_features, test_labels):
    scale = StandardScaler()
    scale.fit(train_features)  # fitting of training data to be scaled
    train_features = scale.transform(train_features)
    test_features = scale.transform(test_features)

    model = SGDClassifier(alpha=0.001, eta0=0.001, l1_ratio=0.9,
                          learning_rate='optimal', loss='modified_huber',
                          penalty='l1')

    model.fit(train_features, train_labels)

    predictions = model.predict(test_features)

    # Performance Metrics
    acc_score = accuracy_score(test_labels, predictions, normalize=True) * 100
    auroc = roc_auc_score(test_labels, predictions)
    tn, fp, fn, tp = confusion_matrix(test_labels, predictions).ravel()
    sens = tp / (tp + fn) * 100
    spec = tn / (tn + fp) * 100

    print("Stochastic Gradient Descent Accuracy: ", round(acc_score, 2), "%.")
    print("Stochastic Gradient Descent Sensitivity: ", round(sens, 2), "%.")
    print("Stochastic Gradient Descent Spec: ", round(spec, 2), "%.")

    return model, round(auroc, 4), round(acc_score, 2), round(sens, 2), round(spec, 2)

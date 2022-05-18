from keras import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score


def neural_model(train_features, train_labels, test_features, test_labels):
    model = Sequential()

    # Adding the input layer and the first hidden layer
    model.add(Dense(units=56, kernel_initializer='uniform', activation='relu', input_dim=56))
    # Adding the second hidden layer
    model.add(Dense(units=23, kernel_initializer='uniform', activation='relu'))
    # Adding the third hidden layer
    model.add(Dense(units=11, kernel_initializer='uniform', activation='relu'))

    # Adding the output layer
    model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

    # Compiling Neural Network
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_features, train_labels, batch_size=40, epochs=50, verbose=0)
    predictions = model.predict(test_features)
    predictions = (predictions > 0.5)

    # Performance Metrics
    acc_score = accuracy_score(test_labels, predictions, normalize=True) * 100
    auroc = roc_auc_score(test_labels, predictions)
    tn, fp, fn, tp = confusion_matrix(test_labels, predictions).ravel()
    sens = tp / (tp + fn) * 100
    spec = tn / (tn + fp) * 100

    print("Neural Network AUROC: ", round(auroc, 2), "%.")
    print("Neural Network Accuracy: ", round(acc_score, 2), "%.")
    print("Neural Network Sensitivity: ", round(sens, 2), "%.")
    print("Neural Network Specificity: ", round(spec, 2), "%.")

    return model, round(auroc, 4), round(acc_score, 2), round(sens, 2), round(spec, 2)

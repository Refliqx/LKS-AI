def accuracy_point(y_test, y_pred):
    correct_pred = 0
    total_data = len(y_test)

    for label in y_test:
        if label == y_pred:
            correct_pred += 1
    accuracy = correct_pred / total_data
    return accuracy
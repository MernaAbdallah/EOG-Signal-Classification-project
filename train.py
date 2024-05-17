import preprocessing as ps
import feature_extraction as fe
import models as mod


def make_train(wavlet=False):
    ys, data, label = ps.PreProcessing.label_encode()
    preprocessed_data = []

    for dt in data:
        preprocessed_data.append(ps.PreProcessing.preprocess_signal(dt))

    if not wavlet:
        x_test, y_test, x_train, y_train = fe.FeatureExtraction.statistical_features(ys, preprocessed_data)
    else:
        x_test, y_test, x_train, y_train = fe.FeatureExtraction.apply_wavelet(ys, preprocessed_data)

    models, acc, _, _, train_acc = mod.Models.classify(x_test, y_test, label, x_train, y_train)

    for i in range(len(models)):
        print(f'========== {models[i]} ==========')
        print(f"Test Accuracy: {acc[i]} %")
        print(f"Train Accuracy: {train_acc[i]} %")
        print('================================')

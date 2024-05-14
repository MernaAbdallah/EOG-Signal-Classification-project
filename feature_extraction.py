import numpy as np
import pandas as pd
import pywt


class FeatureExtraction:
    @staticmethod
    def statistical_features(ys, data):
        features = []
        print(data)
        print(ys)
        for index, (y, dt) in enumerate(zip(ys, data)):
            mean, std = dt.mean(axis=1), dt.std(axis=1)
            feature = pd.DataFrame({'meanV': mean, 'stdV': std})
            if index % 2 == 0:
                feature['labelV'] = y
            else:
                feature['labelH'] = y
            features.append(feature)

        # print(pd.DataFrame([features]).reshape(1, -1))

        tt = [0, 1, 3, 4, 2, 5]
        test = pd.concat([features[0], features[1]], axis=1)
        print(test)
        test = test.iloc[:, tt]
        train = pd.concat([features[2], features[3]], axis=1)
        train = train.iloc[:, tt]

        # print(train,test)
        x_test, y_test = test.iloc[:, :-1], test.iloc[:, -1:]
        x_train, y_train = train.iloc[:, :-1], train.iloc[:, -1:]

        return x_test, y_test, x_train, y_train

    @staticmethod
    def apply_wavelet(ys, data, wavelet='haar', level=1, ):
        def wavelet_decomposition(row):
            return pywt.wavedec(row, wavelet, level=level)[0].tolist()

        features = []

        for dt in data:
            features.append(pd.DataFrame(np.array(dt.apply(wavelet_decomposition, axis=1).tolist())))

        test = pd.concat([features[0], features[1]], axis=1)
        train = pd.concat([features[2], features[3]], axis=1)

        test['labelV'] = ys[0]
        train['labelV'] = ys[2]

        x_test, y_test = test.iloc[:, :-1], test.iloc[:, -1:]
        x_train, y_train = train.iloc[:, :-1], train.iloc[:, -1:]

        return x_test, y_test, x_train, y_train

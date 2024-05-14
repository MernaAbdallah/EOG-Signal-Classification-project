import csv

import preprocessing as ps
import feature_extraction as fe
import models as mod
import gui as gui

gui.Gui()
# ys, data, label = ps.PreProcessing.label_encode()
# preprocessed_data = []
#
# for dt in data:
#     preprocessed_data.append(ps.PreProcessing.preprocess_signal(dt))
#
# x_test, y_test, x_train, y_train = fe.FeatureExtraction.statistical_features(ys, preprocessed_data)
# #
# print(y_test)
# models, acc, mse, reports, _ = mod.Models.classify(x_test, y_test, label, x_train, y_train)
#
# for i in range(len(models)):
#     print(f'========== {models[i]} ==========')
#     print(f"Test Accuracy: {acc[i]} %")
#     print('================================')

import csv

import preprocessing as ps
import feature_extraction as fe
import models as mod
import gui as gui

#gui.Gui()

ys, data = ps.PreProcessing.label_encode()
preprocessed_data = []

for dt in data:
    preprocessed_data.append(ps.PreProcessing.preprocess_signal(dt))

x_test, y_test, x_train, y_train = fe.FeatureExtraction.statistical_features(ys, preprocessed_data)
# #
# # mod.Models.gridSearch(x_test, y_test, x_train, y_train)
#
# models, acc, mse, reports, train_acc = mod.Models.classify(x_test, y_test, x_train, y_train)
#
# for i in range(len(models)):
#     print(f'========== {models[i]} ==========')
#     print(f'Train Accuracy  = {train_acc[i]}')
#     print(f"Test Accuracy: {acc[i]} %")
#     print('================================')

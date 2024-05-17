from collections import Counter

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import pickle


class Models:

    @staticmethod
    def __save_model(model_name, clf):
        with open(f'models/{model_name}.pkl', 'wb') as file:
            pickle.dump(clf, file)

    @staticmethod
    def __get_saved_model(model_name):
        with open(f'models/{model_name}.pkl', 'rb') as file:
            return pickle.load(file)

    @staticmethod
    def classify(x_test, y_test, label_encode, x_train=None, y_train=None, train=True):
        # Identify Models Names
        models = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'SVM Linear Kernel',
                  'SVM RPF Kernel', 'Gaussian Naive Bayes', 'AdaBoost', 'Gradient Boost']

        # Identify Models Classifiers
        clf = [LogisticRegression(max_iter=50, random_state=42, C=0.1, penalty='l2', solver='newton-cg'),
               DecisionTreeClassifier(max_features='sqrt', max_depth=15, min_samples_leaf=2, min_samples_split=10),
               RandomForestClassifier(max_depth=5, min_samples_leaf=1, min_samples_split=5,
                                      n_estimators=50, max_features='log2'),
               SVC(kernel='linear', C=10, degree=2, gamma='scale'),
               SVC(kernel='rbf', C=10, gamma=0.001),

               GaussianNB(),
               AdaBoostClassifier(learning_rate=0.01, n_estimators=50, algorithm='SAMME.R'),
               GradientBoostingClassifier(learning_rate=0.01, max_depth=3, n_estimators=50, max_features='sqrt',
                                          min_samples_leaf=2, min_samples_split=5), ]

        acc, predictions, train_acc = [], [], [],

        # Loop on Models to Determine the Classifier to Work With
        for i in range(len(models)):

            # Check whether we are in test scenario or train one
            if train:
                # In Train scenario
                # Train the Model
                working_clf = clf[i]
                working_clf.fit(x_train, y_train)
                # Save the Model in pickle to use it in testing
                Models.__save_model(models[i], working_clf)
            else:
                # In Test scenario
                # Load the Trained Model from pickle
                working_clf = Models.__get_saved_model(models[i])

            # Get the accuracy of the trained model after predicting
            y_pred = working_clf.predict(x_test)
            predictions.append(label_encode.inverse_transform(y_pred))
            acc.append(accuracy_score(y_test, y_pred) * 100)
            if train:
                y_pred_train = working_clf.predict(x_train)
                train_acc.append(accuracy_score(y_train, y_pred_train) * 100)

        final_predictions = None

        if not train:
            files = [[] for _ in range(5)]

            for pred in predictions:
                for i, val in enumerate(pred):
                    files[i].append(val)

            final_predictions = []
            mapping = {
                'yukari': 'up',
                'yukarı': 'up',
                'asagi': 'down',
                'sag': 'right',
                'sol': 'left',
                'kirp': 'blink'
            }

            for file in files:
                prediction_counts = Counter(file)
                most_common_prediction = prediction_counts.most_common(1)[0][0]
                mapped_prediction = mapping.get(most_common_prediction.lower())
                final_predictions.append(mapped_prediction)

        return models, acc, final_predictions, predictions, train_acc

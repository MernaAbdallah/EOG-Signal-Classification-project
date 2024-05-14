from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
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
    def gridSearch(x_test, y_test, x_train, y_train):
        models = {
            'Gradient Boosting': GradientBoostingClassifier()
        }

        # Define expanded parameters for grid search for each model
        params = {
            'Gradient Boosting': {'n_estimators': [50, 100, 200, 300],
                                  'learning_rate': [0.01, 0.1, 1],
                                  'max_depth': [3, 5, 7, 9],
                                  'min_samples_split': [2, 5, 10],
                                  'min_samples_leaf': [1, 2, 4],
                                  'max_features': ['auto', 'sqrt', 'log2']}
        }

        # Perform GridSearchCV for each model
        for name, model in models.items():
            print(f"Grid search CV for {name}")
            grid_search = GridSearchCV(model, params[name], cv=5, n_jobs=-1)
            grid_search.fit(x_train, y_train)
            print("Best parameters:", grid_search.best_params_)
            print("Best cross-validation score:", grid_search.best_score_)
            print("Test set score:", grid_search.score(x_test, y_test))
            print()

    @staticmethod
    def classify(x_test, y_test, x_train, y_train, train=True):
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
               GradientBoostingClassifier(learning_rate=0.1, max_depth=5, n_estimators=50), ]

        acc, mse, reports, train_acc = [], [], [], []

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
                working_clf, train_time = Models.__get_saved_model(models[i])

            # Get the accuracy of the trained model after predicting
            y_pred = working_clf.predict(x_test)
            y_pred_train = working_clf.predict(x_train)
            mse.append(mean_squared_error(y_test, y_pred))
            acc.append(accuracy_score(y_test, y_pred) * 100)
            train_acc.append(accuracy_score(y_train, y_pred_train) * 100)
            reports.append(classification_report(y_test, y_pred))

        return models, acc, mse, reports, train_acc

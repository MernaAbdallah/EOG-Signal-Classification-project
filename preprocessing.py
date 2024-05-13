import re

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import LabelEncoder, StandardScaler


class PreProcessing:

    @staticmethod
    def __read_csv_to_dataframe(csv_file_path):
        try:
            # Read the CSV file into a DataFrame without header
            df = pd.read_csv(csv_file_path, header=None)

            # Set column names as numbers
            df.columns = range(len(df.columns))

            return df
        except Exception as e:
            print(f"Error occurred while reading CSV file: {e}")
            return None

    @staticmethod
    def __get_train_test_data():
        dfs, csv_paths = [], ['Test-V_CSV.csv', 'Test-H_CSV.csv', 'Train-V_CSV.csv', 'Train-H_CSV.csv']

        for csv_path in csv_paths:
            df = PreProcessing.__read_csv_to_dataframe(csv_path)
            df = df.transpose()
            df = df.rename(columns={df.columns[-1]: 'label'})
            dfs.append(df)

        return dfs

    @staticmethod
    def __butter_bandpass_filter(input_signal, low_cutoff, high_cutoff, sampling_rate, order):
        nyq = 0.5 * sampling_rate
        low = low_cutoff / nyq
        high = high_cutoff / nyq
        wn = [low, high]
        numerator, denominator = butter(order, wn, btype="band", output="ba", analog=False, fs=None)
        filtered = filtfilt(numerator, denominator, input_signal)
        return filtered

    @staticmethod
    def label_encode():
        dfs = PreProcessing.__get_train_test_data()
        le = LabelEncoder()
        ys, data = [], [],

        for df in dfs:
            y = df.iloc[:, -1]
            data.append(df.drop(columns="label", axis=1).astype(float))
            y = y.apply(lambda x: re.search(r'\D+', x).group(0))
            le.fit(y)
            ys.append(le.transform(y))

        return ys, data

    @staticmethod
    def preprocess_signal(train_set):
        # Apply Butterworth bandpass filter
        filtered_signal = PreProcessing.__butter_bandpass_filter(train_set, low_cutoff=0.5, high_cutoff=20.0,
                                                                 sampling_rate=176, order=2)
        # Convert filtered_signal to numpy array
        data = np.array(filtered_signal)
        data = data.astype(float)

        # Subtract mean
        data_dct = data - np.mean(data)
        data_dct = pd.DataFrame(data_dct)

        # Standardize the data and Create a new DataFrame with standardized data
        return pd.DataFrame(StandardScaler().fit_transform(data_dct), columns=data_dct.columns)

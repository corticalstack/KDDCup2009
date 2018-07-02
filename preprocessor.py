# Author:   Jon-Paul Boyd
# Date:     16/01/2018
# IMAT5234  Applied Computational Intelligence - Mini Project
# Customer Relationship Management - Predict churn, appetency and upselling on Orange dataset
# Preprocessor - prepare dataset
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from filehandler import Filehandler
import visualizer


class Preprocessor:

    def __init__(self, envparm):
        self.dataset_raw = None
        self.numerical_features_raw = None
        self.categorical_features_raw = None
        self.filehandler = None
        self.preprocess(envparm)

    @staticmethod
    def drop_features_min_unique(dataset, min_threshold):
        features_dropped_str = ''
        for col in dataset:
            if len(dataset[col].unique()) <= min_threshold:
                features_dropped_str += str(col) + ' '
                dataset.drop(col, inplace=True, axis=1)

        logging.info('Features dropped with unique value count <= {} - {}'.format(min_threshold, features_dropped_str))
        return dataset

    @staticmethod
    def drop_features_max_unique(dataset, max_threshold):
        features_dropped_str = ''
        for col in dataset:
            if len(dataset[col].unique()) >= max_threshold:
                features_dropped_str += str(col) + ' '
                dataset.drop(col, inplace=True, axis=1)

        logging.info('Features dropped with unique value count >= {} - {}'.format(max_threshold, features_dropped_str))
        return dataset

    @staticmethod
    def drop_features_max_null(dataset, max_threshold):
        features_dropped_str = ''
        for col in dataset:
            if sum(dataset[col].isnull()) >= max_threshold:
                features_dropped_str += str(col) + ' '
                dataset.drop(col, inplace=True, axis=1)

        logging.info('Features dropped with null value count >= {} - {}'.format(max_threshold, features_dropped_str))
        return dataset

    @staticmethod
    def plot_num_obs_missing_values(dataset):
        df = pd.DataFrame(data=dataset.isnull().sum(), columns=['Count'])
        df['bin'] = pd.cut(df['Count'], [-1, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000,
                                         6000, 7000, 8000, 9000, 10000,  50000], labels=['0-10', '10-20', '20-30',
                                                                                         '30-40', '40-50', '50-100',
                                                                                         '100-200', '200-300',
                                                                                         '300-400', '400-500',
                                                                                         '500-1K', '1K-2K',
                                                                                         '2K-3K', '3K-4K',
                                                                                         '4K-5K', '5K-6K',
                                                                                         '6K-7K', '7K-8K',
                                                                                         '8K-9K', '9K-10K',
                                                                                         '10K-50K'])
        countplot = sns.countplot(y="bin", data=df)
        countplot.set(ylabel="Observations With Null Values", xlabel="Feature Count",
                      title="Observations With Null Values Per Feature")
        plt.show()

    @staticmethod
    def impute_numeric_feature_with_zero(dataset):
        dataset.fillna(0, inplace=True)

    @staticmethod
    def impute_categorical_feature_with_blank(dataset):
        dataset.fillna('', inplace=True)

    def prepare_output_variant_01(self):
        logging.info('Preparing output variant 01')
        numerical_features = self.numerical_features_raw.copy()

        logging.info('Validating numerical features')
        numerical_features = self.drop_features_min_unique(numerical_features, 1)

        logging.info('Validating categorical features')
        categorical_features = self.categorical_features_raw.copy()
        categorical_features = self.drop_features_min_unique(categorical_features, 1)

        logging.info('Imputing numerical features with mean')
        numerical_features.fillna(numerical_features.mean(), inplace=True)

        logging.info('Imputing categorical features with "missing"')
        categorical_features.fillna('missing', inplace=True)

        # Random Forest needs the categorical features encoding otherwise string to float error
        logging.info('Label encoding categorical features')
        labelencoder_categorical = LabelEncoder()
        labelencoder_categorical = categorical_features.apply(labelencoder_categorical.fit_transform)

        dataset_output = self.filehandler.output_prep_dataset(self.filehandler.dataset_prep_path_01, numerical_features,
                                                              labelencoder_categorical)

        logging.info('Dataset size after feature transformation - {}'.format(dataset_output.shape))
        logging.info('Completed Preparing output variant 01')

    def prepare_output_variant_02(self):
        logging.info('Preparing output variant 02')
        numerical_features = self.numerical_features_raw.copy()

        logging.info('Validating numerical features')
        numerical_features = self.drop_features_min_unique(numerical_features, 1)

        logging.info('Validating categorical features')
        categorical_features = self.categorical_features_raw.copy()
        categorical_features = self.drop_features_min_unique(categorical_features, 1)

        logging.info('Imputing numerical features with zero')
        numerical_features.fillna(0, inplace=True)

        logging.info('Imputing categorical features with "missing"')
        categorical_features.fillna('missing', inplace=True)

        # Random Forest needs the categorical features encoding otherwise string to float error
        logging.info('Label encoding categorical features')
        labelencoder_categorical = LabelEncoder()
        labelencoder_categorical = categorical_features.apply(labelencoder_categorical.fit_transform)

        dataset_output = self.filehandler.output_prep_dataset(self.filehandler.dataset_prep_path_02, numerical_features,
                                                              labelencoder_categorical)

        logging.info('Dataset size after feature transformation - {}'.format(dataset_output.shape))
        logging.info('Completed Preparing output variant 02')

    def prepare_output_variant_03(self):
        logging.info('Preparing output variant 03')
        numerical_features = self.numerical_features_raw.copy()

        logging.info('Validating numerical features')
        numerical_features = self.drop_features_min_unique(numerical_features, 1)

        logging.info('Validating categorical features')
        categorical_features = self.categorical_features_raw.copy()
        categorical_features = self.drop_features_min_unique(categorical_features, 1)
        categorical_features = self.drop_features_max_unique(categorical_features, 11)

        logging.info('Imputing numerical features with -9876')
        numerical_features.fillna(-9876, inplace=True)

        logging.info('Imputing categorical features with "missing"')
        categorical_features.fillna('missing', inplace=True)

        # Random Forest needs the categorical features encoding otherwise string to float error
        logging.info('Label encoding categorical features')
        labelencoder_categorical = LabelEncoder()
        labelencoder_categorical = categorical_features.apply(labelencoder_categorical.fit_transform)

        # One hot encoding results in memory error due to wide dataset
        onehotencoder = OneHotEncoder()
        labelencoder_categorical = onehotencoder.fit_transform(labelencoder_categorical).toarray()
        labelencoder_categorical_df = pd.DataFrame(labelencoder_categorical)

        dataset_output = self.filehandler.output_prep_dataset(self.filehandler.dataset_prep_path_03, numerical_features,
                                                              labelencoder_categorical_df)

        logging.info('Dataset size after feature transformation - {}'.format(dataset_output.shape))
        logging.info('Completed Preparing output variant 03')

    def preprocess(self, envparm):
        self.filehandler = Filehandler()
        self.dataset_raw = self.filehandler.read_csv(self.filehandler.data_raw_path)
        logging.info('Original raw dataset loaded - dataset size {}'.format(self.dataset_raw.shape))

        logging.info('Partitioning numerical features')
        self.numerical_features_raw = self.dataset_raw.iloc[:, 0:190].copy()

        logging.info('Partitioning categorical features')
        self.categorical_features_raw = self.dataset_raw.iloc[:, 190:].copy()

        if envparm['PlotGraphs']:
            num_size = 28
            sample_df = self.dataset_raw.iloc[:, :num_size].copy()
            visualizer.matrix_missing(sample_df, 'Data Completion First ' + str(num_size) + ' Numeric Features')
            visualizer.bar_missing(sample_df, 'Nullity Count First ' + str(num_size) + ' Numeric Features')
            visualizer.heat_missing(sample_df, 'Nullity Correlation Of First ' + str(num_size) + ' Numeric Features')

            sample_df = self.dataset_raw.iloc[:, 190:].copy()
            visualizer.matrix_missing(sample_df, 'Data Completion Categorical Features')
            visualizer.bar_missing(sample_df, 'Nullity Count Categorical Features')
            visualizer.heat_missing(sample_df, 'Nullity Correlation Of Categorical Features')

        if envparm['ProcessDS01']:
            self.prepare_output_variant_01()

        if envparm['ProcessDS02']:
            self.prepare_output_variant_02()

        if envparm['ProcessDS03']:
            self.prepare_output_variant_03()

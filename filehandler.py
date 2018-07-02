# Author:   Jon-Paul Boyd
# Date:     16/01/2018
# IMAT5234  Applied Computational Intelligence - Mini Project
# Customer Relationship Management - Predict churn, appetency and upselling on Orange dataset
# File Handler
import logging
from os import path
import pandas as pd


class Filehandler:
    """Represent constants"""

    def __init__(self):
        self.path_data = 'data'
        self.file_raw_dataset = 'orange_small_train.csv'
        self.file_prep_dataset_01 = 'preprocessed_01_small_train.csv'
        self.file_prep_dataset_02 = 'preprocessed_02_small_train.csv'
        self.file_prep_dataset_03 = 'preprocessed_03_small_train.csv'
        self.file_scores = 'scores.csv'
        self.file_l_churn = 'orange_small_train_churn.labels.csv'
        self.file_l_appetency = 'orange_small_train_appetency.labels.csv'
        self.file_l_upselling = 'orange_small_train_upselling.labels.csv'
        self.file_feature_ranking = 'Feature ranking'
        self.data_raw_path = None
        self.l_churn_path = None
        self.l_appetency_path = None
        self.l_upselling_path = None
        self.dataset_prep_path_01 = None
        self.dataset_prep_path_02 = None
        self.dataset_prep_path_03 = None
        self.scores_path = None
        self.feature_ranking_path = None
        self.set_paths()

    def set_paths(self):
        self.data_raw_path = path.join(self.path_data, self.file_raw_dataset)
        self.l_churn_path = path.join(self.path_data, self.file_l_churn)
        self.l_appetency_path = path.join(self.path_data, self.file_l_appetency)
        self.l_upselling_path = path.join(self.path_data, self.file_l_upselling)
        self.feature_ranking_path = path.join(self.path_data, self.file_feature_ranking)
        self.dataset_prep_path_01 = path.join(self.path_data, self.file_prep_dataset_01)
        self.dataset_prep_path_02 = path.join(self.path_data, self.file_prep_dataset_02)
        self.dataset_prep_path_03 = path.join(self.path_data, self.file_prep_dataset_03)
        self.scores_path = path.join(self.path_data, self.file_scores)

    def read_csv(self, datapath, names=None):
        logging.info("Reading file - {}".format(datapath))
        dataset = pd.read_csv(datapath, names=names)
        return dataset

    def output_prep_dataset(self, prep_path, num_dataset, cat_dataset):
        logging.info("Outputing preparation file - {}".format(prep_path))
        dataset_concat = pd.concat([num_dataset, cat_dataset], axis=1)
        dataset_concat.to_csv(prep_path, index=False)
        return dataset_concat

    def output_scores(self, scores):
        logging.info("Outputing scores file - {}".format(self.scores_path))
        df = pd.DataFrame(scores)
        df.to_csv(self.scores_path, header=None)

    def output_feature_ranking(self, feature_ranking, target_label, clfname, scoring_variant):
        full_path = self.feature_ranking_path + ' - ' + target_label + ' - ' + clfname + ' - ' + \
                    scoring_variant + '.csv'
        logging.info("Outputing feature ranking file - {}".format(full_path))
        df = pd.DataFrame(feature_ranking)
        df.index.name = 'Rank'
        df.columns = ['Feature', 'Importance']
        df.to_csv(full_path)

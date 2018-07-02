# Author:   Jon-Paul Boyd
# Date:     16/01/2018
# IMAT5234  Applied Computational Intelligence - Mini Project
# Customer Relationship Management - Predict churn, appetency and upselling on Orange dataset
# Modeller

import logging
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier,\
    VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import numpy as np
import visualizer
import os
os.environ["PATH"] += os.pathsep + 'D:/Program Files (x86)/graphviz-2.38/release/bin/'

class Modeller:

    def __init__(self):
        self.dataset_train = None
        self.target_train = None
        self.dataset_test = None
        self.target_test = None
        self.test_size = 0.2
        self.random_state = 20
        self.target_label = None
        self.scores = []
        self.score_count = 0
        self.dtcname = 'Decision Tree Classifier'
        self.dtc = 'DTC'
        self.rfcname = 'Random Forest Classifier'
        self.rfc = 'RFC'
        self.bgcname = 'Bagging Classifier'
        self.bgc = 'BGC'
        self.abcname = 'AdaBoost Classifier'
        self.abc = 'ABC'
        self.gbcname = 'Gradient Boosting Classifier'
        self.gbc = 'GBC'
        self.vtcname = 'Voting Classifier'
        self.vtc = 'VTC'

    def fit_predict_score(self, clf, clf_name, clf_name_short, gs_param, envparm=None):
        clf.fit(self.dataset_train, self.target_train)

        if gs_param == 'Baseline' or gs_param == 'Final':
            predict_proba = clf.predict_proba(self.dataset_test)[:, 1]
        else:
            predict_proba = clf.best_estimator_.predict_proba(self.dataset_test)[:, 1]

        score_auc = roc_auc_score(self.target_test, predict_proba)

        if gs_param == 'Baseline' or gs_param == 'Final':
            self.score_count += 1
            self.scores.append([self.score_count, clf_name, clf_name_short, 'roc_auc_score', gs_param,
                                str(clf).split('(')[1].replace(",", ' ').
                                replace('\r', '').replace('\n', '').replace("             ", " "),
                                self.target_label, score_auc])
            logging.info("Score {} model - {} - {} - {}".format(gs_param, clf_name, self.target_label, score_auc))
            predict = clf.predict(self.dataset_test)
            cm = confusion_matrix(self.target_test, predict)
            logging.info(
                "Confusion matrix for {} {} - TN {}  FN {}  TP {}  FP {}".format(clf_name, gs_param, cm[0][0], cm[1][0],
                                                                                 cm[1][1], cm[0][1]))
            threshold_adj = 0.09
            predict_threshold_adj = np.where(predict_proba > threshold_adj, 1, -1)
            score_auc_adj = roc_auc_score(self.target_test, predict_threshold_adj)
            logging.info("Score (threshold {}) {} model - {} - {} - {}".format(threshold_adj, gs_param, clf_name,
                                                                               self.target_label, score_auc_adj))
            cm_adj = confusion_matrix(self.target_test, predict_threshold_adj)
            logging.info(
                "Confusion matrix (adjusted) for {} {} - TN {}  FN {}  TP {}  FP {}".format(clf_name, gs_param,
                                                                                            cm_adj[0][0], cm_adj[1][0],
                                                                                            cm_adj[1][1], cm_adj[0][1]))

            if envparm['PlotGraphs']:
                visualizer.confusion_matrix(cm, self.target_label, clf_name, gs_param)
                visualizer.confusion_matrix(cm_adj, self.target_label, clf_name, gs_param + ' Threshold ' +
                                            str(threshold_adj))
            return

        self.score_count += 1
        self.scores.append([self.score_count, clf_name, clf_name_short,  'roc_auc_score', 'GS Best', clf.best_params_,
                            self.target_label, score_auc])
        scores_mean = clf.cv_results_['mean_test_score']

        if gs_param == 'IsMulti':
            self.score_count += 1
            self.scores.append([self.score_count, clf_name, clf_name_short, 'GS Mean', 'Multi', clf.best_params_,
                                self.target_label, score_auc])
        else:
            for mean, params in zip(scores_mean, clf.cv_results_['params']):
                self.score_count += 1
                self.scores.append([self.score_count, clf_name, clf_name_short, 'GS Mean', gs_param, params[gs_param],
                                    self.target_label, mean])

        logging.info("Score grid search best model - {} - {} - {} - {}".format(clf_name, clf.best_params_,
                                                                               self.target_label, score_auc))

    def feature_ranking(self, clf, clfname, scoring_variant, filehandler):
        feature_ranking = list()
        importances = clf.feature_importances_
        indices = np.argsort(importances)[::-1]

        for feature in range(self.dataset_train.shape[1]):
            feature_ranking.append([indices[feature], importances[indices[feature]]])
        filehandler.output_feature_ranking(feature_ranking, self.target_label, clfname, scoring_variant)

    def score_baseline_dtc(self, envparm):
        clf = DecisionTreeClassifier(random_state=self.random_state)
        self.fit_predict_score(clf, self.dtcname, self.dtc, 'Baseline', envparm)

        if envparm['PlotGraphs']:
            visualizer.graph_decision_tree(clf, self.dataset_train, self.dtcname + ' - baseline - ')

    def get_gridsearch_single_dtc(self, param_grid):
        grid_search = GridSearchCV(estimator=DecisionTreeClassifier(random_state=self.random_state),
                                   param_grid=param_grid,
                                   scoring='roc_auc', n_jobs=8, iid=False, cv=3, verbose=0)
        return grid_search

    def get_gridsearch_multi_dtc(self, param_grid):
        grid_search = GridSearchCV(
            estimator=DecisionTreeClassifier(random_state=self.random_state, max_features=None,
                                             class_weight={-1: 1, 1: 9}),
            param_grid=param_grid, scoring='roc_auc', n_jobs=8, iid=False, cv=3, verbose=0)
        return grid_search

    def gridsearch_single_dtc(self):
        # Class weight
        gs_param = 'class_weight'
        param_grid = {gs_param: ['balanced', {-1: 1, 1: 9}, {-1: 1, 1: 19}, {-1: 1, 1: 1}, {-1: 1, 1: 2}]}
        grid_search = self.get_gridsearch_single_dtc(param_grid)
        self.fit_predict_score(grid_search, self.dtcname, self.dtc, gs_param)

        # Criterion
        gs_param = 'criterion'
        param_grid = {gs_param: ["gini", "entropy"]}
        grid_search = self.get_gridsearch_single_dtc(param_grid)
        self.fit_predict_score(grid_search, self.dtcname, self.dtc, gs_param)

        # Max features
        gs_param = 'max_features'
        param_grid = {gs_param: [None, 'auto', 'sqrt', 'log2']}
        grid_search = self.get_gridsearch_single_dtc(param_grid)
        self.fit_predict_score(grid_search, self.dtcname, self.dtc, gs_param)

        # Max depth
        gs_param = 'max_depth'
        param_grid = {gs_param: [None, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
        grid_search = self.get_gridsearch_single_dtc(param_grid)
        self.fit_predict_score(grid_search, self.dtcname, self.dtc, gs_param)

        # Min samples split
        gs_param = 'min_samples_split'
        param_grid = {gs_param: range(20, 1020, 20)}
        grid_search = self.get_gridsearch_single_dtc(param_grid)
        self.fit_predict_score(grid_search, self.dtcname, self.dtc, gs_param)

        # Min samples leaf
        gs_param = 'min_samples_leaf'
        param_grid = {gs_param: range(30, 80, 2)}
        grid_search = self.get_gridsearch_single_dtc(param_grid)
        self.fit_predict_score(grid_search, self.dtcname, self.dtc, gs_param)

        # Max leaf nodes
        gs_param = 'max_leaf_nodes'
        param_grid = {gs_param: [None, 5, 10, 15, 20, 25, 30, 35, 40]}
        grid_search = self.get_gridsearch_single_dtc(param_grid)
        self.fit_predict_score(grid_search, self.dtcname, self.dtc, gs_param)

    def gridsearch_multi_dtc(self):
        param_grid = {'criterion': ["gini", "entropy"], 'max_depth': [5, 6, 7], 'max_features': [None, 'auto'],
                      'max_leaf_nodes': range(10, 30, 5), 'min_samples_leaf': range(70, 80, 2),
                      'min_samples_split': [500, 700, 900]}
        grid_search = self.get_gridsearch_multi_dtc(param_grid)
        self.fit_predict_score(grid_search, self.dtcname, self.dtc, 'IsMulti')

    def score_final_dtc_churn(self, envparm):
        clf = DecisionTreeClassifier(random_state=self.random_state, criterion='gini', max_features=None, max_depth=5,
                                     max_leaf_nodes=25, min_samples_leaf=43, min_samples_split=70)
        self.fit_predict_score(clf, self.dtcname, self.dtc, 'Final', envparm)

    def score_final_dtc_appetency(self, envparm):
        clf = DecisionTreeClassifier(random_state=self.random_state, criterion='entropy', max_features=None,
                                     max_depth=5, max_leaf_nodes=10, min_samples_leaf=43, min_samples_split=46)
        self.fit_predict_score(clf, self.dtcname, self.dtc, 'Final', envparm)

    def score_final_dtc_upselling(self, envparm):
        clf = DecisionTreeClassifier(random_state=self.random_state, criterion='entropy', max_features=None,
                                     max_depth=5, max_leaf_nodes=20, min_samples_leaf=43, min_samples_split=46)
        self.fit_predict_score(clf, self.dtcname, self.dtc, 'Final', envparm)

    def score_baseline_rfc(self, envparm, filehandler,):
        clf = RandomForestClassifier(random_state=self.random_state)
        self.fit_predict_score(clf, self.rfcname, self.rfc, 'Baseline', envparm)
        self.feature_ranking(clf, self.rfcname, 'Baseline', filehandler)

    def get_gridsearch_single_rfc(self, param_grid):
        grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=self.random_state),
                                   param_grid=param_grid, scoring='roc_auc', n_jobs=8, iid=False, cv=3, verbose=0)
        return grid_search

    def get_gridsearch_multi_rfc(self, param_grid):
        grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=self.random_state, criterion='gini',
                                        n_estimators=1900, max_leaf_nodes=25, max_features=None, min_samples_leaf=43,
                                        class_weight={-1: 1, 1: 9}), param_grid=param_grid, scoring='roc_auc',
                                        n_jobs=8, iid=False, cv=3, verbose=0)
        return grid_search

    def gridsearch_single_rfc(self):
        # Class weight
        gs_param = 'class_weight'
        param_grid = {gs_param: ['balanced', 'balanced_subsample', {-1: 1, 1: 9}, {-1: 1, 1: 19}, {-1: 1, 1: 1},
                                 {-1: 1, 1: 2}]}
        grid_search = self.get_gridsearch_single_rfc(param_grid)
        self.fit_predict_score(grid_search, self.rfcname, self.rfc, gs_param)

        # Criterion
        gs_param = 'criterion'
        param_grid = {gs_param: ["gini", "entropy"]}
        grid_search = self.get_gridsearch_single_rfc(param_grid)
        self.fit_predict_score(grid_search, self.rfcname, self.rfc, gs_param)

        # N estimators
        gs_param = 'n_estimators'
        param_grid = {gs_param: [400, 500, 600, 700, 1000, 1200, 1500, 1700, 1900, 2000]}
        grid_search = self.get_gridsearch_single_rfc(param_grid)
        self.fit_predict_score(grid_search, self.rfcname, self.rfc, gs_param)

        # Max features
        gs_param = 'max_features'
        param_grid = {gs_param: [None, 'auto', 'sqrt', 'log2']}
        grid_search = self.get_gridsearch_single_rfc(param_grid)
        self.fit_predict_score(grid_search, self.rfcname, self.rfc, gs_param)

        # Max depth
        gs_param = 'max_depth'
        param_grid = {gs_param: [None, 2, 3, 4, 5]}
        grid_search = self.get_gridsearch_single_rfc(param_grid)
        self.fit_predict_score(grid_search, self.rfcname, self.rfc, gs_param)

        # min_samples_leaf
        gs_param = 'min_samples_leaf'
        param_grid = {gs_param: range(29, 51, 2)}
        grid_search = self.get_gridsearch_single_rfc(param_grid)
        self.fit_predict_score(grid_search, self.rfcname, self.rfc, gs_param)

        # max_leaf_nodes
        gs_param = 'max_leaf_nodes'
        param_grid = {gs_param: [None, 20, 25, 30, 35, 40]}
        grid_search = self.get_gridsearch_single_rfc(param_grid)
        self.fit_predict_score(grid_search, self.rfcname, self.rfc, gs_param)

        # min_samples_split
        gs_param = 'min_samples_split'
        param_grid = {gs_param: range(20, 1020, 20)}
        grid_search = self.get_gridsearch_single_rfc(param_grid)
        self.fit_predict_score(grid_search, self.rfcname, self.rfc, gs_param)

    def gridsearch_multi_rfc(self):
        param_grid = {'max_depth': [3, 4, 5], 'min_samples_split': [640, 790, 940]}
        grid_search = self.get_gridsearch_multi_rfc(param_grid)
        self.fit_predict_score(grid_search, self.rfcname, self.rfc, 'IsMulti')

    def score_final_rfc_churn(self, envparm, filehandler):
        clf = RandomForestClassifier(random_state=self.random_state, criterion='gini', max_features=None, max_depth=5,
                                     max_leaf_nodes=25, min_samples_leaf=39, min_samples_split=46, n_estimators=500)
        self.fit_predict_score(clf, self.rfcname, self.rfc, 'Final', envparm)
        self.feature_ranking(clf, self.rfcname, 'Final', filehandler)

    def score_final_rfc_appetency(self, envparm, filehandler):
        clf = RandomForestClassifier(random_state=self.random_state, criterion='entropy', max_features=None, max_depth=2,
                                     max_leaf_nodes=25, min_samples_leaf=39, min_samples_split=46, n_estimators=700)
        self.fit_predict_score(clf, self.rfcname, self.rfc,  'Final', envparm)
        self.feature_ranking(clf, self.rfcname, 'Final', filehandler)

    def score_final_rfc_upselling(self, envparm, filehandler):
        clf = RandomForestClassifier(random_state=self.random_state, criterion='entropy', max_features=None,
                                     max_depth=5, max_leaf_nodes=35, min_samples_leaf=31, min_samples_split=46,
                                     n_estimators=500)
        self.fit_predict_score(clf, self.rfcname, self.rfc, 'Final', envparm)
        self.feature_ranking(clf, self.rfcname, 'Final', filehandler)

    def score_baseline_bgc(self, envparm):
        clf = BaggingClassifier(random_state=self.random_state)
        self.fit_predict_score(clf, self.bgcname, self.bgc, 'Baseline', envparm)

    def get_gridsearch_single_bgc(self, param_grid):
        grid_search = GridSearchCV(
            estimator=BaggingClassifier(DecisionTreeClassifier(random_state=self.random_state, criterion='gini',
                                    max_leaf_nodes=25, min_samples_leaf=43, min_samples_split=70),
                                    random_state=self.random_state),
            param_grid=param_grid, scoring='roc_auc', n_jobs=8, iid=False, cv=3, verbose=0)
        return grid_search

    def get_gridsearch_multi_bgc(self, param_grid):
        grid_search = GridSearchCV(
            estimator=BaggingClassifier(DecisionTreeClassifier(random_state=self.random_state, criterion='gini',
                                    max_features=None, max_depth=5, max_leaf_nodes=25, min_samples_leaf=43,
                                    min_samples_split=70), random_state=self.random_state, n_estimators=500),
                                    param_grid=param_grid,
            scoring='roc_auc', n_jobs=8, iid=False, cv=3, verbose=0)
        return grid_search

    def gridsearch_single_bgc(self):
        # N estimators
        gs_param = 'n_estimators'
        param_grid = {gs_param: [10, 100, 200, 300, 400, 500, 1000, 1200, 1500]}
        grid_search = self.get_gridsearch_single_bgc(param_grid)
        self.fit_predict_score(grid_search, self.bgcname, self.bgc, gs_param)

        # Max depth
        gs_param = 'max_depth'
        param_grid = {gs_param: [None, 2, 3, 4, 5]}
        grid_search = self.get_gridsearch_single_bgc(param_grid)
        self.fit_predict_score(grid_search, self.bgcname, self.bgc, gs_param)

        # Max samples
        gs_param = 'max_samples'
        param_grid = {gs_param: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
        grid_search = self.get_gridsearch_single_bgc(param_grid)
        self.fit_predict_score(grid_search, self.bgcname, self.bgc, gs_param)

        # Max features
        gs_param = 'max_features'
        param_grid = {gs_param: [0.1, 0.5, 1.0]}
        grid_search = self.get_gridsearch_single_bgc(param_grid)
        self.fit_predict_score(grid_search, self.bgcname, self.bgc, gs_param)

    def gridsearch_multi_bgc(self):
        param_grid = {'max_samples': [0.5, 0.6, 1.0]}
        grid_search = self.get_gridsearch_multi_bgc(param_grid)
        self.fit_predict_score(grid_search, self.bgcname, self.bgc, 'IsMulti')

    def score_final_bgc_churn(self, envparm):
        clf = BaggingClassifier(
            DecisionTreeClassifier(random_state=self.random_state, criterion='gini', max_features=None, max_depth=5,
                                   max_leaf_nodes=25, min_samples_leaf=43, min_samples_split=70), n_estimators=500,
                                   max_features=1.0, max_samples=0.6)
        self.fit_predict_score(clf, self.bgcname, self.bgc, 'Final', envparm)

    def score_final_bgc_appetency(self, envparm):
        clf = BaggingClassifier(
            DecisionTreeClassifier(random_state=self.random_state, criterion='entropy', max_features=None, max_depth=2,
                                   max_leaf_nodes=10, min_samples_leaf=43, min_samples_split=46), n_estimators=700,
                                   max_features=1.0, max_samples=0.6)
        self.fit_predict_score(clf, self.bgcname, self.bgc, 'Final', envparm)

    def score_final_bgc_upselling(self, envparm):
        clf = BaggingClassifier(
            DecisionTreeClassifier(random_state=self.random_state, criterion='entropy', max_features=None, max_depth=5,
                                   max_leaf_nodes=20, min_samples_leaf=43, min_samples_split=46), n_estimators=600,
                                   max_features=1.0, max_samples=0.6)
        self.fit_predict_score(clf, self.bgcname, self.bgc, 'Final', envparm)

    def score_baseline_abc(self, envparm, filehandler):
        clf = AdaBoostClassifier(DecisionTreeClassifier(criterion='entropy', max_depth=5, max_leaf_nodes=15,
                                    min_samples_leaf=43, min_samples_split=46), random_state=self.random_state)
        self.fit_predict_score(clf, self.abcname, self.abc, 'Baseline', envparm)
        self.feature_ranking(clf, self.abcname, 'Baseline', filehandler)

    def get_gridsearch_single_abc(self, param_grid):
        grid_search = GridSearchCV(estimator=AdaBoostClassifier(DecisionTreeClassifier(random_state=self.random_state,
                    criterion='gini', max_depth=5, max_leaf_nodes=20, min_samples_leaf=44,
                    min_samples_split=46), random_state=self.random_state,  n_estimators=500), param_grid=param_grid,
                                   scoring='roc_auc', n_jobs=8, iid=False, cv=3, verbose=0)
        return grid_search

    def get_gridsearch_multi_abc(self, param_grid):
        grid_search = GridSearchCV(estimator=AdaBoostClassifier(DecisionTreeClassifier(random_state=self.random_state,
                        criterion='gini', max_depth=1, max_leaf_nodes=20, min_samples_leaf=44, min_samples_split=46),
                    random_state=self.random_state, n_estimators=700), param_grid=param_grid, scoring='roc_auc',
                                   n_jobs=8, iid=False, cv=3, verbose=0)
        return grid_search

    def gridsearch_single_abc(self):
        # N estimators
        gs_param = 'n_estimators'
        param_grid = {gs_param: range(50, 1550, 50)}
        grid_search = self.get_gridsearch_single_abc(param_grid)
        self.fit_predict_score(grid_search, self.abcname, self.abc, gs_param)

        # Learning rate
        gs_param = 'learning_rate'
        param_grid = {gs_param: [0.1, 0.2, 0.5, 1.0]}
        grid_search = self.get_gridsearch_single_abc(param_grid)
        self.fit_predict_score(grid_search, self.abcname, self.abc, gs_param)

    def gridsearch_multi_abc(self):
        param_grid = {'learning_rate': [0.1, 0.2, 0.5, 1.0]}
        grid_search = self.get_gridsearch_single_abc(param_grid)
        self.fit_predict_score(grid_search, self.abcname, self.abc, 'IsMulti')

    def score_final_abc_churn(self, envparm, filehandler):
        clf = AdaBoostClassifier(DecisionTreeClassifier(random_state=self.random_state, criterion='gini',
                                    max_features=None, max_depth=5, max_leaf_nodes=15, min_samples_leaf=50,
                                    min_samples_split=70), random_state=self.random_state, learning_rate=0.2,
                                    n_estimators=1450)
        self.fit_predict_score(clf, self.abcname, self.abc, 'Final', envparm)
        self.feature_ranking(clf, self.abcname, 'Final', filehandler)

    def score_final_abc_appetency(self, envparm, filehandler):
        clf = AdaBoostClassifier(DecisionTreeClassifier(random_state=self.random_state, criterion='gini',
                                    max_features=None, max_depth=2, max_leaf_nodes=20, min_samples_leaf=44,
                                    min_samples_split=70), random_state=self.random_state, learning_rate=6,
                                    n_estimators=500)
        self.fit_predict_score(clf, self.abcname, self.abc, 'Final', envparm)
        self.feature_ranking(clf, self.abcname, 'Final', filehandler)

    def score_final_abc_upselling(self, envparm, filehandler):
        clf = AdaBoostClassifier(DecisionTreeClassifier(random_state=self.random_state, criterion='gini',
                                    max_features=None, max_depth=4, max_leaf_nodes=20, min_samples_leaf=44,
                                    min_samples_split=70), random_state=self.random_state, learning_rate=4,
                                    n_estimators=500)
        self.fit_predict_score(clf, self.abcname, self.abc, 'Final', envparm)
        self.feature_ranking(clf, self.abcname, 'Final', filehandler)

    def score_baseline_gbc(self, envparm):
        clf = GradientBoostingClassifier(random_state=self.random_state)
        self.fit_predict_score(clf, self.gbcname, self.gbc, 'Baseline', envparm)

    def get_gridsearch_single_gbc(self, param_grid):
        grid_search = GridSearchCV(estimator=GradientBoostingClassifier(random_state=self.random_state),
                                   param_grid=param_grid, scoring='roc_auc', n_jobs=8, iid=False, cv=3, verbose=0)
        return grid_search

    def get_gridsearch_multi_gbc(self, param_grid):
        grid_search = GridSearchCV(estimator=GradientBoostingClassifier(random_state=self.random_state,
                                    n_estimators=100, max_features=None, loss='exponential', subsample=1.0),
                                   param_grid=param_grid, scoring='roc_auc', n_jobs=8, iid=False, cv=3, verbose=0)
        return grid_search

    def gridsearch_single_gbc(self):
        # Grid search params include default values
        # N estimators
        gs_param = 'n_estimators'
        param_grid = {gs_param: [100, 200, 300, 400, 500, 1000, 1200, 1500]}
        grid_search = self.get_gridsearch_single_gbc(param_grid)
        self.fit_predict_score(grid_search, self.gbcname, self.gbc, gs_param)

        # Learning rate
        gs_param = 'learning_rate'
        param_grid = {gs_param: [0.1, 0.05, 0.15, 0.20]}
        grid_search = self.get_gridsearch_single_gbc(param_grid)
        self.fit_predict_score(grid_search, self.gbcname, self.gbc, gs_param)

        # Max features
        gs_param = 'max_features'
        param_grid = {gs_param: [None, 'auto', 'sqrt', 'log2']}
        grid_search = self.get_gridsearch_single_gbc(param_grid)
        self.fit_predict_score(grid_search, self.gbcname, self.gbc, gs_param)

        # Max depth
        gs_param = 'max_depth'
        param_grid = {gs_param: [None, 2, 3, 4, 5]}
        grid_search = self.get_gridsearch_single_gbc(param_grid)
        self.fit_predict_score(grid_search, self.gbcname, self.gbc, gs_param)

        # Min samples split
        gs_param = 'min_samples_split'
        param_grid = {gs_param: range(10, 500, 10)}
        grid_search = self.get_gridsearch_single_gbc(param_grid)
        self.fit_predict_score(grid_search, self.gbcname, self.gbc, gs_param)

        # Loss
        gs_param = 'loss'
        param_grid = {gs_param: ["deviance", "exponential"]}
        grid_search = self.get_gridsearch_single_gbc(param_grid)
        self.fit_predict_score(grid_search, self.gbcname, self.gbc, gs_param)

        # Min samples leaf
        gs_param = 'min_samples_leaf'
        param_grid = {gs_param: range(1, 45, 2)}
        grid_search = self.get_gridsearch_single_gbc(param_grid)
        self.fit_predict_score(grid_search, self.gbcname, self.gbc, gs_param)

        # Max leaf nodes
        gs_param = 'max_leaf_nodes'
        param_grid = {gs_param: [None, 5, 10, 15, 20, 25, 30, 35, 40]}
        grid_search = self.get_gridsearch_single_gbc(param_grid)
        self.fit_predict_score(grid_search, self.gbcname, self.gbc, gs_param)

        # Sub sample
        gs_param = 'subsample'
        param_grid = {gs_param: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
        grid_search = self.get_gridsearch_single_gbc(param_grid)
        self.fit_predict_score(grid_search, self.gbcname, self.gbc, gs_param)

    def gridsearch_multi_gbc(self):
        param_grid = {'max_depth': [2, 3], 'min_samples_leaf': [23, 33, 43], 'max_leaf_nodes': [None, 5, 10],
                      'min_samples_split': [280, 380, 10], 'learning_rate': [0.05, 0.1, 0.2]}
        grid_search = self.get_gridsearch_multi_gbc(param_grid)
        self.fit_predict_score(grid_search, self.gbcname, self.gbc, 'IsMulti')

    def score_final_gbc_churn(self, envparm):
        clf = GradientBoostingClassifier(random_state=self.random_state, max_features=None, loss='exponential',
                                         max_depth=3, max_leaf_nodes=10, min_samples_leaf=17, min_samples_split=8,
                                         n_estimators=100, subsample=0.7)
        self.fit_predict_score(clf, self.gbcname, self.gbc, 'Final', envparm)

    def score_final_gbc_appetency(self, envparm):
        clf = GradientBoostingClassifier(random_state=self.random_state, max_features=None, loss='exponential',
                                         max_depth=2, max_leaf_nodes=5, min_samples_leaf=14, min_samples_split=48,
                                         n_estimators=100, subsample=1.0)
        self.fit_predict_score(clf, self.gbcname, self.gbc, 'Final', envparm)

    def score_final_gbc_upselling(self, envparm):
        clf = GradientBoostingClassifier(random_state=self.random_state, max_features=None, loss='exponential',
                                         max_depth=3, max_leaf_nodes=10, min_samples_leaf=11, min_samples_split=48,
                                         n_estimators=100, subsample=1.0)
        self.fit_predict_score(clf, self.gbcname, self.gbc, 'Final', envparm)

##
    def score_baseline_vtc(self, envparm):
        clf1 = GradientBoostingClassifier(random_state=self.random_state)
        clf2 = RandomForestClassifier(random_state=self.random_state)
        clf = VotingClassifier(estimators=[('gbc', clf1), ('rfc', clf2)], voting='soft')
        self.fit_predict_score(clf, self.vtcname, self.vtc, 'Baseline', envparm)

    def score_final_vtc_churn(self, envparm):
        clf1 = GradientBoostingClassifier(random_state=self.random_state, max_features=None, loss='exponential',
                                          max_depth=3, max_leaf_nodes=10, min_samples_leaf=17, min_samples_split=8,
                                          n_estimators=100, subsample=0.7)
        clf2 = RandomForestClassifier(random_state=self.random_state, criterion='gini', max_features=None, max_depth=5,
                                     max_leaf_nodes=25, min_samples_leaf=39, min_samples_split=46, n_estimators=500)
        clf3 = BaggingClassifier(DecisionTreeClassifier(random_state=self.random_state, criterion='gini',
                                    max_features=None, max_depth=5, max_leaf_nodes=25, min_samples_leaf=43,
                                    min_samples_split=70), n_estimators=500, max_features=1.0, max_samples=0.6)

        clf = VotingClassifier(estimators=[('gbc', clf1), ('rfc', clf2), ('bgc', clf3)], voting='soft',
                               weights=[2.1, 1.2, 1])
        self.fit_predict_score(clf, self.vtcname, self.vtc, 'Final', envparm)

    def score_final_vtc_appetency(self, envparm):
        clf1 = GradientBoostingClassifier(random_state=self.random_state, max_features=None, loss='exponential',
                                          max_depth=2, max_leaf_nodes=5, min_samples_leaf=14, min_samples_split=48,
                                          n_estimators=100, subsample=1.0)

        clf2 = RandomForestClassifier(random_state=self.random_state, criterion='entropy', max_features=None,
                                     max_depth=2, max_leaf_nodes=25, min_samples_leaf=39, min_samples_split=46,
                                     n_estimators=700)
        clf3 = BaggingClassifier(DecisionTreeClassifier(random_state=self.random_state, criterion='entropy',
                                    max_features=None, max_depth=2, max_leaf_nodes=10, min_samples_leaf=43,
                                    min_samples_split=46), n_estimators=700, max_features=1.0, max_samples=0.6)
        clf = VotingClassifier(estimators=[('gbc', clf1), ('rfc', clf2), ('bgc', clf3)], voting='soft',
                               weights=[2.1, 1.2, 1])
        self.fit_predict_score(clf, self.vtcname, self.vtc, 'Final', envparm)

    def score_final_vtc_upselling(self, envparm):
        clf1 = GradientBoostingClassifier(random_state=self.random_state, max_features=None, loss='exponential',
                                          max_depth=3, max_leaf_nodes=10, min_samples_leaf=11, min_samples_split=48,
                                          n_estimators=100, subsample=1.0)
        clf2 = RandomForestClassifier(random_state=self.random_state, criterion='entropy', max_features=None,
                                     max_depth=5, max_leaf_nodes=35, min_samples_leaf=31, min_samples_split=46,
                                     n_estimators=500)
        clf3 = BaggingClassifier(DecisionTreeClassifier(random_state=self.random_state, criterion='entropy',
                                    max_features=None, max_depth=5, max_leaf_nodes=20, min_samples_leaf=43,
                                    min_samples_split=46), n_estimators=600, max_features=1.0, max_samples=0.6)
        clf = VotingClassifier(estimators=[('gbc', clf1), ('rfc', clf2), ('bgc', clf3)], voting='soft',
                               weights=[2.1, 1.2, 1])
        self.fit_predict_score(clf, self.vtcname, self.vtc, 'Final', envparm)

    def classifier_tune(self, envparm):
        # Random Forest Classifier
        if envparm['GridSearchSingleRFC']:
            logging.info("Start grid search tune for Random Forest, target is {}".format(self.target_label))
            self.gridsearch_single_rfc()

        if envparm['GridSearchMultiRFC']:
            logging.info(
                "Start grid search combined param tune for Random Forest, target is {}".format(self.target_label))
            self.gridsearch_multi_rfc()

        # Decision Tree Classifier
        if envparm['GridSearchSingleDTC']:
            logging.info("Start grid search tune for Decision Tree, target is {}".format(self.target_label))
            self.gridsearch_single_dtc()

        if envparm['GridSearchMultiDTC']:
            logging.info(
                "Start grid search combined param tune for Decision Tree, target is {}".format(self.target_label))
            self.gridsearch_multi_dtc()

        # AdaBoost Classifier
        if envparm['GridSearchSingleABC']:
            logging.info("Start grid search tune for AdaBoost, target is {}".format(self.target_label))
            self.gridsearch_single_abc()

        if envparm['GridSearchMultiABC']:
            logging.info(
                "Start grid search combined param tune for AdaBoost, target is {}".format(self.target_label))
            self.gridsearch_multi_abc()

        # Gradient Boosting Classifier
        if envparm['GridSearchSingleGBC']:
            logging.info("Start grid search tune for Gradient Boosting, target is {}".format(self.target_label))
            self.gridsearch_single_gbc()

        if envparm['GridSearchMultiGBC']:
            logging.info(
                "Start grid search combined param tune for Gradient Boosting, target is {}".format(self.target_label))
            self.gridsearch_multi_gbc()

        # Bagging Classifier
        if envparm['GridSearchSingleBGC']:
            logging.info("Start grid search tune for Bagging, target is {}".format(self.target_label))
            self.gridsearch_single_bgc()

        if envparm['GridSearchMultiBGC']:
            logging.info(
                "Start grid search combined param tune for Bagging, target is {}".format(self.target_label))
            self.gridsearch_multi_bgc()

    def train_test_split(self, dataset, target, target_label):
        self.target_label = target_label
        logging.info("Splitting dataset for target - {}".format(self.target_label))
        self.dataset_train, self.dataset_test, self.target_train, self.target_test = \
            train_test_split(dataset, target.values.ravel(), test_size=self.test_size, random_state=0)

    def standard_scaler(self):
        logging.info("Scaling dataset")
        sc_dataset_train = StandardScaler()
        self.dataset_train = sc_dataset_train.fit_transform(self.dataset_train)
        self.dataset_test = sc_dataset_train.transform(self.dataset_test)

    def output_scores(self, filehandler):
        filehandler.output_scores(self.scores)
        visualizer.barplot_scores(self.scores)

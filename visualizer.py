# Author:   Jon-Paul Boyd
# Date:     16/01/2018
# IMAT5234  Applied Computational Intelligence - Mini Project
# Customer Relationship Management - Predict churn, appetency and upselling on Orange dataset
# Visualizer
import logging
import pandas as pd
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
import seaborn as sns
import pydotplus
import missingno as msno
import collections
import os
os.environ["PATH"] += os.pathsep + 'D:/Program Files (x86)/graphviz-2.38/release/bin/'


def graph_decision_tree(clf, dataset, name):
    return
    path_depth = 'graphs/Graph - ' + name + ' depth 3.png'
    path_full = 'graphs/Graph - ' + name + ' full.png'

    dataset_feature_names = dataset.columns.get_values()
    dataset_feature_names.tolist()

    logging.info("Creating graph - {}".format(path_depth))
    dot_data = tree.export_graphviz(clf, out_file=None, filled=True, rounded=True,
                                    feature_names=dataset_feature_names, max_depth=3)
    graph = pydotplus.graph_from_dot_data(dot_data)
    colors = ('cornsilk', 'deepskyblue')
    edges = collections.defaultdict(list)
    for edge in graph.get_edge_list():
        edges[edge.get_source()].append(int(edge.get_destination()))

    for edge in edges:
        edges[edge].sort()
        for i in range(2):
            dest = graph.get_node(str(edges[edge][i]))[0]
            dest.set_fillcolor(colors[i])
    graph.write_png(path_depth)

    logging.info("Creating graph - {}".format(path_full))
    dot_data = tree.export_graphviz(clf, out_file=None, filled=True, rounded=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    colors = ('cornsilk', 'deepskyblue')
    edges = collections.defaultdict(list)
    for edge in graph.get_edge_list():
        edges[edge.get_source()].append(int(edge.get_destination()))

    for edge in edges:
        edges[edge].sort()
        for i in range(2):
            dest = graph.get_node(str(edges[edge][i]))[0]
            dest.set_fillcolor(colors[i])

    graph.write_png(path_full)


def matrix_missing(sample_df, title):
    missing_data_df = sample_df.columns[sample_df.isnull().any()].tolist()
    msno.matrix(sample_df[missing_data_df], sparkline=False, fontsize=12)
    plt.title(title, fontsize=20, y=1.08)
    fig = plt.gcf()
    fig.savefig('graphs/' + title + '.png')
    plt.show()


def bar_missing(sample_df, title):
    missing_data_df = sample_df.columns[sample_df.isnull().any()].tolist()
    msno.bar(sample_df[missing_data_df], color="black", log=False, figsize=(30, 18))
    plt.title(title, fontsize=24, y=1.05)
    fig = plt.gcf()
    fig.savefig('graphs/' + title + '.png')
    plt.show()


def heat_missing(sample_df, title):
    missing_data_df = sample_df.columns[sample_df.isnull().any()].tolist()
    msno.heatmap(sample_df[missing_data_df], figsize=(20, 20))
    plt.title(title, fontsize=24)
    fig = plt.gcf()
    fig.savefig('graphs/' + title + '.png')
    plt.show()


def confusion_matrix(cm, target_label, clf_name, gs_param):
    title = 'CM - ' + target_label + ' - ' + clf_name + ' - ' + gs_param
    cm_array = [[cm[1][1], cm[1][0]], [cm[0][1], cm[0][0]]]
    df = pd.DataFrame(cm_array, ['Positive', 'Negative'], ['Positive', 'Negative'])
    sns.heatmap(df, annot=True, annot_kws={'size': 16}, fmt='g', cbar=False, center=cm[1][0], cmap="Greys")
    plt.title(title)
    plt.xlabel('Predicted Class')
    plt.ylabel('Actual Class')
    fig = plt.gcf()
    fig.savefig('graphs/Graph - ' + title + '.png')
    plt.show()


def barplot_scores(scores):
    sns.set(style="white")
    m = np.asarray(scores)
    df = pd.DataFrame()
    df["Index"] = m[:, 0]
    df["Classifier Name"] = m[:, 1]
    df["Classifier Short"] = m[:, 2]
    df["Type"] = m[:, 4]
    df["Target"] = m[:, 6]
    df["Target Short"] = df.Target.str[:2]
    df["Score"] = np.float16(m[:, 7])
    df["Classifier"] = df['Classifier Short'] + ' ' + df['Type'] + ' ' + df['Target Short']
    df.sort_values(['Classifier'], inplace=True)

    fig, ax = plt.subplots(figsize=(24, 9))
    sns.barplot(x='Classifier', y='Score', data=df, palette="Paired")
    plt.title('Classifier AUC Scores', fontsize=18)
    plt.xlabel('Classifier', fontsize=18)
    plt.ylabel('AUC Score', fontsize=18)

    for p in ax.patches:
        ax.annotate("%.4f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=16, color='black', xytext=(0, -15),
                    textcoords='offset points')

    fig.savefig('graphs/Graph - Scores.png')
    plt.show()


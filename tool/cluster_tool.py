# -*- coding: utf-8 -*-

from sklearn import metrics
import numpy as np


def cluster_evaluation(clustering_input, real_y, pre_y, sample_size=300, seed=2022):
    ari_score = metrics.adjusted_rand_score(real_y, pre_y)
    ami_score = metrics.adjusted_mutual_info_score(real_y, pre_y)
    nmi_score = metrics.normalized_mutual_info_score(real_y, pre_y)
    fmi_score = metrics.fowlkes_mallows_score(real_y, pre_y)
    try:
        aws_score = metrics.silhouette_score(clustering_input, pre_y, metric="euclidean", sample_size=sample_size,
                                             random_state=seed)
    except:
        aws_score = None
    return ari_score, ami_score, nmi_score, fmi_score, aws_score


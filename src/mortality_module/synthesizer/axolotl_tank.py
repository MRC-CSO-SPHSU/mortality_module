from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

import pandas as pd
import numpy as np
from collections import defaultdict

from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm


def get_optimal_features(_X, _X_train, _X_test, _y_train, _y_test, verbose=True,
                         _min_cutoff = 0.5, _max_cutoff = 1.5):

    corr = spearmanr(_X).correlation

    # Ensure the correlation matrix is symmetric
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1)

    # We convert the correlation matrix to a distance matrix before performing
    # hierarchical clustering using Ward's linkage.
    distance_matrix = 1 - np.abs(corr)
    dist_linkage = hierarchy.ward(squareform(distance_matrix))

    scores = []
    cutoffs = []

    _range = np.arange(_min_cutoff, _max_cutoff, 0.05)
    _g = tqdm(_range) if verbose else _range

    for _cutoff in _g:
        cluster_ids = hierarchy.fcluster(dist_linkage,
                                         _cutoff,
                                         criterion="distance")
        cluster_id_to_feature_ids = defaultdict(list)

        for idx, cluster_id in enumerate(cluster_ids):
            cluster_id_to_feature_ids[cluster_id].append(idx)

        selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
        selected_features_names = _X.columns[selected_features]

        clf_sel = RandomForestClassifier(n_estimators=1024,
                             criterion='entropy',
                             max_depth=None,
                             max_features=None,
                             random_state=1,
                             n_jobs=4)

        clf_sel.fit(_X_train[selected_features_names], _y_train)

        cutoffs.append(_cutoff)
        scores.append(clf_sel.score(_X_test[selected_features_names], _y_test))

    _data = pd.DataFrame({"cutoff": cutoffs,
                          "score": scores}).sort_values(by="cutoff")

    _s = _data["score"].max()
    _max = _data[_data["score"].eq(_s)]["cutoff"].max()

    cluster_ids = hierarchy.fcluster(dist_linkage, _max, criterion="distance")
    cluster_id_to_feature_ids = defaultdict(list)

    for idx, cluster_id in enumerate(cluster_ids):
        cluster_id_to_feature_ids[cluster_id].append(idx)

    selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
    return _X.columns[selected_features].to_list()

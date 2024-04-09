"""SAP score from `Variational Inference of Disentangled Latent Concepts from Unlabeled Observations <https://arxiv.org/abs/1711.00848>`.

Part of Code is adapted from `disentanglement lib<https://github.com/google-research/disentanglement_lib/blob/86a644d4ed35c771560dc3360756363d35477357/disentanglement_lib/evaluation/metrics/sap_score.py>`.
"""

# coding=utf-8
# Copyright 2018 Ubisoft La Forge Authors.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np

from sklearn import tree
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import minmax_scale



def sap(factors, codes, discrete_factors = False, continuous_factors=True, nb_bins=10, regression=True):
    ''' SAP metric from A. Kumar, P. Sattigeri, and A. Balakrishnan,
        “Variational inference of disentangled latent concepts from unlabeledobservations,”
        in ICLR, 2018.

    :param factors:                         dataset of factors
                                            each column is a factor and each line is a data point
    :param codes:                           latent codes associated to the dataset of factors
                                            each column is a latent code and each line is a data point
    :param continuous_factors:              True:   factors are described as continuous variables
                                            False:  factors are described as discrete variables
    :param nb_bins:                         number of bins to use for discretization
    :param regression:                      True:   compute score using regression algorithms
                                            False:  compute score using classification algorithms
    '''
    # count the number of factors and latent codes
    nb_factors = factors.shape[1]
    nb_codes = codes.shape[1]

    # perform regression
    if regression:
        assert (continuous_factors), f'Cannot perform SAP regression with discrete factors.'
        return _sap_regression(factors, codes, nb_factors, nb_codes)

        # perform classification
    else:
        # quantize factors if they are continuous
        if continuous_factors:
            factors = minmax_scale(factors)  # normalize in [0, 1] all columns
            factors = get_bin_index(factors, nb_bins)  # quantize values and get indexes

        # normalize in [0, 1] all columns
        codes = minmax_scale(codes)

        # compute score using classification algorithms
        return _sap_classification(factors, codes, nb_factors, nb_codes)


def _sap_regression(factors, codes, nb_factors, nb_codes):
    ''' Compute SAP score using regression algorithms

    :param factors:         factors dataset
    :param codes:           latent codes dataset
    :param nb_factors:      number of factors in the factors dataset
    :param nb_codes:        number of codes in the latent codes dataset
    '''
    # compute R2 score matrix
    s_matrix = np.zeros((nb_factors, nb_codes))
    for f in range(nb_factors):
        for c in range(nb_codes):
            # train a linear regressor
            regr = LinearRegression()

            # train the model using the training sets
            regr.fit(codes[:, c].reshape(-1, 1), factors[:, f].reshape(-1, 1))

            # make predictions using the testing set
            y_pred = regr.predict(codes[:, c].reshape(-1, 1))

            # compute R2 score
            r2 = r2_score(factors[:, f], y_pred)
            s_matrix[f, c] = max(0, r2)

            # compute the mean gap for all factors
    sum_gap = 0
    for f in range(nb_factors):
        # get diff between highest and second highest term and add it to total gap
        s_f = np.sort(s_matrix[f, :])
        sum_gap += s_f[-1] - s_f[-2]

    # compute the mean gap
    sap_score = sum_gap / nb_factors

    return sap_score


def _sap_classification(factors, codes, nb_factors, nb_codes):
    ''' Compute SAP score using classification algorithms

    :param factors:         factors dataset
    :param codes:           latent codes dataset
    :param nb_factors:      number of factors in the factors dataset
    :param nb_codes:        number of codes in the latent codes dataset
    '''
    # compute accuracy matrix
    s_matrix = np.zeros((nb_factors, nb_codes))
    for f in range(nb_factors):
        for c in range(nb_codes):
            # find the optimal number of splits
            best_score, best_sp = 0, 0
            for sp in range(1, 10):
                # perform cross validation on the tree classifiers
                clf = tree.DecisionTreeClassifier(max_depth=sp)
                scores = cross_val_score(clf, codes[:, c].reshape(-1, 1), factors[:, f].reshape(-1, 1), cv=10)
                scores = scores.mean()

                if scores > best_score:
                    best_score = scores
                    best_sp = sp

            # train the model using the best performing parameter
            clf = tree.DecisionTreeClassifier(max_depth=best_sp)
            clf.fit(codes[:, c].reshape(-1, 1), factors[:, f].reshape(-1, 1))

            # make predictions using the testing set
            y_pred = clf.predict(codes[:, c].reshape(-1, 1))

            # compute accuracy
            s_matrix[f, c] = accuracy_score(y_pred, factors[:, f])

    # compute the mean gap for all factors
    sum_gap = 0
    for f in range(nb_factors):
        # get diff between highest and second highest term and add it to total gap
        s_f = np.sort(s_matrix[f, :])
        sum_gap += s_f[-1] - s_f[-2]

    # compute the mean gap
    sap_score = sum_gap / nb_factors

    return sap_score


def get_bin_index(x, nb_bins):
    ''' Discretize input variable

    :param x:           input variable
    :param nb_bins:     number of bins to use for discretization
    '''
    # get bins limits
    bins = np.linspace(0, 1, nb_bins + 1)

    # discretize input variable
    return np.digitize(x, bins[:-1], right=False).astype(int)


# def label_transformer(train_data):
#     classes = np.unique(train_data)
#
#     def transform(new_data):
#         nonlocal classes
#         new_classes = np.array(list(set(np.unique(new_data)) - set(classes)))
#         classes = np.hstack([classes, new_classes])
#         indices = (new_data == classes.reshape(-1, 1)).argmax(axis=0)
#         return indices
#
#     return transform
#
#
# def get_score_matrix(
#     codes: np.ndarray,
#     factors: np.ndarray,
#     discrete_factors: Union[List[bool], bool] = True,
#     test_size: float = 0.2,
#     random_state: Union[float, None] = None,
#     **kwargs,
# ) -> np.ndarray:
#
#     """Compute the relationship matrix for SAP.
#
#     Part of Code is adapted from `disentanglement lib<https://github.com/google-research/disentanglement_lib/blob/86a644d4ed35c771560dc3360756363d35477357/disentanglement_lib/evaluation/metrics/sap_score.py>`.
#
#     Args:
#         codes (np.ndarray): [Shape (batch_size, n_codes)] The latent codes.
#         factors (np.ndarray): [Shape (batch_size, n_factors)] The real generative factors.
#         discrete_factors (Union[List[bool], bool]): implies if each factor is discrete. Default: True.
#         test_size (float, optional): The rate of test samples, between 0 and 1. Default: 0.3.
#         random_state (Union[float, None], optional): Seed of random generator for sampling test data. Default: None.
#
#     Returns:
#         score matrix (np.ndarray): score matrix where ij entry represents the relationship between code i and factor j
#     """
#
#     n_factors, n_codes = factors.shape[1], codes.shape[1]
#     if type(discrete_factors) is not list:
#         discrete_factors = [discrete_factors] * n_factors
#     X_train, X_test, y_train, y_test = train_test_split(
#         codes, factors, test_size=test_size, random_state=random_state
#     )
#     score_matrix = np.zeros([n_codes, n_factors])
#     for j, discrete_factor in enumerate(discrete_factors):
#         y_j = y_train[:, j]
#         if discrete_factor:
#             y_j_test = y_test[:, j]
#             transform = label_transformer(y_j)
#             y_j_encoded = transform(y_j)[:, np.newaxis]
#             y_j_test_encoded = transform(y_j_test)[:, np.newaxis]
#
#         for i in range(n_codes):
#             x_i = X_train[:, i]
#             if discrete_factor:
#                 x_i_test = X_test[:, i]
#                 classifier = XGBClassifier(tree_method="hist", device = "cudanv")
#                 classifier.fit(
#                     x_i[:, np.newaxis].astype(np.float32), y_j_encoded
#                 )
#                 pred = classifier.predict(
#                     x_i_test[:, np.newaxis].astype(np.float32)
#                 )
#                 score_matrix[i, j] = np.mean(pred == y_j_test_encoded)
#             else:
#                 cov_x_i_y_j = np.cov(x_i, y_j, ddof=1)
#                 var_x_i_y_j = cov_x_i_y_j[0, 1] ** 2
#                 var_x = cov_x_i_y_j[0, 0]
#                 var_y = cov_x_i_y_j[1, 1]
#                 if var_x > 1e-10:
#                     score_matrix[i, j] = var_x_i_y_j * 1.0 / (var_x * var_y)
#                 else:
#                     score_matrix[i, j] = 0.0
#     return score_matrix
#
#
# def sap(
#     factors: np.ndarray,
#     codes: np.ndarray,
#     discrete_factors: Union[List[bool], bool] = True,
#     test_size: float = 0.2,
#     random_state: Union[float, None] = None,
#     **kwargs,
# ) -> float:
#     """Compute SAP score.
#
#     Args:
#         codes (np.ndarray): [Shape (batch_size, n_codes)] The latent codes.
#         factors (np.ndarray): [Shape (batch_size, n_factors)] The real generative factors.
#         discrete_factors (Union[List[bool], bool]): implies if each factor is discrete. Default: True.
#         test_size (float, optional): The rate of test samples, between 0 and 1. Default: 0.3.
#         random_state (Union[float, None], optional): Seed of random generator for sampling test data. Default: None.
#
#     Returns:
#         score (float): SAP score
#     """
#     matrix = get_score_matrix(
#         codes,
#         factors,
#         discrete_factors=discrete_factors,
#         test_size=test_size,
#         random_state=random_state,
#     )
#     sorted = np.sort(matrix, axis=0)
#     score = np.mean(sorted[-1, :] - sorted[-2, :])
#     return score

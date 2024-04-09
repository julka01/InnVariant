"""Disentanglement, Completeness and Informativeness (DCI) from `A Framework for the Quantitative Evaluation of Disentangled Representations <https://openreview.net/forum?id=By-7dz-AZ>.

Part of code adapted from: https://github.com/google-research/disentanglement_lib
"""
from typing import Union, List, Tuple
import scipy
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import Lasso
from xgboost import XGBClassifier, XGBRegressor


def label_transformer(train_data):
    classes = np.unique(train_data)

    def transform(new_data):
        nonlocal classes
        new_classes = np.array(list(set(np.unique(new_data)) - set(classes)))
        classes = np.hstack([classes, new_classes])
        indices = (new_data == classes.reshape(-1, 1)).argmax(axis=0)
        return indices

    return transform


def dci_collect_relationship(
    factors: np.ndarray,
    codes: np.ndarray,
    test_size: float = 0.3,
    random_state: Union[float, None] = None,
    discrete_factors: Union[List[bool], bool] = True,
    **kwargs,
) -> Tuple[np.ndarray, List[float], List[float]]:
    """Compute the relationship for D, C (feature importance matrix), and I (prediction accuracy).

    Args:
        factors: The real generative factors (batch_size, factor_dims).
        codes: The latent codes (batch_size, code_dims).
        test_size (float, optional): The rate of test samples, between 0 and 1. Default: 0.3.
        random_state (Union[float, None], optional): Seed of random generator for sampling test data. Default: None.
        discrete_factors (Union[List[bool], bool]): implies if each factor is discrete. Default: True.

    Returns:
        important_matrix (np.ndarray): [Shape (n_codes, n_factors)] The importance of each code to each factor, where the ij entry represents the importance of code i to the factor j.
        train_accuracies (List(float)): [Shape (n_factors, )] The accuracy of each factor predition using codes in training.
        test_accuracies (List(float)): [Shape (n_factors, )] The accuracy of each factor predition using codes in testing.
    """
    n_factors = factors.shape[1]
    if type(discrete_factors) is bool:
        discrete_factors = [discrete_factors] * n_factors
    n_codes = codes.shape[1]
    x_train, x_test, y_train, y_test = train_test_split(
        codes, factors, test_size=test_size, random_state=random_state
    )
    importances, train_accuracies, test_accuracies = [], [], []
    for i, discrete_factor in enumerate(discrete_factors):
        if discrete_factor:
            transform = label_transformer(y_train[:, i])
            y_train_encoded = transform(y_train[:, i])
            y_test_encoded = transform(y_test[:, i])
            model = XGBClassifier(tree_method="hist", device = 'cuda')
            model.fit(x_train, y_train_encoded)
            importances.append(np.abs(model.feature_importances_))
            train_accuracies.append(model.score(x_train, y_train_encoded))
            test_accuracies.append(model.score(x_test, y_test_encoded))
        else:
            model = XGBRegressor(tree_method="hist", device = 'cuda')
            model.fit(x_train, y_train[:, i])
            importances.append(np.abs(model.feature_importances_))
            train_accuracies.append(model.score(x_train, y_train[:, i]))
            test_accuracies.append(model.score(x_test, y_test[:, i]))

    importance_matrix = np.stack(importances, axis=1)

    return importance_matrix, train_accuracies, test_accuracies


def dci_xgb(
    factors: np.ndarray,
    codes: np.ndarray,
    discrete_factors: Union[List[bool], bool] = True,
    test_size: float = 0.3,
    random_state: Union[float, None] = None,
    **kwargs,
) -> float:
    """Compute DCI scores.

    Args:
        factors (np.ndarray): [Shape (batch_size, n_factors)] The real generative factors.
        codes (np.ndarray): [Shape (batch_size, n_codes)] The latent codes.
        discrete_factors (Union[List[bool], bool]): implies if each factor is discrete. Default: True.
        test_size (float, optional): The rate of test samples, between 0 and 1. Default: 0.3.
        random_state (Union[float, None], optional): Seed of random generator for sampling test data. Default: None.

    Returns:
        scores (dict): Dictionary where
            - "d" represents average disentanglement score,
            - "c" represents average completeness score,
            - "i" represents informativeness score (in test stage).
            - "i_train" represents informativeness score (in test stage).
    """
    (
        importance_matrix,
        train_accuracies,
        test_accuracies,
    ) = dci_collect_relationship(
        factors,
        codes,
        test_size=test_size,
        random_state=random_state,
        discrete_factors=discrete_factors,
    )

    # calculte the d c i score from relationship.
    train_accuracy = np.mean(train_accuracies)
    test_accuracy = np.mean(test_accuracies)

    d_score = disentanglement(importance_matrix)
    c_score = completeness(importance_matrix)
    i_score = test_accuracy
    #return d_score, c_score, i_score
    return dict(d=d_score, c=c_score, i=i_score, i_train=train_accuracy)


###
# the following code is taken or adapted from `disentanglement lib <https://github.com/google-research/disentanglement_lib>`
###
def dci_from_disentanglement_lib(
    factors, codes, test_size=0.3, random_state=None, **kwargs
):
    """
    Args:
        factors (np.ndarray): [Shape (batch_size, n_factors)] The real generative factors.
        codes (np.ndarray): [Shape (batch_size, n_codes)] The latent codes.
    Returns:
        scores: Dictionary with average disentanglement score, completeness and
        informativeness (train and test).
    """
    x_train, x_test, y_train, y_test = train_test_split(
        codes, factors, test_size=test_size, random_state=random_state
    )
    return _compute_dci(x_train.T, y_train.T, x_test.T, y_test.T)


def _compute_dci(mus_train, ys_train, mus_test, ys_test):
    """Computes score based on both training and testing codes and factors."""
    scores = {}

    importance_matrix, train_err, test_err = compute_importance_gbt(
        mus_train, ys_train, mus_test, ys_test
    )
    assert importance_matrix.shape[0] == mus_train.shape[0]
    assert importance_matrix.shape[1] == ys_train.shape[0]
    d = disentanglement(importance_matrix)
    c = completeness(importance_matrix)
    scores["informativeness_train"] = train_err
    scores["informativeness_test"] = test_err
    scores["disentanglement"] = d
    scores["completeness"] = c
    scores["i"] = test_err
    scores["d"] = d
    scores["c"] = c
    return scores


def compute_importance_gbt(x_train, y_train, x_test, y_test):
    """Compute importance based on gradient boosted trees."""
    num_factors = y_train.shape[0]
    num_codes = x_train.shape[0]
    importance_matrix = np.zeros(
        shape=[num_codes, num_factors], dtype=np.float64
    )
    train_loss = []
    test_loss = []
    for i in range(num_factors):
        model = GradientBoostingClassifier()
        model.fit(x_train.T, y_train[i, :])
        importance_matrix[:, i] = np.abs(model.feature_importances_)
        train_loss.append(np.mean(model.predict(x_train.T) == y_train[i, :]))
        test_loss.append(np.mean(model.predict(x_test.T) == y_test[i, :]))
    return importance_matrix, np.mean(train_loss), np.mean(test_loss)


def disentanglement_per_code(importance_matrix):
    """Compute disentanglement score of each code."""
    return 1.0 - scipy.stats.entropy(
        importance_matrix.T + 1e-11, base=importance_matrix.shape[1]
    )


def disentanglement(importance_matrix):
    """Compute the disentanglement score of the representation."""
    per_code = disentanglement_per_code(importance_matrix)
    if importance_matrix.sum() == 0.0:
        importance_matrix = np.ones_like(importance_matrix)
    code_importance = importance_matrix.sum(axis=1) / importance_matrix.sum()

    return np.sum(per_code * code_importance)


def completeness_per_factor(importance_matrix):
    """Compute completeness of each factor."""
    return 1.0 - scipy.stats.entropy(
        importance_matrix + 1e-11, base=importance_matrix.shape[0]
    )


def completeness(importance_matrix):
    """ "Compute completeness of the representation."""
    per_factor = completeness_per_factor(importance_matrix)
    if importance_matrix.sum() == 0.0:
        importance_matrix = np.ones_like(importance_matrix)
    factor_importance = importance_matrix.sum(axis=0) / importance_matrix.sum()
    return np.sum(per_factor * factor_importance)


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
import math

import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import minmax_scale


def dci(factors, codes, discrete_factors = False, continuous_factors=True, model='lasso'):
    ''' DCI metrics from C. Eastwood and C. K. I. Williams,
        “A framework for the quantitative evaluation of disentangled representations,”
        in ICLR, 2018.

    :param factors:                         dataset of factors
                                            each column is a factor and each line is a data point
    :param codes:                           latent codes associated to the dataset of factors
                                            each column is a latent code and each line is a data point
    :param continuous_factors:              True:   factors are described as continuous variables
                                            False:  factors are described as discrete variables
    :param model:                           model to use for score computation
                                            either lasso or random_forest
    '''
    # TODO: Support for discrete data
    assert (continuous_factors), f'Only continuous factors are supported'

    # count the number of factors and latent codes
    nb_factors = factors.shape[1]
    nb_codes = codes.shape[1]

    # normalize in [0, 1] all columns
    factors = minmax_scale(factors)
    codes = minmax_scale(codes)

    # compute entropy matrix and informativeness per factor
    e_matrix = np.zeros((nb_factors, nb_codes))
    informativeness = np.zeros((nb_factors,))
    for f in range(nb_factors):
        if model == 'lasso':
            informativeness[f], weights = _fit_lasso(factors[:, f].reshape(-1, 1), codes)
            e_matrix[f, :] = weights
        elif model == 'random_forest':
            informativeness[f], weights = _fit_random_forest(factors[:, f].reshape(-1, 1), codes)
            e_matrix[f, :] = weights
        else:
            raise ValueError("Regressor must be lasso or random_forest")

    # compute disentanglement per code
    rho = np.zeros((nb_codes,))
    disentanglement = np.zeros((nb_codes,))
    for c in range(nb_codes):
        # get importance weight for code c
        rho[c] = np.sum(e_matrix[:, c])
        if rho[c] == 0:
            disentanglement[c] = 0
            break

        # transform weights in probabilities
        prob = e_matrix[:, c] / rho[c]

        # compute entropy for code c
        H = 0
        for p in prob:
            if p:
                H -= p * math.log(p, len(prob))

        # get disentanglement score
        disentanglement[c] = 1 - H

    # compute final disentanglement
    if np.sum(rho):
        rho = rho / np.sum(rho)
    else:
        rho = rho * 0

    # compute completeness
    completeness = np.zeros((nb_factors,))
    for f in range(nb_factors):
        if np.sum(e_matrix[f, :]) != 0:
            prob = e_matrix[f, :] / np.sum(e_matrix[f, :])
        else:
            prob = np.ones((len(e_matrix[f, :]), 1)) / len(e_matrix[f, :])

            # compute entropy for code c
        H = 0
        for p in prob:
            if p:
                H -= p * math.log(p, len(prob))

        # get disentanglement score
        completeness[f] = 1 - H

    # average all results
    disentanglement = np.dot(disentanglement, rho)
    completeness = np.mean(completeness)
    informativeness = np.mean(informativeness)

    return disentanglement, completeness, informativeness


def _fit_lasso(factors, codes):
    ''' Fit a Lasso regressor on the data

    :param factors:         factors dataset
    :param codes:           latent codes dataset
    '''
    # alpha values to try
    alphas = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.4, 0.8, 1]

    # make sure factors are N by 1
    factors.reshape(-1, 1)

    # find the optimal alpha regularization parameter
    best_a = 0
    best_mse = 10e10
    for a in alphas:
        # perform cross validation on the tree classifiers
        clf = Lasso(alpha=a, max_iter=5000)
        mse = cross_val_score(clf, codes, factors, cv=10, scoring='neg_mean_squared_error')
        mse = -mse.mean()

        if mse < best_mse:
            best_mse = mse
            best_a = a

    # train the model using the best performing parameter
    clf = Lasso(alpha=best_a)
    clf.fit(codes, factors)

    # make predictions using the testing set
    y_pred = clf.predict(codes)

    # compute informativeness from prediction error (max value for mse/2 = 1/12)
    mse = mean_squared_error(y_pred, factors)
    informativeness = max(1 - 12 * mse, 0)

    # get the weight from the regressor
    predictor_weights = np.ravel(np.abs(clf.coef_))

    return informativeness, predictor_weights


def _fit_random_forest(factors, codes):
    ''' Fit a Random Forest regressor on the data

    :param factors:         factors dataset
    :param codes:           latent codes dataset
    '''
    # alpha values to try
    max_depth = [8, 16, 32, 64, 128]
    max_features = [0.2, 0.4, 0.8, 1.0]

    # make sure factors are N by 0
    factors = np.ravel(factors)

    # find the optimal alpha regularization parameter
    best_mse = 10e10
    best_mf = 0
    best_md = 0
    for md in max_depth:
        for mf in max_features:
            # perform cross validation on the tree classifiers
            clf = RandomForestRegressor(n_estimators=10, max_depth=md, max_features=mf)
            mse = cross_val_score(clf, codes, factors, cv=10, scoring='neg_mean_squared_error')
            mse = -mse.mean()

            if mse < best_mse:
                best_mse = mse
                best_mf = mf
                best_md = md

    # train the model using the best performing parameter
    clf = RandomForestRegressor(n_estimators=10, max_depth=best_md, max_features=best_mf)
    clf.fit(codes, factors)

    # make predictions using the testing set
    y_pred = clf.predict(codes)

    # compute informativeness from prediction error (max value for mse/2 = 1/12)
    mse = mean_squared_error(y_pred, factors)
    informativeness = max(1 - 12 * mse, 0)

    # get the weight from the regressor
    predictor_weights = clf.feature_importances_

    return informativeness, predictor_weights

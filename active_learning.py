import torch
import numpy as np
from modAL.models import ActiveLearner

from acquisition_functions import uniform, max_entropy, bald, var_ratios, mean_std,batch_bald


def active_learning_procedure(
    query_strategy,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    X_init: np.ndarray,
    y_init: np.ndarray,
    estimator,
    T: int = 100,
    n_query: int = 10,
    training: bool = True,
):
    """Active Learning Procedure

    Attributes:
        query_strategy: Choose between Uniform(baseline), max_entropy, bald,
        X_val, y_val: Validation dataset,
        X_test, y_test: Test dataset,
        X_pool, y_pool: Query pool set,
        X_init, y_init: Initial training set data points,
        estimator: Neural Network architecture, e.g. CNN,
        T: Number of MC dropout iterations (repeat acqusition process T times),
        n_query: Number of points to query from X_pool,
        training: If False, run test without MC Dropout (default: True)
    """
    learner = ActiveLearner(
        estimator=estimator,
        X_training=X_init,
        y_training=y_init,
        query_strategy=query_strategy,
    )
    perf_hist = [learner.score(X_test, y_test)]
    for index in range(T):
        query_idx, query_instance = learner.query(
            X_pool, n_query=n_query, T=T, training=training
        )
        #oraculo.
        learner.teach(X_pool[query_idx], y_pool[query_idx])
        X_pool = np.delete(X_pool, query_idx, axis=0)
        y_pool = np.delete(y_pool, query_idx, axis=0)
        model_accuracy_val = learner.score(X_val, y_val)
        if (index + 1) % 5 == 0:
            print(f"Val Accuracy after query {index+1}: {model_accuracy_val:0.4f}")
        perf_hist.append(model_accuracy_val)
    model_accuracy_test = learner.score(X_test, y_test)
    print(f"********** Test Accuracy per experiment: {model_accuracy_test} **********")
    return perf_hist, model_accuracy_test


def select_acq_function(acq_func: int = 0) -> list:
    """Choose types of acqusition function

    Attributes:
        acq_func: 0-all(unif, max_entropy, bald), 1-unif, 2-maxentropy, 3-bald, \
                  4-var_ratios, 5-mean_std
    """
    acq_func_dict = {
        0: [uniform, max_entropy, bald, var_ratios, mean_std],
        1: [uniform],
        2: [max_entropy],
        3: [bald],
        4: [var_ratios],
        5: [mean_std],
        6: [batch_bald],
    }
    return acq_func_dict[acq_func]

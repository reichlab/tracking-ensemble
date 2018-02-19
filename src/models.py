"""
Models
"""

from abc import ABC, abstractmethod
from collections import Counter
from functools import lru_cache
from typing import List, Tuple
import json
import utils.dists as udists
import utils.misc as u
import numpy as np
import pandas as pd


EPSILON = np.sqrt(np.finfo(float).eps)


def beta_softmax(vector, beta):
    """
    Calculate softmax with a beta (inverse parameter)
    """

    expv = np.exp(beta * vector)
    return expv / np.sum(expv)


def dem(mat, weights=None, epsilon=None):
    """
    Run the degenerate EM algorithm on given data. Return a set of weights for
    each model. Code replicates the method implemented in epiforecast-R package
    here https://github.com/cmu-delphi/epiforecast-R/blob/master/epiforecast/R/ensemble.R
    Parameters
    ----------
    mat : np.ndarray
        Shape (n_obs, n_models). Probabilities from n_models models for n_obs
        observations.
    weights : np.ndarray
        Initial weights
    epsilon : float
        Tolerance value
    """

    if weights is None:
        weights = np.ones(mat.shape[1]) / mat.shape[1]

    if not epsilon:
        epsilon = EPSILON

    w_mat = mat * weights
    marginals = np.sum(w_mat, axis=1)
    log_marginal = np.mean(np.log(marginals))

    if np.isneginf(log_marginal):
        raise ValueError("All methods assigned a probability of 0 to at least one observed event.")
    else:
        while True:
            prev_log_marginal = log_marginal
            weights = np.mean(w_mat.T / marginals, axis=1)
            w_mat = mat * weights
            marginals = np.sum(w_mat, axis=1)
            log_marginal = np.mean(np.log(marginals))

            if log_marginal + epsilon < prev_log_marginal:
                raise ValueError("Log marginal less than prev_log_marginal")
            marginal_diff = log_marginal - prev_log_marginal
            if (marginal_diff <= epsilon) or ((marginal_diff / -log_marginal) <= epsilon):
                break
    return weights


class Model(ABC):
    """
    A general time series model. Training is batched while prediction is not.
    There is also a feedback method to patch things up.
    """

    @abstractmethod
    def train(self, index_vec, component_predictions_vec, truth_vec):
        """
        Train the model

        Parameters
        ----------
        index_vec : pd.DataFrame
            Dataframe with atleast two columns "epiweek" and "region".
            Shape is (n_instances, 2).
        component_predictions_vec : List[np.ndarray]
            One matrix for each component. Matrix is (n_instances, n_bins).
        truth_vec : np.ndarray
            True values. Shape (n_instances, )
        """
        ...

    @abstractmethod
    def predict(self, index, component_predictions, truth=None):
        """
        Predict output for a single timepoint

        Parameters
        ----------
        index : List/Tuple
            Pair of epiweek and region values
        component_predictions : List[np.ndarray]
            One vector for each component. Vector is of shape (n_bins, )
        truth
            True value for the time point. This is only used by oracular models.
        """
        ...

    @abstractmethod
    def feedback(self, last_truth):
        """
        Take feedback in the form of last truth. This can then be used for updating the
        weights.
        """
        ...

    @abstractmethod
    def save(self, file_path: str):
        """
        Save model parameters to the file given
        """
        ...

    @abstractmethod
    def load(self, file_path: str):
        """
        Load model parameters from the file given
        """
        ...


class OracleEnsemble(Model):
    """
    Oracular ensemble. Outputs the prediction from the best model.
    """

    def __init__(self, target: str, n_comps: int):
        self.target = target
        self.n_comps = n_comps

    def train(self, index_vec, component_predictions_vec, truth_vec):
        pass

    def predict(self, index, component_predictions, truth):
        """
        Use the truth to identify the best component. Then output its
        prediction
        """

        one_hot = udists.actual_to_one_hot(np.array([truth]), self.target)
        best_model_idx = np.argmax((np.array(component_predictions) * one_hot).sum(axis=1))
        return component_predictions[best_model_idx]

    def feedback(self, component_losses):
        pass

    def save(self, file_name):
        pass

    def load(self, file_name):
        pass


class MeanEnsemble(Model):
    """
    Mean ensemble. Outputs the mean of predictions from the components.
    """

    def __init__(self, target: str, n_comps: int):
        self.target = target
        self.n_comps = n_comps

    def train(self, index_vec, component_predictions_vec, truth_vec):
        pass

    def predict(self, index, component_predictions):
        """
        Use the truth to identify the best component. Then output its
        prediction
        """

        return udists.mean_ensemble(component_predictions)

    def feedback(self, component_losses):
        pass

    def save(self, file_name):
        pass

    def load(self, file_name):
        pass


class DemWeightEnsemble(Model):
    """
    Degenerate EM ensemble.
    """

    def __init__(self, target: str, n_comps: int):
        self.target = target
        self.n_comps = n_comps

    def train(self, index_vec, component_predictions_vec, truth_vec):
        """
        Use degenerate EM to find the best set of weights optimizing the log scores
        """

        probabilities = udists.prediction_probabilities(component_predictions_vec, truth_vec, self.target)
        self.weights = dem(probabilities)

    def predict(self, index, component_predictions):
        """
        Use the truth to identify the best component. Then output its
        prediction
        """

        return udists.weighted_ensemble(component_predictions, self.weights)

    def feedback(self, component_losses):
        pass

    def save(self, file_name):
        with open(file_name, "w") as fp:
            json.dump({ "weights": self.weights }, fp)

    def load(self, file_name):
        with open(file_name) as fp:
            self.weights = json.load(fp)["weights"]


class HitWeightEnsemble(Model):
    """
    Ensemble that weighs components according to the number of times they have
    been the best. This is similar to the score weight ensemble but since hits
    are relatively sparse (by definition), we make this a whole training data
    thing as compared to the score weight model which is per week.
    """

    def __init__(self, target: str, n_comps: int, beta: float):
        """
        Parameters
        ----------
        target : str
            Target identifier
        n_comps : int
            Number of components
        beta : float
            Beta for the softmax
        """

        self.target = target
        self.n_comps = n_comps
        self.beta = beta

    def train(self, index_vec, component_predictions_vec, truth_vec):
        """
        Count the number of best hits and pass through the softmax to get weights
        """

        probabilities = udists.prediction_probabilities(component_predictions_vec, truth_vec, self.target)
        hits = Counter(np.argmax(probabilities, axis=1))
        self.weights = beta_softmax([hits[i] for i in range(self.n_comps)], self.beta)

    def predict(self, index, component_predictions):
        """
        Use the truth to identify the best component. Then output its
        prediction
        """

        return udists.weighted_ensemble(component_predictions, self.weights)

    def feedback(self, component_losses):
        pass

    def save(self, file_name):
        with open(file_name, "w") as fp:
            json.dump({ "weights": self.weights, "beta": self.beta }, fp)

    def load(self, file_name):
        with open(file_name) as fp:
            data = json.load(fp)
            self.weights = data["weights"]
            self.beta = data["beta"]


class ScoreWeightEnsemble(Model):
    """
    Ensemble that weighs components according to the scores they get for each model week.
    """

    def __init__(self, target: str, n_comps: int, beta: float):
        """
        Parameters
        ----------
        target : str
            Target identifier
        n_comps : int
            Number of components
        beta : float
            Beta for the softmax
        """

        self.target = target
        self.n_comps = n_comps
        self.beta = beta

    def train(self, index_vec, component_predictions_vec, truth_vec):
        """
        Group the scores according to model weeks
        """

        model_weeks = index_vec["epiweek"].map(u.epiweek_to_model_week)
        probabilities = udists.prediction_probabilities(component_predictions_vec, truth_vec, self.target)

        dfdict = { "model_weeks": model_weeks }
        for i in range(self.n_comps):
            dfdict[i] = probabilities[:, i]

        # Mean probabilities per model week
        mean_probabilities = pd.DataFrame(dfdict).groupby("model_weeks").mean()
        self.model_weeks = list(mean_probabilities.index)

        # beta softmax simplifies since log and exp cancel
        probs = mean_probabilities.values
        probs **= self.beta
        self.weights = probs / probs.sum(axis=1)[:, None]

    def predict(self, index, component_predictions):
        """
        Use the truth to identify the best component. Then output its
        prediction
        """

        model_week = u.epiweek_to_model_week(index["epiweek"])
        weights = self.weights[self.model_weeks.index(model_week)]

        return udists.weighted_ensemble(component_predictions, weights)

    def feedback(self, component_losses):
        pass

    def save(self, file_name):
        with open(file_name, "w") as fp:
            data = {
                "weights": self.weights.tolist(),
                "beta": self.beta,
                "model_weeks": self.model_weeks
            }
            json.dump(data, fp)

    def load(self, file_name):
        with open(file_name) as fp:
            data = json.load(fp)
            self.weights = np.array(data["weights"])
            self.beta = data["beta"]
            self.model_weeks = pd.Index(data["model_weeks"])


class KDemWeightEnsemble(Model):
    """
    Degenerate EM ensemble trained on optimal k parition of epiweeks.
    """

    def __init__(self, target: str, n_comps: int, k: int):
        self.target = target
        self.n_comps = n_comps
        self.k = k

    @lru_cache(None)
    def _score_partition(self, start_wk, length):
        """
        For a partition specified by start_wk and length, fit weights
        and find score
        """

        # Model weeks in the partition
        weeks = list(range(start_wk, start_wk + length))
        selection = self.index.isin(weeks)

        weights = dem(self.probabilities[selection])
        score = (self.probabilities[selection] * weights).sum(axis=1).mean()

        return np.log(score), weights

    @lru_cache(None)
    def _partition(self, start_wk, k):
        """
        Find optimal number of partitions
        """

        if k == 1:
            # We work on the complete remaining chunk
            length = self.data["nweeks"] - start_wk
            score, weights = self._score_partition(start_wk, length)
            return score, [length], [weights]

        optimal_score = -np.inf
        optimal_partitions = []
        optimal_weights = []

        for length in range(1, self.data["nweeks"] - (k - 1) - start_wk):
            score, weights = self._score_partition(start_wk, length)
            rest_score, rest_partitions, rest_weights = self._partition(start_wk + length, k - 1)

            # Find the mean of scores
            rest_length = self.data["nweeks"] - start_wk - length
            total_score = ((score * length) + (rest_score * rest_length)) / (length + rest_length)

            if total_score > optimal_score:
                optimal_score = total_score
                optimal_partitions = [length, *rest_partitions]
                optimal_weights = [weights, *rest_weights]

        return optimal_score, optimal_partitions, optimal_weights

    def train(self, index_vec, component_predictions_vec, truth_vec):
        """
        Use degenerate EM to find the best set of weights optimizing the log scores
        """

        self.probabilities = udists.prediction_probabilities(component_predictions_vec, truth_vec, self.target)
        self.index = index_vec["epiweek"].map(u.epiweek_to_model_week)

        score, partitions, weights = self._partition(0, self.k)
        self.data {
            "partitions": partitions,
            "weights": weights,
            "nweeks": len(np.unique(self.index))
        }

        print(f"Training complete for {self.k} partitions, best score {score}")

    def predict(self, index, component_predictions):
        """
        Use the truth to identify the best component. Then output its
        prediction
        """

        model_week = u.epiweek_to_model_week(index["epiweek"])
        c_partitions = np.cumsum(self.data["partitions"])
        partition_idx = np.sum(np.cumsum(c_partitions) <= model_week)
        weights = self.data["weights"][partition_idx]

        return udists.weighted_ensemble(component_predictions, weights)

    def feedback(self, component_losses):
        pass

    def save(self, file_name):
        with open(file_name, "w") as fp:
            data = {
                "k": self.k,
                "partitions": self.data["partitions"],
                "weights": [w.tolist() for w in self.data["weights"]],
                "nweeks": self.data["nweeks"]
            }
            json.dump({ "weights": self.weights }, fp)

    def load(self, file_name):
        with open(file_name) as fp:
            data = json.load(fp)
            self.weights = data["weights"]
            self.k = data["k"]
            self.data = {
                "nweeks": data["nweeks"],
                "weights": data["weights"],
                "partitions": data["partitions"]
            }


class MPWeightEnsemble(Model):
    pass

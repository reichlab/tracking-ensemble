"""
Models
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
import utils.dists as udists
import numpy as np


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
        epsilon = np.sqrt(np.finfo(float).eps)

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
    pass


class HitWeightEnsemble(Model):
    pass


class ScoreWeightEnsemble(Model):
    pass


class KDemWeightEnsemble(Model):
    pass


class MPWeightEnsemble(Model):
    pass

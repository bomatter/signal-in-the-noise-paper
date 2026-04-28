import numpy as np
from scipy.stats import gaussian_kde


class KDE:
    def __init__(self, features):
        """
        features: list of column names representing the summary statistics.
        """
        self.features = features
        self.kdes = {}

    def fit(self, df_train, subsample=None):
        """
        Fits an independent 1D KDE for each summary statistic using the training set.

        For computational efficiency, the training data can be subsampled.
        `subsample` specifies the maximum number of samples to use; if the training
        set is larger, a random subset will be used to fit the KDEs.
        """
        if subsample is not None and len(df_train) > subsample:
            print(f"Subsampling training data from {len(df_train)} to {subsample} samples for KDE fitting.")
            df_train = df_train.sample(n=subsample, random_state=42)

        for feature in self.features:
            train_data = df_train[feature].values
            self.kdes[feature] = gaussian_kde(train_data)

    def score(self, df_test):
        """
        The score is the sum of the log-probabilities across all independent KDEs.
        """

        total_log_prob = np.zeros(len(df_test))
        for feature in self.features:
            test_data = df_test[feature].values
            feature_log_prob = self.kdes[feature].logpdf(test_data)
            total_log_prob += feature_log_prob

        return total_log_prob


class MaxQuantile:
    def __init__(self, feature_configs):
        """
        feature_configs: A dictionary mapping column names to higher_is_ood booleans.
        Example: {"anderson_darling_statistic": True, "ps_cv": True}
        """
        self.feature_configs = feature_configs
        self.reference_data = {}

    def fit(self, df_ref):
        for feature in self.feature_configs:
            self.reference_data[feature] = np.sort(df_ref[feature].values)

    def score(self, df_test):
        """
        Computes the max percentile rank (considering OOD direction) across all features.
        """
        all_quantiles = []

        for feature, higher_is_ood in self.feature_configs.items():
            sorted_ref = self.reference_data[feature]
            test_vals = df_test[feature].values

            # Compute percentile rank [0, 1]
            q = np.searchsorted(sorted_ref, test_vals, side="right") / len(sorted_ref)

            # Flip if lower is OOD so that 1.0 always means "most OOD-like"
            if not higher_is_ood:
                q = 1.0 - q

            all_quantiles.append(q)

        # Return the element-wise maximum across all metrics
        return np.max(all_quantiles, axis=0)

# pylint: disable=missing-module-docstring
import numpy as np
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd


class HierarchicalRiskParity:
    """
    This class implements the Hierarchical Risk Parity algorithm mentioned in the following paper: `López de Prado, Marcos,
    Building Diversified Portfolios that Outperform Out-of-Sample (May 23, 2016). Journal of Portfolio Management,
    2016 <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678>`_; The code is reproduced with modification from his book:
    Advances in Financial Machine Learning, Chp-16
    By removing exact analytical approach to the calculation of weights and instead relying on an approximate
    machine learning based approach (hierarchical tree-clustering), Hierarchical Risk Parity produces weights which are stable to
    random shocks in the stock-market. Moreover, previous algorithms like CLA involve the inversion of covariance matrix which is
    a highly unstable operation and tends to have major impacts on the performance due to slight changes in the covariance matrix.
    By removing dependence on the inversion of covariance matrix completely, the Hierarchical Risk Parity algorithm is fast,
    robust and flexible.
    """

    def __init__(self, asset_names, linkage='single'):
        self.asset_names = np.array(asset_names)
        self.linkage = linkage
        
    @staticmethod
    def _get_cluster_var(covariance_matrix, cluster_items):
        """
        Compute the variance per cluster

        :param cov: covariance matrix
        :type cov: np.ndarray
        :param cluster_items: tickers in the cluster
        :type cluster_items: list
        :return: the variance per cluster
        :rtype: float
        """
        # Compute variance per cluster
        cov_slice = covariance_matrix[np.ix_(cluster_items, cluster_items)]
        w = 1 / np.diag(cov_slice)  # Inverse variance weights
        w /= w.sum()
        return np.linalg.multi_dot((w, cov_slice, w))

    def _allocate(self, cov, ordered_indices, ordered_tickers):
        """
        Given the clusters, compute the portfolio that minimises risk by
        recursively traversing the hierarchical tree from the top.
        """
        weights = np.full_like(ordered_indices, fill_value=1.0, dtype=np.float64)
        cluster_items = [ordered_indices]  # initialize all items in one cluster

        while len(cluster_items) > 0:
            cluster_items = [
                i[start:end]
                for i in cluster_items
                # halving algorithm
                # start -> first half
                # end -> second half
                for start, end in ((0, len(i) // 2), (len(i) // 2, len(i)))
                if len(i) > 1
            ]
            # bi-section
            # For each pair, optimise locally.
            for subcluster in range(0, len(cluster_items), 2):
                left_cluster = cluster_items[subcluster]
                right_cluster = cluster_items[subcluster + 1]

                # Form the inverse variance portfolio for this pair
                left_variance = self._get_cluster_var(cov, left_cluster)
                right_variance = self._get_cluster_var(cov, right_cluster)

                allocation_factor = 1 - left_variance / (right_variance + left_variance)
                weights[left_cluster] *= allocation_factor  # weight 1
                weights[right_cluster] *= 1 - allocation_factor  # weight 2

            return weights

    @staticmethod
    def build_long_short_portfolio(weights, side_weights):
        """
        Adjust weights according the shorting constraints specified.
        :param side_weights: (numpy matrix) With asset_names in index and value 1 for Buy, -1 for Sell
                                                      (default 1 for all)
        """

        short_ptf = np.where(side_weights == -1)
        buy_ptf = np.where(side_weights == 1)
        if len(short_ptf) > 0:

            # Short half size
            weights[short_ptf] /= np.sum(weights[short_ptf])
            weights[short_ptf] *= -0.5

            # Buy other half
            weights[buy_ptf] /= np.sum(weights[buy_ptf])
            weights[buy_ptf] *= 0.5

        return weights

    def optimize(self, asset_prices: np.ndarray = None, asset_returns: np.ndarray = None, distance_matrix: np.ndarray = None, side_weights):
        if asset_return is not None:
            asset_returns = np.diff(asset_prices) / asset_prices[:, :-1]

        # Calculate covariance and correlation of returns
        covariance_matrix = np.cov(asset_returns, bias=False)
        correlation_matrix = np.corrcoef(asset_returns)
        
        if distance_matrix is not None:
            # Calculate distance from covariance matrix
            distance_matrix = np.sqrt((1 - correlation_matrix).round(5) / 2)

        # Step-1: Tree Clustering
        clusters = sch.linkage(ssd.squareform(distance_matrix), method=self.linkage)

        # Step-2: Quasi Diagnalization
        ordered_indices = sch.to_tree(clusters, rd=False).pre_order()
        ordered_tickers = self.asset_names[ordered_indices]

        # Step-3: Recursive Bisection
        weights = self._allocate(covariance_matrix, ordered_indices, ordered_tickers)

        self.weights = weights

        weights = self.build_long_short_portfolio(weights, np.array(side_weights))
        
        return dict(zip(self.asset_names, weights)
    

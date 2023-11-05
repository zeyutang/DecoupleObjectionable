# utils/simulation_utils.py
# Simulation related utilities

import numpy as np
from .graph_utils import topological_sort


class SimulationCausalModel:
    """
    A class used to generate a dataset based on a causal model with specified characteristics.

    For simulation purpose, all variables (except for root nodes) are continuous.

    Attributes
    ----------
    nodes : list of str
        A list of node names in the causal graph.

    node_types : dict
        A dictionary of node types ('binary' or 'continuous').

    parent_dict : dict
        A dictionary with node names as keys and a list of their parent nodes as values.

    is_linear : bool
        If the type of relationships between nodes is linear.

    exogenous_noise_std : float
        The standard deviation of the Gaussian noise added to the generated data.

    relation_functions_dict : dict
        A dictionary with node names as keys and a list of relation functions as values.

    rng : Generator
        A random number generator.

    Methods
    -------
    generate_data(n_samples: int) -> Dict[str, np.NDArray]:
        Generates data for the specified causal model with the given number of samples.
        Returns a dictionary with node names as keys and their corresponding data as values.

    Example
    -------
    >>> nodes = ['A', 'B', 'C', 'D', 'Y']
    >>> parent_dict = {
        'A': [],
        'B': [],
        'C': ['A', 'B'],
        'D': ['B', 'C'],
        'Y': ['C', 'D']
    }
    >>> node_types = {
        'A': 'binary',
        'B': 'continuous',
        'C': 'continuous',
        'D': 'continuous',
        'Y': 'continuous'
    }
    >>> sim_causal_model = SimulationCausalModel(nodes, node_types, parent_dict, is_linear=True)
    >>> rng = np.random.default_rng(42)
    >>> n_samples = 10000
    >>> input_data = {
        node: rng.normal(size=n_samples)
        for node in nodes if not parent_dict[node]
    }
    >>> sim_causal_model.nodes = input_data
    >>> data = sim_causal_model.generate_data(n_samples)
    """

    def __init__(
        self,
        nodes,
        node_types,
        parent_dict,
        is_linear=True,
        exogenous_noise_std=0.5,
        coef_choices=[-2, -1.5, -1, 1, 1.5, 2],
        rng=None,
    ):
        self.nodes = nodes
        self.node_types = node_types
        self.parent_dict = parent_dict
        self.is_linear = is_linear

        self.exogenous_noise_std = exogenous_noise_std
        self.coef_choices = coef_choices
        if None == rng:
            self.rng = np.random.default_rng(2023)
        else:
            self.rng = rng

        self = topological_sort(self)
        self.relation_functions_dict = self._generate_relation_functions_dict()

    def _generate_relation_functions_dict(self):
        relation_functions_dict = {}
        functions = [np.sin, "poly"]

        for node, parents in self.parent_dict.items():
            node_func = []

            # Linear combination, term idx range(0, n_parents).
            for _ in range(len(parents)):
                coef = self.rng.choice(self.coef_choices)
                node_func.append(lambda x, coef=coef: coef * x)

            if not self.is_linear:
                # Post-nonlinear transformation of the linear combination, term idx n_parents
                # additional individual nonlinear terms, term idx n_parents + range(0, n_parents)
                for _ in range(1 + len(parents)):
                    coef = self.rng.choice(self.coef_choices)
                    func = self.rng.choice(functions)
                    if func == "poly":
                        power = self.rng.choice([1, 2])
                        node_func.append(lambda x: coef * (x**power))
                    else:
                        node_func.append(lambda x, coef=coef, func=func: coef * func(x))

            relation_functions_dict[node] = node_func

        return relation_functions_dict

    def generate_data(self, n_samples):
        data = {node: np.zeros(n_samples) for node in self.nodes}

        for node, parents in self.parent_dict.items():
            node_data = np.zeros(n_samples)

            # If root node
            if 0 == len(parents) and "binary" == self.node_types[node]:
                # Select a proportion, and set to 1
                proportions = [0.3, 0.4, 0.5, 0.6, 0.7]
                _proportion = self.rng.choice(proportions)

                # Random choose a indices to set to 1
                random_idx = self.rng.choice(
                    n_samples, int(_proportion * n_samples), replace=False
                )
                node_data[random_idx] = 1

            elif "binary" == self.node_types[node]:  # not root node
                raise ValueError(
                    "Only root nodes can be binary when generating simulation data. One can mannually set final output to binary if necessary."
                )

            else:
                # Linear combination.
                for i, parent in enumerate(parents):
                    parent_data = data[parent]
                    node_data += self.relation_functions_dict[node][i](parent_data)

                # Post-nonlinear transformation.
                if not self.is_linear:
                    post_nonlinear_idx = len(parents)
                    # Need to incorporate independent noise in temp node data.
                    _temp_node_data = node_data + self.rng.normal(
                        0, self.exogenous_noise_std, n_samples
                    )
                    # Post-nonlinear term.
                    node_data = self.relation_functions_dict[node][post_nonlinear_idx](
                        _temp_node_data
                    )

                    # Additional individual nonlinear terms
                    # for i, parent in enumerate(parents):
                    #     parent_data = data[parent]
                    #     nonlinear_term_idx = post_nonlinear_idx + 1 + i
                    #     node_data += self.relation_functions_dict[node][
                    #         nonlinear_term_idx](parent_data)

                # Final exogenous noise addon.
                node_data += self.rng.normal(0, self.exogenous_noise_std, n_samples)

            data[node] = node_data

        return data

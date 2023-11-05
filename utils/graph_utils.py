# utils/graph_utils.py
# Graph related utilities

from .misc_utils import (
    OPTION_REFERENCE_POINT,
    OPTION_REFERENCE_DOWNSTREAM,
    OPTION_ORIGINAL_VALUE,
)


class CausalGraph:
    """
    A directed acyclic graph (DAG) for representing causal relationships among variables.

    Attributes:
    -----------
    nodes : list
        List of node names in the causal graph.

    parent_dict : dict
        Dictionary that maps each node to its list of parent nodes in the graph.

    node_types : dict
        Dictionary that maps each node to its type ('continuous' or 'binary').
        For non-binary categorical variables, use 'continuous'.

    Methods:
    --------
    out_degree(node: str) -> int
        Returns the number of child nodes (out-degree) for a given node.

    in_degree(node: str) -> int
        Returns the number of parent nodes (in-degree) for a given node.

    ancestor_dict() -> dict
        Returns a dictionary where the key is a node, and the value is a list containing the node's ancestors.

    descendant_dict() -> dict
        Returns a dictionary where the key is a node, and the value is a list containing the node's descendants.

    generate_parent_dict_enhanced(tail_ref_pt_config: str) -> dict
        Generates an enhanced version of the parent_dict that includes value instantiation rules.

    _update_descendants(parent_dict_enhanced: dict, current_node: str)
        Recursively updates the parent_dict_enhanced entries for the descendants of a given node.
    """

    def __init__(self, nodes, parent_dict, node_types):
        self.nodes = nodes
        self.parent_dict = parent_dict
        self.node_types = node_types

        self = topological_sort(self)

    def out_degree(self, node):
        """
        The number of direct children of a node.
        """
        return sum(
            [1 for parent_list in self.parent_dict.values() if node in parent_list]
        )

    def in_degree(self, node):
        """
        The number of parents of a node.
        """
        return len(self.parent_dict[node])

    def ancestor_dict(self):
        ancestor_dict = {}
        for node in self.nodes:
            ancestor_dict[node] = self._get_ancestors(node, set())
        return ancestor_dict

    def _get_ancestors(self, node, visited_nodes):
        ancestors = []
        for parent_node in self.parent_dict[node]:
            if parent_node not in visited_nodes:
                ancestors.append(parent_node)
                visited_nodes.add(parent_node)
                ancestors.extend(self._get_ancestors(parent_node, visited_nodes))
        return ancestors

    def descendant_dict(self):
        descendant_dict = {}
        for node in self.nodes:
            descendant_dict[node] = self._get_descendants(node, set())
        return descendant_dict

    def _get_descendants(self, node, visited_nodes):
        descendants = []
        for child_node, parent_list in self.parent_dict.items():
            if node in parent_list and child_node not in visited_nodes:
                descendants.append(child_node)
                visited_nodes.add(child_node)
                descendants.extend(self._get_descendants(child_node, visited_nodes))
        return descendants

    def generate_parent_dict_enhanced(self, tail_ref_pt_config):
        """
        Generate an enhanced version of the parent_dict that includes value instantiation rules.

        Input:
        ------
        tail_ref_pt_config: dict
            Dictionary that maps tail_node -> head_node pairs to their numeric values,
            indicating specific reference point configurations in the graph.

        Output:
        -------
        parent_dict_enhanced: dict
            Dictionary containing tuples (parent_node, value_instantiation_rule) for each node,
            where value_instantiation_rule is one of the following: 'original_value', 'reference_point', or 'reference_downstream'.

        Example:
        --------
        >>> from utils import compare_dicts
        >>>
        >>> # --> Example 1: regulated edges form a path between A to Y
        >>> nodes = ['A', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'Y']
        >>> parent_dict = {
            'A': [],
            'X1': ['A'],
            'X2': ['A'],
            'X3': [],
            'X4': ['X1', 'X2'],
            'X5': ['X1', 'X2', 'X3', 'X4'],
            'X6': ['X3'],
            'Y': ['X4', 'X5', 'X6']
        }
        >>> node_types = {node: 'continuous' for node in nodes}
        >>> tail_ref_pt_config = {'A->X1': 1.0, 'X1->X4': 2.0, 'X4->Y': 0.5}
        >>> causal_graph = CausalGraph(nodes, parent_dict, node_types)
        >>> answer = causal_graph.generate_parent_dict_enhanced(tail_ref_pt_config)
        >>> # compare with correct answer
        >>> truth = {
            'A': [],
            'X1': [('A', 'reference_point')],
            'X2': [('A', 'original_value')],
            'X3': [],
            'X4': [('X1', 'reference_point'), ('X2', 'original_value')],
            'X5': [('X1', 'reference_downstream'), ('X2', 'original_value'),
                ('X3', 'original_value'), ('X4', 'reference_downstream')],
            'X6': [('X3', 'original_value')],
            'Y': [('X4', 'reference_point'), ('X5', 'reference_downstream'),
                ('X6', 'original_value')]
        }
        >>> compare_dicts(truth, answer)
        >>>
        >>> # --> Example 2: regulated edges at different places
        >>> nodes = ['A', 'X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'Y']
        >>> parent_dict = {
            'X0': [],
            'A': ['X0'],
            'X1': ['A'],
            'X2': ['A'],
            'X3': ['X0'],
            'X4': ['X1', 'X2'],
            'X5': ['X1', 'X2', 'X3', 'X4'],
            'X6': ['X3', 'X5'],
            'X7': ['X4'],
            'X8': ['X5', 'X6'],
            'X9': ['X6'],
            'Y': ['X3', 'X7', 'X8', 'X9']
        }
        >>> node_types = {node: 'continuous' for node in nodes}
        >>> tail_ref_pt_config = {'A->X2': 1.0, 'X2->X4': 2.0, 'X3->X6': 0.5, 'X5->X8': -1.0, 'X7->Y': -2.0}
        >>> causal_graph = CausalGraph(nodes, parent_dict, node_types)
        >>> answer = causal_graph.generate_parent_dict_enhanced(tail_ref_pt_config)
        >>> # compare with correct answer
        >>> truth = {
            'A': [('X0', 'original_value')],
            'X0': [],
            'X1': [('A', 'original_value')],
            'X2': [('A', 'reference_point')],
            'X3': [('X0', 'original_value')],
            'X4': [('X1', 'original_value'), ('X2', 'reference_point')],
            'X5': [('X1', 'original_value'), ('X2', 'reference_downstream'),
                ('X3', 'original_value'), ('X4', 'reference_downstream')],
            'X6': [('X3', 'reference_point'), ('X5', 'reference_downstream')],
            'X7': [('X4', 'reference_downstream')],
            'X8': [('X5', 'reference_point'), ('X6', 'reference_downstream')],
            'X9': [('X6', 'reference_downstream')],
            'Y': [('X3', 'original_value'), ('X7', 'reference_point'),
                ('X8', 'reference_downstream'), ('X9', 'reference_downstream')]
        }
        >>> compare_dicts(truth, answer)
        """
        # Initialize parent_dict_enhanced with original_value as default
        parent_dict_enhanced = {
            node: [(parent, OPTION_ORIGINAL_VALUE) for parent in parent_list]
            for node, parent_list in self.parent_dict.items()
        }

        if tail_ref_pt_config is not None:
            # Iterate over edges in tail_ref_pt_config and update parent_dict_enhanced accordingly
            for edge_key, _ in tail_ref_pt_config.items():
                tail_node, head_node = edge_key.split("->")

                # Check if the edge exists in the parent_dict.
                if tail_node in self.parent_dict.get(head_node, []):
                    # Update the parent_dict_enhanced entry for the head_node.
                    for idx, (parent, _) in enumerate(parent_dict_enhanced[head_node]):
                        if parent == tail_node:
                            parent_dict_enhanced[head_node][idx] = (
                                tail_node,
                                OPTION_REFERENCE_POINT,
                            )

                    # Update the parent_dict_enhanced entries for the descendants of the head_node
                    self._update_descendants(parent_dict_enhanced, head_node)

                else:
                    # Print a warning if the edge does not exist in the parent_dict
                    print(
                        f"Warning: Edge '{edge_key}' does not exist in the CausalGraph. Ignoring tail_ref_pt_config item."
                    )

        return parent_dict_enhanced

    def _update_descendants(self, parent_dict_enhanced, current_node):
        """
        Update parent_dict_enhanced entries for the descendants of the current_node.

        Input:
        ------
        parent_dict_enhanced: dict
            Dictionary containing tuples (parent_node, value_instantiation_rule) for each node.

        current_node: str
            The node whose descendants need to be updated.
        """
        # Iterate over the parent_dict to find child nodes of the current_node
        for child_node, parent_list in self.parent_dict.items():
            if current_node in parent_list:
                # Iterate over the parents of the child node of current_node
                for idx, (parent_node, option) in enumerate(
                    parent_dict_enhanced[child_node]
                ):
                    if parent_node == current_node and option == OPTION_ORIGINAL_VALUE:
                        parent_dict_enhanced[child_node][idx] = (
                            parent_node,
                            OPTION_REFERENCE_DOWNSTREAM,
                        )

                        # Update the parent_dict_enhanced entries for the descendants
                        # of the child_node recursively
                        self._update_descendants(parent_dict_enhanced, child_node)


def topological_sort(causal_graph):
    """
    Perform a topological sort on a directed acyclic graph (DAG) to ensure that parent nodes
    of any node appear before the node itself.

    Input:
    ------
    causal_graph: CausalGraph
        A DAG containing attributes (at least) nodes, parent_dict.

    Output:
    -------
    causal_graph: CausalGraph
        A DAG that has the same structure as parent_dict but with nodes sorted
        in topological order.
    """
    if not hasattr(causal_graph, "nodes"):
        raise ValueError("No attribute nodes found in CausalModel.")
    if not hasattr(causal_graph, "parent_dict"):
        raise ValueError("No attribute parent_dict found.")
    if not set(causal_graph.parent_dict.keys()) == set(causal_graph.nodes):
        raise RuntimeError("The sets of nodes do not match.")

    sorted_nodes = []
    visited = set()

    def visit(node):
        """
        Perform a depth-first search (DFS) on the DAG, visiting parent nodes before child nodes.
        """
        if node not in visited:
            visited.add(node)
            for parent in causal_graph.parent_dict[node]:
                visit(parent)
            sorted_nodes.append(node)

    for node in causal_graph.parent_dict:
        visit(node)

    sorted_parent_dict = {node: causal_graph.parent_dict[node] for node in sorted_nodes}

    # Update nodes and parent_dict
    causal_graph.nodes = sorted_nodes
    causal_graph.parent_dict = sorted_parent_dict

    return causal_graph

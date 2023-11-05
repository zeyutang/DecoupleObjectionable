# utils/network_utils.py
# Neural network related utilities

import numpy as np
import torch
import torch.nn as nn
from .misc_utils import (
    extract_value_with_priority,
    OPTION_REFERENCE_POINT,
    OPTION_REFERENCE_DOWNSTREAM,
    OPTION_ORIGINAL_VALUE,
)


class SubNetworkLinear(nn.Module):
    """
    Local causal module in the CausalGraph, use linear model hypothesis class.

    Assume each node is 1D valued. If not, make sure to separate into multiple nodes.

    NOTE For final output node, ensure the output is between 0 and 1 while maintaining a mostly
    linear relationship. To this end, we use a Hardtanh activation function for the final output.

    Attributes:
    -----------
    input_dim : int
        The input dimension for the submodel.

    output_dim : int
        The output dimension for the submodel.

    output_type : str
        The data type of output node has to be 'continuous'.

    Methods:
    --------
    forward: Computes the forward pass through the submodel.
    """

    def __init__(self, input_dim, output_dim, output_type):
        super(SubNetworkLinear, self).__init__()

        self.output_type = output_type
        if output_type == "binary":
            raise RuntimeError("If use linear model, output has to be continuous.")
        elif output_type in ["continuous", "continuous_final_output"]:
            self.linear_layer = nn.Linear(input_dim, output_dim)

            if output_type == "continuous":
                self.last_layer = nn.Identity()
            else:  # final output only
                self.last_layer = nn.Hardtanh(min_val=0.0, max_val=1.0)
        else:
            raise ValueError(f"Unsupported output_type: {output_type}.")

    def forward(self, x):
        x = self.linear_layer(x)
        x = self.last_layer(x)
        return x


class SubNetworkNonlinear(nn.Module):
    """
    Local causal module in the CausalGraph, use nonlinear hypothesis class.

    Assume each node is 1D valued. If not, make sure to separate into multiple nodes.

    NOTE For final output node, ensure the output is between 0 and 1. For nonlinear models,
    We use a Sigmoid activation function so that the output is in the range [0, 1].

    Attributes:
    -----------
    input_dim : int
        The input dimension for the submodel.

    output_dim : int
        The output dimension for the submodel.

    hidden_dim : int
        The dimension of the hidden layers for the submodel.

    output_type : str
        The data type of output node ('continuous' or 'binary').
        For categorical variables, if cardinality is larger than 2, treat it as continuous.
        The type 'continuous_final_output' will be automatically set by CustomNetwork
        if the final output is continuous.

    Methods:
    --------
    forward: Computes the forward pass through the submodel.
    """

    def __init__(self, input_dim, output_dim, hidden_dim, output_type):
        super(SubNetworkNonlinear, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SELU(),
        )

        self.output_type = output_type
        if output_type == "continuous":
            self.last_layer = nn.Linear(hidden_dim, output_dim)
        elif output_type in ["binary", "continuous_final_output"]:
            # For binary, output Sigmoid(logits), s.t. BCELoss can be used
            # this is to make sure the output of CustomNetwork is in [0, 1]
            self.last_layer = nn.Sequential(
                nn.Linear(hidden_dim, output_dim), nn.Sigmoid()
            )
        else:
            raise ValueError(f"Unsupported output_type: {output_type}.")

    def forward(self, x):
        x = self.layers(x)
        x = self.last_layer(x)
        return x


class CustomNetwork(nn.Module):
    """
    Custom network corresponding to the causal graph.

    According to the causal graph, construct the model via SubNetwork for local causal mechanisms.

    An additional node type is used: {'binary', 'continuous', 'continuous_final_output'}.
    For intermediate variables (outputs of SubNetworks), if they are non-binary categorical
    variables, treat them as continuous.

    NOTE
    - Assume each node is 1D valued. If not, make sure to separate into multiple nodes.
    - Assume inputs for each non-target variable are standard scaled (mean 0, std 1).
    - Assume final output falls in [0, 1].
        E.g., if used MinMaxScaler for continuous target (or multi-class target),
        and {0, 1} for binary target.
    """

    def __init__(self, causal_graph, min_hidden_neurons=4, is_linear=False):
        super(CustomNetwork, self).__init__()

        submodels = {}

        for node in causal_graph.nodes:
            # Number of children
            out_degree = causal_graph.out_degree(node)

            if 0 == out_degree:  # Final outputs
                final_output_type = causal_graph.node_types[node]
                # If continuous, explicitly specify it
                if "continuous" == final_output_type:
                    causal_graph.node_types[node] = "continuous_final_output"
                elif "binary" == final_output_type:
                    pass
                else:
                    raise ValueError(
                        f"Unsupported output_type for final outcome: {final_output_type}"
                    )

            # Number of parents
            in_degree = causal_graph.in_degree(node)

            # If the node has no parent, skip the submodel creation
            if in_degree == 0:
                continue

            output_type = causal_graph.node_types[node]
            hidden_dim = max(in_degree, min_hidden_neurons)

            if is_linear:
                submodels[node] = SubNetworkLinear(
                    input_dim=in_degree,
                    output_dim=1,
                    output_type=output_type,
                )
            else:
                submodels[node] = SubNetworkNonlinear(
                    input_dim=in_degree,
                    output_dim=1,
                    hidden_dim=hidden_dim,
                    output_type=output_type,
                )

        self.submodels = nn.ModuleDict(submodels)
        self.causal_graph = causal_graph
        self.is_linear = is_linear

    def forward(self, original_value_dict):
        """
        Forward pass of the CustomNetwork using original values (as the default forward).
        Forward will be called for each SubNetworks. The intermediate predicted values will
        not be propagated.

        It is a good practice to make sure `.to(device)` is called outside forward() in training.

        Input:
        ------
        original_value_dict: Dict[str, torch.Tensor]
            Dictionary containing tensors for each node original values.

        Output:
        -------
        output_dict: Dict[str, torch.Tensor]
            Output of SubNetworks in CustomNetwork with original_value in data set as input.
            The list of nodes only include non-root nodes, i.e., output of some SubNetwork.
        """
        output_dict = {}
        parent_dict = self.causal_graph.parent_dict

        for node, submodel in self.submodels.items():
            parent_nodes = parent_dict[node]
            input_tensor_list = [
                original_value_dict[parent_node] for parent_node in parent_nodes
            ]
            input_tensor = torch.cat(input_tensor_list, dim=-1)
            output_dict[node] = submodel(input_tensor)

        return output_dict

    def inference(
        self,
        original_value_dict,
        feature_standard_scaler_dict=None,
        tail_ref_pt_config=None,
        tail_ref_pt_config_as_dummy=False,
    ):
        """
        Custom forward pass of the CustomNetwork.

        This custom forward function should NOT be used for training.

        Since the values in reference points are used together with values of other variables
        in the data set after normalization (i.e., the use of Scaler before defining DataLoader),
        the scale of reference points should match how variables are normalized.

        NOTE For interpretability, the input reference point value should be at the raw scale.
        Since SubNetworks assume normalized inputs, the scaling of raw reference point input
        is taken care of by this function.

        Although tail_ref_pt_config can contain arrays, unless one would like to assign
        different reference points at certain location to all records in a batch, only use
        a single number to make sure everyone in a batch get the same reference point.
        Out of pre-caution, this way of reference point configuration is not supported.

        Input:
        ------
        original_value_dict: Dict[str, torch.Tensor]
            Dictionary containing tensors for each node original values.

        feature_standard_scaler_dict: CustomStandardScaler
            Custom standard scalers for feature variables, used when scaling intermediate
            categorical outputs of SubNetworks.
            See `utils.data_utils`, `CustomStandardScaler`.

        tail_ref_pt_config: Dict[str, float] or Dict[str, np.NDArray]
            Dictionary containing tail reference point configuration for certain edge.
            See `utils.graph_utils`, `CausalGraph.generate_parent_dict_enhanced()`.

        tail_ref_pt_config_as_dummy: bool
            Indicator of if using tail_ref_pt_config as dummy, i.e., use the inference to
            hierarchically derive the prediction output, but do not use tail_ref_pt values.

        Output:
        -------
        output_dict: Dict[str, torch.Tensor]
            Output of SubNetworks in CustomNetwork when propagating local predictions.
            The value is extracted with priority reference downstream over original value
            from node_data_with_option Dict[str, Dict[str, torch.Tensor]].
        """
        # Set model to evaluation mode, explicitly avoid using inference() for training
        self.eval()

        # Set device to the same as the model parameters
        device = self.parameters().__next__().device

        # Fetch reference point value instantiation options
        parent_dict_enhanced = self.causal_graph.generate_parent_dict_enhanced(
            tail_ref_pt_config
        )

        if True is tail_ref_pt_config_as_dummy:
            tail_ref_pt_config = None  # only used for getting the parent_dict_enhanced

        # Get an example shape of tensor by looking at the first node in parent_dict
        # Every node will have original value tensors from the data set
        node = next(iter(parent_dict_enhanced))
        node_tensor_shape = original_value_dict[node].shape

        # Construct Dict[str, Dict[str, torch.Tensor]]
        node_data_with_option = {}
        for node, node_tensor in original_value_dict.items():
            node_data_with_option[node] = {
                OPTION_ORIGINAL_VALUE: node_tensor,
                # OPTION_REFERENCE_DOWNSTREAM: # leave uninitialized
                # OPTION_REFERENCE_POINT: # leave uninitialized
            }

        # Wrap in torch.no_grad()
        with torch.no_grad():
            for node, submodel in self.submodels.items():
                # Visiting node in the sequence of parent_dict keys, this makes
                # sure that the parent nodes are inferred before the node itself.
                parent_tuples = parent_dict_enhanced[node]
                input_tensor_list = []

                for parent_node, option in parent_tuples:
                    # If tail_ref_pt_config only used as dummy, OPTION_REFERENCE_POINT not activated,
                    # fall back to {OPTION_REFERENCE_DOWNSTREAM, OPTION_ORIGINAL_VALUE}.
                    if True is tail_ref_pt_config_as_dummy:
                        if 0 == self.causal_graph.in_degree(parent_node):
                            option = OPTION_ORIGINAL_VALUE
                        else:
                            option = OPTION_REFERENCE_DOWNSTREAM

                    # Additional steps for reference point option
                    # NOTE Make sure the raw input value is scaled by StandardScaler
                    if option == OPTION_REFERENCE_POINT:
                        edge_key = f"{parent_node}->{node}"

                        # Perform standard scaling
                        edge_tail_value_raw = tail_ref_pt_config.get(
                            edge_key, float("nan")
                        )

                        _tail_node_standard_scaler = (
                            feature_standard_scaler_dict.scaler_dict[parent_node]
                        )
                        edge_tail_value = float(
                            _tail_node_standard_scaler.transform(
                                X=np.array([edge_tail_value_raw]).reshape(-1, 1)
                            )
                        )

                        edge_tail_value = torch.tensor(
                            edge_tail_value,
                            dtype=torch.float32,
                        ).to(device)

                        # If single value for reference point, then the tensor shape is torch.Size([])
                        if torch.Size([]) == edge_tail_value.shape:
                            node_data_with_option[parent_node][option] = (
                                torch.ones(node_tensor_shape).to(device)
                                * edge_tail_value
                            )

                        else:
                            raise ValueError(
                                f"The provided dimension of reference point value is {edge_tail_value.shape}, which is not a single value."
                            )

                    # Option in {OPTION_REFERENCE_DOWNSTREAM, OPTION_ORIGINAL_VALUE}
                    parent_value = node_data_with_option[parent_node].get(
                        option, torch.tensor(float("nan")).to(device)
                    )

                    if torch.isnan(parent_value).any():
                        raise ValueError(
                            f"NaN value encountered in tensor for node {parent_node} with option {option}."
                        )

                    input_tensor_list.append(parent_value)

                # Concatenate parent data to form input tensor
                input_tensor = torch.cat(input_tensor_list, dim=-1)

                # Always generate a copy of inference result and save it to reference downstream
                output_reference_downstream = submodel(input_tensor)
                node_data_with_option[node][
                    OPTION_REFERENCE_DOWNSTREAM
                ] = output_reference_downstream

        # Extract value with priority keys, and compute loss on reference downstream
        # Use extract, since reference downstream not available for root nodes
        output_dict = {}
        for node, value_dict in node_data_with_option.items():
            # Only put nodes that are outputs of some SubNetworks
            if node in self.submodels.keys():
                value_for_option = extract_value_with_priority(
                    value_dict,
                    [
                        OPTION_REFERENCE_DOWNSTREAM,
                        OPTION_ORIGINAL_VALUE,
                    ],
                )
                output_dict[node] = value_for_option

        return output_dict

    def linear_model_coef_penalty_L1(self, edge_linear_coef_constraint_config=None):
        """
        L1 penalty for coefficient related to certain input in SubNetworkLinear.

        The L1 penalty acts as a regularization term to constrain certain linear
        coefficients to be close to 0.

        NOTE This function is ONLY used for illustrative purpose, to demonstrate
        how previous works that focus on the optimization of coefficients to enforce
        fairness along a path (which contains edges).

        Input:
        ------
        edge_linear_coef_constraint_config: Dict[str, bool]
            Dictionary containing whether the linear coefficients along certain edge should
            be constrained to close to 0.
            Example:
            >>> edge_linear_coef_constraint_config = {
                'A->X2': True,
                'X2->X4': False,
                # NOTE there is no complicated parser, operation only perfrom left to right
                '"A->X2" * "X2->X4" + "X4->Y"': True,  # make sure ' is at the outside
                # Use tuple to make sure complicated formulas are calculated correctly (add only)
                ('"A->X2" * "X2->X4" + "X4->Y"', '"A->X3" * "X3->X5" * "X5->Y"'): True
            }.
        """
        if (True != self.is_linear) and (None != edge_linear_coef_constraint_config):
            raise ValueError(
                "Invalid specification. This is ONLY for illustrative purpose with linear models."
            )

        L1_penalty = torch.tensor(
            0.0, device=self.parameters().__next__().device, dtype=torch.float32
        )

        # Operator dictionary
        op_dict = {"+": torch.add, "-": torch.sub, "*": torch.mul, "/": torch.div}

        def get_weight(edge_key):
            tail_node, head_node = edge_key.split("->")
            try:
                input_index = self.causal_graph.parent_dict[head_node].index(tail_node)
                return self.submodels[head_node].linear_layer.weight[:, input_index]
            except:
                # Print a warning if any error occurs
                print(
                    f"Warning: The specified edge '{edge_key}' does not exist in the CausalGraph. Ignoring edge_linear_coef_constraint_config item."
                )
                return torch.tensor(
                    0.0, device=self.parameters().__next__().device, dtype=torch.float32
                )

        if edge_linear_coef_constraint_config is not None:
            for (
                formula,
                should_regularize,
            ) in edge_linear_coef_constraint_config.items():
                if should_regularize:
                    # Check if tuple is used (multiple formula need to add together, before abs)
                    # Since there is no advanced parser, need to calculate correctly even if only
                    # process the formula from left to right.
                    if isinstance(formula, tuple):
                        temp_L1_penalty = torch.tensor(
                            0.0,
                            device=self.parameters().__next__().device,
                            dtype=torch.float32,
                        )
                        for single_formula in formula:
                            # Split the string
                            _parts = single_formula.split()
                            _value = get_weight(_parts[0].strip('"'))
                            for i in range(1, len(_parts), 2):
                                _op = op_dict[_parts[i]]
                                _weight = get_weight(_parts[i + 1].strip('"'))
                                _value = _op(_value, _weight)
                            temp_L1_penalty += _value.sum()  # no abs()
                        L1_penalty += temp_L1_penalty.abs()

                    else:  # just edge_key or single formula
                        # Split the string
                        parts = formula.split()
                        value = get_weight(parts[0].strip('"'))
                        for i in range(1, len(parts), 2):
                            op = op_dict[parts[i]]
                            weight = get_weight(parts[i + 1].strip('"'))
                            value = op(value, weight)
                        L1_penalty += value.abs().sum()

        return L1_penalty

    def set_lambda_linear_model_coef_penalty_L1(self, lambda_linear_L1=1e-4):
        """
        Set lambda value for L1 penalty regularization.
        """
        if True != self.is_linear:
            self.lambda_linear_model_coef_penalty_L1 = 0.0

        self.lambda_linear_model_coef_penalty_L1 = lambda_linear_L1

    def get_lambda_linear_model_coef_penalty_L1(self):
        """
        Get lambda value for L1 penalty regularization.
        """
        if True != self.is_linear:
            self.lambda_linear_model_coef_penalty_L1 = 0.0

        return self.lambda_linear_model_coef_penalty_L1

# utils/misc_utils.py
# Miscellaneous utilities

OPTION_REFERENCE_POINT = "reference_point"
OPTION_REFERENCE_DOWNSTREAM = "reference_downstream"
OPTION_ORIGINAL_VALUE = "original_value"

CUSTOM_ORANGE = (213.0 / 256, 94.0 / 256, 0.0 / 256, 1.0)
CUSTOM_GREEN = (2.0 / 256, 158.0 / 256, 115.0 / 256, 1.0)
CUSTOM_BLUE = (35.0 / 256, 88.0 / 256, 199.0 / 256, 1.0)
CUSTOM_GREY = (0, 0, 0, 0.7)
CUSTOM_LIGHT_YELLOW = (249.0 / 256, 220.0 / 256, 125.0 / 256, 0.1)

from pprint import pprint
import numpy as np
from torch.utils.tensorboard import SummaryWriter


# ... Dictionary operations
def compare_dicts(truth, answer):
    missing_keys = []
    added_keys = []
    content_mismatch = []

    for key in truth:
        if key not in answer:
            missing_keys.append(key)
        elif truth[key] != answer[key]:
            content_mismatch.append(
                f"For {key}, in 'truth' the entry should be {truth[key]},\nbut in 'answer' it is {answer[key]}\n"
            )

    for key in answer:
        if key not in truth:
            added_keys.append(key)

    if missing_keys:
        print("Missing keys in 'answer':", missing_keys)
    if added_keys:
        print("Keys added to 'answer':", added_keys)
    if content_mismatch:
        print("Content mismatch:")
        for mismatch in content_mismatch:
            pprint(mismatch)


def extract_value_with_priority(dictionary, priority_keys):
    for key in priority_keys:
        if key in dictionary:
            return dictionary[key]

    # If empty dict, or dict doesn't contain key in list, or empty list
    return None


# ... Check linear coefficients for linear models
class LinearCoefficientsGroundTruth:
    """
    Get the ground truth linear coefficients from the simulation causal model.
    """

    def __init__(self, sim_model):
        """
        Input sim_model is an instance of class `SimulationCausalModel` in
        `utils/simulation_utils`.
        """
        self.sim_model = sim_model
        self.parent_dict_linear_coef = self.retrieve_coefficients()

        if not self.sim_model.is_linear:
            raise ValueError("Model is not a linear model.")

    def __str__(self):
        result = []
        for node, parent_list in self.parent_dict_linear_coef.items():
            terms = [
                f"{float(coef):+.3f} {input_node}" for input_node, coef in parent_list
            ]
            result.append(f"{node} = {' '.join(terms)} + noise_{node}")
        return "\n".join(result)

    def retrieve_coefficients(self):
        parent_dict_linear_coef = {}

        for node, relation_functions in self.sim_model.relation_functions_dict.items():
            input_nodes = self.sim_model.parent_dict[node]
            node_coefficients = []
            for i, input_node in enumerate(input_nodes):
                coef = relation_functions[i](1)  # use 1 as input to get the coefficient

                node_coefficients.append((input_node, coef))

            parent_dict_linear_coef[node] = node_coefficients

        for node in self.sim_model.nodes:
            if node not in parent_dict_linear_coef:
                # Add root nodes
                parent_dict_linear_coef[node] = []

        return parent_dict_linear_coef


class LinearCoefficientsFittedRescaled:
    """
    Get the fitted linear coefficients and rescale to original scale.

    Need to consider normalization of input and output variables when showing
    fitted parameters in Linear model.

    Can set scalers to `None` to reveal the original parameters in Linear layers.
    """

    def __init__(self, model, feature_standard_scaler_dict, target_minmax_scaler_dict):
        self.model = model
        self.feature_standard_scaler_dict = feature_standard_scaler_dict
        self.target_minmax_scaler_dict = target_minmax_scaler_dict
        self.parent_dict_linear_coef = self.calculate_coefficients()

        if not self.model.is_linear:
            raise ValueError("Model is not a linear model.")

    def __str__(self):
        result = []
        for node, parent_list in self.parent_dict_linear_coef.items():
            if len(parent_list) == 0:
                continue
            terms = [
                f"{float(coef):+.3f} {input_node}"
                for input_node, coef in parent_list[:-1]
            ]
            bias = float(parent_list[-1][-1])
            terms.append(f"{bias:+.3f}")
            result.append(f"{node}_hat = {' '.join(terms)}")
        return "\n".join(result)

    def calculate_coefficients(self):
        parent_dict_linear_coef = {}

        for node in self.model.submodels.keys():
            input_nodes = self.model.causal_graph.parent_dict[node]

            linear_layer = self.model.submodels[node].linear_layer
            weights = linear_layer.weight.data
            biases = linear_layer.bias.data

            node_coefficients = []
            bias = biases[0].item()
            for i, input_node in enumerate(input_nodes):
                coef = weights[0, i].item()

                if self.feature_standard_scaler_dict is not None:
                    # Scaling and offsetting of input variables
                    # --> if input_node is a feature node
                    _scaler = self.feature_standard_scaler_dict.scaler_dict.get(
                        input_node, None
                    )
                    coef /= np.sqrt(_scaler.var_) if _scaler is not None else 1.0
                    _offset = _scaler.mean_ if _scaler is not None else 0.0
                    bias -= coef * _offset

                    # Scaling and offsetting of output variables, if available
                    # --> if node is a feature node
                    _scaler = self.feature_standard_scaler_dict.scaler_dict.get(
                        node, None
                    )
                    _coef = np.sqrt(_scaler.var_) if _scaler is not None else 1.0
                    _offset = _scaler.mean_ if _scaler is not None else 0.0
                    coef *= _coef
                    bias += _offset

                if self.target_minmax_scaler_dict is not None:
                    # Scaling and offsetting of output variables, if available
                    # --> if node is a target node
                    _scaler = self.target_minmax_scaler_dict.scaler_dict.get(node, None)
                    _coef = (
                        (
                            _scaler.data_range_
                            / (_scaler.feature_range[1] - _scaler.feature_range[0])
                        )
                        if _scaler is not None
                        else 1.0
                    )
                    coef *= _coef

                node_coefficients.append((input_node, coef))

            node_coefficients.append(("bias", bias))
            parent_dict_linear_coef[node] = node_coefficients

        for node in self.model.causal_graph.nodes:
            if node not in parent_dict_linear_coef:
                # Add root nodes
                parent_dict_linear_coef[node] = []

        return parent_dict_linear_coef


def compare_linear_coefficients(ground_truth, fitted_rescaled):
    """
    Compared rescaled fitted linear parameters with ground truth, and return
    the signed relative deviation from the ground truth coefficient.

    Ignore bias term, only focus on linear relationship.

    Inputs are instances of LinearCoefficients, and the outputs have the format
    Dict[str, Tuple[str, float]], float, float.
    """
    signed_relative_deviation = {}

    vmin, vmax = float("inf"), float("-inf")

    # The ground_truth does not contain bias terms
    for node, coefficients in ground_truth.parent_dict_linear_coef.items():
        if node not in fitted_rescaled.parent_dict_linear_coef.keys():
            # Root nodes
            signed_relative_deviation[node] = []
            continue

        fitted_rescaled_coefficients = dict(
            fitted_rescaled.parent_dict_linear_coef[node]
        )
        node_difference = []

        for input_node, true_coef in coefficients:
            fit_coef = fitted_rescaled_coefficients[input_node]
            relative_signed_diff = (fit_coef - true_coef) / (abs(true_coef) + 1e-12)
            vmin = min(relative_signed_diff, vmin)
            vmax = max(relative_signed_diff, vmax)
            node_difference.append((input_node, relative_signed_diff))

        signed_relative_deviation[node] = node_difference

    return signed_relative_deviation, vmin, vmax


# ... Custom TensorBoard writer add_graph()
class CustomSummaryWriter(SummaryWriter):
    """
    The custom writer to accomodate different forward functions of CustomNetwork model.

    The writer accepts an additional argument inference_args, which should be a tuple
    containing the arguments for inference(). If use custom forward, a wrapper function
    is defined that takes the input x and calls the inference() method with the provided
    additional arguments. The forward() method of the model is then temporarily replaced with
    this wrapper function.
    """

    def add_graph(
        self,
        model,
        input_to_model=None,
        use_custom_forward=False,
        custom_forward_kwargs=None,
        **kwargs,
    ):
        """
        `custom_forward_kwargs` is a dict to handle additional arguments only for custom forward
        `CustomNetwork.inference()`, as compared to default forward `CustomNetwork.forward()`.

        `**kwargs` is used to handle additional arguments only for `SummaryWriter.add_graph()`.
        """
        if None is custom_forward_kwargs:
            custom_forward_kwargs = {}

        if use_custom_forward:
            # Create a wrapper function for inference with the additional arguments
            def inference_wrapper(x):
                return model.inference(x, **custom_forward_kwargs)

            # Replace model.forward with the wrapper function temporarily
            default_forward = model.forward
            model.forward = inference_wrapper

        super().add_graph(model, input_to_model, **kwargs)

        if use_custom_forward:
            # Restore the original forward method
            model.forward = default_forward

# utils/pipeline_utils.py
# Pipeline related utilities

from collections import defaultdict
import torch
import numpy as np
from simanneal import Annealer


# ... Train the CustomNetwork hierarchically, i.e., train local causal mechanisms one by one
def train_hierarchical(
    model,
    dataloader,
    criterion_dict,
    optimizer_dict,
    edge_linear_coef_constraint_config=None,
):
    """
    Train SubNetworks locally and compute the average training loss per batch in current epoch.
    The loss is with respect to nodes that have parent nodes.

    The function works with both linear and nonlinear networks.

    NOTE Focus on specific submodels. Although the forward pass is shared, optimizer for
    specific SubNetwork will only update parameters in the local module. Notice that this
    is different from using only one optimizer for the CustomNetwork parameter update.

    Train individual subnetworks using the CustomNetwork-level forward pass while
    keeping the gradients and optimizer steps localized to each subnetwork:

    Prepare the input and output tensors for the CustomNetwork-level forward pass.

    For each SubNetwork:
    - Zero the gradients for the optimizer corresponding to the SubNetwork.
    - Perform the CustomNetwork-level forward pass.
    - Extract the output of the specific SubNetwork.
    - Compute the loss on the SubNetwork's output.
    - Call loss.backward() to compute the gradients for the SubNetwork's parameters.
    - Update the SubNetwork's parameters using the optimizer's step method.

    The purpose of using the forward() of CustomNetwork is to avoid re-defining forward()
    in SubNetwork, now that there is a need to concatenate inputs at the CustomNetwork level.

    Input:
    ------
    model: CustomNetwork
        The neural network model to be trained.

    dataloader:
        A DataLoader object providing batches of input data.

    criterion_dict:
        The loss function used to evaluate the submodel's performance, defined on
        specific node.

    optimizer_dict:
        The optimization algorithm used for updating the submodel's parameters.

    edge_linear_coef_constraint_config: Dict[str, bool]
        Dictionary containing whether the linear coefficient along certain edge should
        be constrained to close to 0.
        See `utils.network_utils`, `CustomNetwork.linear_model_coef_penalty_L1()`.

    Output:
    -------
    average_loss_per_node_per_batch: Dict[str, float]
        A dictionary containing the average training loss of nodes for the current epoch.
    """
    model.train()
    device = model.parameters().__next__().device

    active_lambda_linear_L1 = (
        model.get_lambda_linear_model_coef_penalty_L1()
    )  # 0.0 if not linear

    running_loss_dict = defaultdict(float)
    for inputs in dataloader:
        # During training, only original value is used
        inputs = {node: node_tensor.to(device) for node, node_tensor in inputs.items()}

        # Train each SubNetworks separately without explicitly accessing submodel
        for node in model.submodels.keys():
            # Get corresponding optimizer and criterion
            optimizer = optimizer_dict[node]
            criterion = criterion_dict[node]

            # Zero the gradients
            optimizer.zero_grad()

            # Perform forward pass in CustomNetwork using default forward()
            outputs = model(inputs)

            # Extract the output for the specific SubNetwork and compute loss
            loss = criterion(outputs[node], inputs[node])

            # If linear model, and use L1 penalty regularization for illustrative purpose
            loss += active_lambda_linear_L1 * model.linear_model_coef_penalty_L1(
                edge_linear_coef_constraint_config
            )

            # Perform backpropagation
            loss.backward()

            # Update parameters in SubNetwork
            optimizer.step()

            # Update running_loss
            running_loss_dict[node] += loss.item()

    return {
        node: running_loss_sum / len(dataloader)
        for node, running_loss_sum in running_loss_dict.items()
    }


# ... Evaluate the CustomNetwork hierarchically, i.e., outputs of all local causal mechanisms
def evaluate_hierarchical(
    model,
    dataloader,
    criterion_dict,
    feature_standard_scaler_dict,
    forward_func_option="default",
    tail_ref_pt_config=None,
    tail_ref_pt_config_as_dummy=True,
):
    """
    Evaluate SubNetworks locally and compute the average training loss per batch in current epoch.
    The loss is with respect to nodes that have parent nodes.

    The function works with both linear and nonlinear networks.

    NOTE Focus on specific submodels. Notice that this is different from evaluating the
    CustomNetwork as a whole by only looking at the final output node.

    The purpose of using the forward() of CustomNetwork is to avoid re-defining forward()
    in SubNetwork, now that there is a need to concatenate inputs at the CustomNetwork level.

    Input:
    ------
    model: CustomNetwork
        The neural network model to be trained.

    dataloader:
        A DataLoader object providing batches of input data.

    criterion_dict:
        The loss function used to evaluate the submodel's performance, defined on
        specific node.

    optimizer_dict:
        The optimization algorithm used for updating the submodel's parameters.

    feature_standard_scaler_dict: CustomStandardScaler
            Custom standard scalers for feature variables, used by `CustomNetwork.inference()`.

    forward_func_option: str
        The forward function used for forward propagation ('default' or 'inference').
        If the default forward, i.e., `CustomNetwork.forward()`, the evaluation is on locally
        reconstructed nodes with original value as input for each SubNetwork. In other words,
        the prediction is not propagated to downstream local causal modules.
        If the inference forward, i.e., `CustomNetwork.inference()`, the evaluation is on locally
        reconsructed nodes with reference downstream (if available) as input. In other words,
        the prediction is propagated to downstream local causal modules.

    tail_ref_pt_config:
        Dictionary containing tail reference point configuration for certain edge.
        See `utils.graph_utils`, `CausalGraph.generate_parent_dict_enhanced()`.

    tail_ref_pt_config_as_dummy: bool
        Indicator of if using tail_ref_pt_config as dummy.
        See `utils.network_utils`, `CustomNetwork.inference()`.

    Output:
    -------
    average_loss_per_node_per_batch: Dict[str, float]
        A dictionary containing the average training loss of nodes for the current epoch.
    """
    model.eval()
    device = model.parameters().__next__().device

    running_loss_dict = defaultdict(float)
    with torch.no_grad():
        for inputs in dataloader:
            # Take care of .to(device)
            inputs = {
                node: node_tensor.to(device) for node, node_tensor in inputs.items()
            }

            if forward_func_option == "inference":
                # The custom inference()
                outputs = model.inference(
                    inputs,
                    feature_standard_scaler_dict,
                    tail_ref_pt_config=tail_ref_pt_config,
                    tail_ref_pt_config_as_dummy=tail_ref_pt_config_as_dummy,
                )

            else:
                # The default forward()
                outputs = model(inputs)

            # Evaluate each SubNetworks separately without explicitly accessing submodel
            for node, _ in model.submodels.items():
                criterion = criterion_dict[node]
                loss = criterion(outputs[node], inputs[node])
                running_loss_dict[node] += loss.item()

    return {
        node: running_loss_sum / len(dataloader)
        for node, running_loss_sum in running_loss_dict.items()
    }


# ... Simulated annealing to poptimize tail_ref_pt_config to maximize average outcome
class TailRefPtConfigAnnealer(Annealer):
    """
    Simulated annealing to find optimal tail reference points to maximize average outcome.

    NOTE
    - The input to SubNetwork is assumed to be normalized by StandardScaler
    - The custom forward `CustomNetwork.inference()` has the scaling from raw scale to standard scale built-in
    - The model should not have additional fairness constraints imposed already

    In order to make the reference point value interpretable, raw reference point configuration should always be
    in the original scale. One need to scale TailRefPtConfigAnnealer.state with StandardScaler back to the
    original data scale.
    """

    def __init__(
        self,
        state,
        model,
        dataloader,
        node_types,
        output_node,
        feature_standard_scaler_dict,
        feature_minmax_scaler_dict,
    ):
        """
        Initialize the simulated annealing.

        Utilize an additional MinMaxScaler to inverse_transform [0, 1] to the original support of features.
        `CustomNetwork.inference()` treats reference point configurations as raw inputs (for better interpretability).

        NOTE The minmax scaler should use (0, 1) as (min, max) when fitting.
        """
        # The state is tail_ref_pt_config
        super(TailRefPtConfigAnnealer, self).__init__(state)
        self.model = model
        self.dataloader = dataloader
        self.node_types = node_types
        self.device = model.parameters().__next__().device
        self.output_node = output_node

        self.feature_standard_scaler_dict = feature_standard_scaler_dict
        self.feature_minmax_scaler_dict = feature_minmax_scaler_dict

    def move(self):
        """
        NOTE Since self.state is the referenece point value configuration for CustomNetwork,
        make sure the input is at the original data scale.

        For categorical input, need to make sure after moving, it is still a valid input value.

        Use MinMaxScaler to make sure the moving within [0, 1] is mapped back to raw scale (min, max).
        """
        # Randomly select a key from tail_ref_pt_config and update its value
        # by adding a random number from a standard normal distribution.
        edge_key_to_update = np.random.choice(list(self.state.keys()))
        tail_node, _ = edge_key_to_update.split("->")

        # Feature inputs at the raw scale
        ref_pt_before_moving_at_raw_scale = self.state[edge_key_to_update]

        # Bound on [0, 1] scale
        tail_minmax_scaler = self.feature_minmax_scaler_dict.scaler_dict[tail_node]
        ref_pt_before_moving_at_zero_one_scale = tail_minmax_scaler.transform(
            X=np.array(ref_pt_before_moving_at_raw_scale).reshape(-1, 1)
        ).reshape(
            -1,
        )

        ref_pt_after_moving_at_zero_one_scale = -1
        sigma = 2  # adjust the normal distribution standard deviation
        while not 0 <= ref_pt_after_moving_at_zero_one_scale <= 1:
            ref_pt_after_moving_at_zero_one_scale = (
                float(ref_pt_before_moving_at_zero_one_scale)
                + sigma * np.random.randn()
            )

        # From scale [0, 1] to raw scale using inverse_transform of MinMaxScaler
        ref_pt_after_moving_at_raw_scale = tail_minmax_scaler.inverse_transform(
            X=np.array(ref_pt_after_moving_at_zero_one_scale).reshape(-1, 1)
        ).reshape(
            -1,
        )

        if "binary" == self.node_types[tail_node]:
            # Get it to binary
            ref_pt_after_moving_at_raw_scale = np.rint(ref_pt_after_moving_at_raw_scale)

        self.state[edge_key_to_update] = float(ref_pt_after_moving_at_raw_scale)

    def energy(self):
        self.model.eval()
        batch_outcomes = []

        with torch.no_grad():
            for inputs in self.dataloader:
                inputs = {
                    node: node_tensor.to(self.device)
                    for node, node_tensor in inputs.items()
                }

                outputs = self.model.inference(
                    original_value_dict=inputs,
                    feature_standard_scaler_dict=self.feature_standard_scaler_dict,
                    tail_ref_pt_config=self.state,
                    tail_ref_pt_config_as_dummy=False,
                )
                output_node_tensor = outputs[self.output_node].detach()
                batch_outcomes.append(output_node_tensor.mean().item())

            avg_outcome = np.mean(batch_outcomes)

        # Return the negative value for minimization (for the purpose of maximization)
        return -avg_outcome

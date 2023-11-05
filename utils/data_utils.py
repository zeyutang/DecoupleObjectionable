# utils/data_utils.py
# Data related utilities

from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from folktables import ACSDataSource
from folktables.acs import adult_filter


# ... Data processing
class CustomStandardScaler(StandardScaler):
    """
    Custom StandardScaler for numpy_array_dict Dict[str, np.NDArray].

    The transform applies to keys that are not in exclude_keys (list).

    Apply the transform before constructing TensorDataset Dict[str, torch.Tensor].
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scaler_dict = {}
        self.args = args
        self.kwargs = kwargs

    def fit(self, numpy_array_dict, exclude_keys, idx):
        """
        Fit StandardScaler on idx individuals with keys that are not excluded.
        """
        for node, node_array in numpy_array_dict.items():
            if node in exclude_keys:
                continue
            scaler = StandardScaler(*self.args, **self.kwargs)
            scaler.fit(X=node_array[idx].reshape(-1, 1))
            self.scaler_dict[node] = scaler
        return self

    def transform(self, numpy_array_dict, exclude_keys, idx):
        """
        Transform with StandardScaler on idx individuals with keys that are not excluded.
        """
        transformed_numpy_array_dict = {}
        for node, node_array in numpy_array_dict.items():
            # If not transform, just get the corresponding entries.
            if node in exclude_keys:
                transformed_numpy_array_dict[node] = node_array[idx]
                continue

            scaler = self.scaler_dict[node]
            transformed_numpy_array_dict[node] = scaler.transform(
                X=node_array[idx].reshape(-1, 1)
            ).reshape(
                -1,
            )
        return transformed_numpy_array_dict

    def inverse_transform(self, numpy_array_dict, exclude_keys, idx):
        """
        Inverse transform with StandardScaler on idx individuals with keys that are not excluded.
        """
        inverse_transformed_numpy_array_dict = {}
        for node, node_array in numpy_array_dict.items():
            # If not transform, just get the corresponding entries
            if node in exclude_keys:
                inverse_transformed_numpy_array_dict[node] = node_array[idx]
                continue

            scaler = self.scaler_dict[node]
            inverse_transformed_numpy_array_dict[node] = scaler.inverse_transform(
                X=node_array[idx].reshape(-1, 1)
            ).reshape(
                -1,
            )
        return inverse_transformed_numpy_array_dict


class CustomMinMaxScaler(MinMaxScaler):
    """
    Custom MinMaxScaler for numpy_array_dict Dict[str, np.NDArray].

    The transform applies to keys that are in target_keys (list).

    Apply the transform before constructing TensorDataset Dict[str, torch.Tensor].
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scaler_dict = {}
        self.args = args
        self.kwargs = kwargs

    def fit(self, numpy_array_dict, target_keys, idx):
        """
        Fit MinMaxScaler on idx individuals with keys that are targets.
        """
        for node, node_array in numpy_array_dict.items():
            if node not in target_keys:
                continue
            scaler = MinMaxScaler(*self.args, **self.kwargs)
            scaler.fit(X=node_array[idx].reshape(-1, 1))
            self.scaler_dict[node] = scaler
        return self

    def transform(self, numpy_array_dict, target_keys, idx):
        """
        Transform with MinMaxScaler on idx individuals with keys that are targets.
        """
        transformed_numpy_array_dict = {}
        for node, node_array in numpy_array_dict.items():
            # If not transform, just get the corresponding entries
            if node not in target_keys:
                transformed_numpy_array_dict[node] = node_array[idx]
                continue

            scaler = self.scaler_dict[node]
            transformed_numpy_array_dict[node] = scaler.transform(
                X=node_array[idx].reshape(-1, 1)
            ).reshape(
                -1,
            )
        return transformed_numpy_array_dict

    def inverse_transform(self, numpy_array_dict, target_keys, idx):
        """
        Inverse transform with MinMaxScaler on idx individuals with keys that are targets.
        """
        inverse_transformed_numpy_array_dict = {}
        for node, node_array in numpy_array_dict.items():
            # If not transform, just get the corresponding entries
            if node not in target_keys:
                inverse_transformed_numpy_array_dict[node] = node_array[idx]
                continue

            scaler = self.scaler_dict[node]
            inverse_transformed_numpy_array_dict[node] = scaler.inverse_transform(
                X=node_array[idx].reshape(-1, 1)
            ).reshape(
                -1,
            )
        return inverse_transformed_numpy_array_dict


class CustomDataset(Dataset):
    """
    Convert Dict[str, np.NDArray] to Dict[str, torch.Tensor] format.
    """

    def __init__(self, numpy_array_dict):
        self.torch_tensor_dict = {}
        for node, numpy_array in numpy_array_dict.items():
            # np.NDArray use np.float_, which is double (float64) for torch
            self.torch_tensor_dict[node] = torch.unsqueeze(
                torch.tensor(numpy_array, dtype=torch.float), dim=-1
            )  # convert to (n_samples, 1)

    def __len__(self):
        # Get first item, and count n_samples
        node_tensor = self.torch_tensor_dict.get(next(iter(self.torch_tensor_dict)), {})
        return len(node_tensor)

    def __getitem__(self, idx):
        sample = {}
        for node, node_tensor in self.torch_tensor_dict.items():
            sample[node] = node_tensor[idx]
        return sample


def torch_tensor_to_ndarray_dict(torch_tensor_dict):
    """
    Convert Dict[str, torch.Tensor] to Dict[str, np.NDArray] format.
    """
    ndarray_dict = {}
    if isinstance(torch_tensor_dict, CustomDataset):
        _to_enumerate = torch_tensor_dict.torch_tensor_dict
    else:
        _to_enumerate = torch_tensor_dict

    for node, node_tensor in _to_enumerate.items():
        ndarray = node_tensor.cpu().numpy()
        ndarray = np.squeeze(ndarray, axis=-1)
        ndarray_dict[node] = ndarray
    return ndarray_dict


def ndarray_dict_to_ndarray(ndarray_dict, node_list):
    # Arrange arrays according to node_list
    arrays = [ndarray_dict[node].reshape(-1, 1) for node in node_list]

    # Concatenate arrays along columns
    ndarray = np.hstack(arrays)

    return ndarray


# ... Generate simulation data
def generate_data_with_causal_model(
    causal_graph, model, init_data=None, noise=0.1, n_samples=10000, device="cpu"
):
    """
    Generate the simulation data with pre-trained causal models.
    Parameters:
        causal_graph: CausalGraph
        model: CustomNetwork
        init_data: pandas.DataFrame
        n_samples: int
        device: str
    Return:
        data: pandas.DataFrame
    """
    nodes = causal_graph.nodes

    if init_data is None:
        # initialize the data with N(0, 1)
        data = pd.DataFrame(
            np.random.normal(size=(n_samples, len(nodes))), columns=nodes
        )
    else:
        try:
            data = init_data[nodes]
            n_samples = data.shape[0]
        except:
            print(
                f"The columns of init_data {init_data.columns.tolist()} mismatched with the nodes of causal graph {nodes}."
            )
            return None

    for node, parents in causal_graph.parent_dict.items():
        if len(parents) > 0:
            input_data = torch.Tensor(data[parents].values).to(device)
            with torch.no_grad():
                output = model.submodels[node](input_data)

            if device == "cpu":
                output = output.numpy()
            else:
                output = output.cpu().numpy()

            # add random noises
            data[node] = output + np.random.normal(scale=noise, size=output.shape)

    return data


# ... Load real-world data
def load_acs_pums_person_data(
    nodes, *, year=2018, states=["CA"], download=True, addon_feature=None
):
    """
    Load folktables data, which contains recent US census data.
    """
    data_source = ACSDataSource(survey_year=year, horizon="1-Year", survey="person")
    acs_data = data_source.get_data(states=states, download=download)

    # Perform basic filter (mimic the one implemented in UCI Adult)
    acs_data = adult_filter(acs_data)

    # Include add-on features, e.g., ['ADJINC', 'PWGTP']
    if None is not addon_feature:
        include = nodes + list(addon_feature)

    df = acs_data[include]

    data_dict = df.dropna().to_dict(orient="list")

    n_samples = len(data_dict[next(iter(data_dict))])

    for node, value in data_dict.items():
        data_dict[node] = np.array(value).reshape(
            -1,
        )

    return data_dict, n_samples


def load_uci_adult_preprocessed(nodes, *, data_file_path):
    """
    Load UCI Adult data.
    """
    with open(data_file_path, "r") as f:
        df = pd.read_csv(f, sep=",")

    # Included columns
    df = df[nodes]
    data_dict = df.dropna(axis=0).to_dict(orient="list")

    n_samples = len(data_dict[next(iter(data_dict))])

    for node, value in data_dict.items():
        data_dict[node] = np.array(value).reshape(
            -1,
        )

    return data_dict, n_samples

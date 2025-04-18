###############################################
#
# This file severs as an example of how to
# train a MLP model using the SDD dataset using
# the PAL library and a spline-distribution.
#
###############################################

if __name__ == "__main__":
    import os
    import sys

    module_path = os.path.abspath(os.path.join("."))
    if module_path not in sys.path:
        sys.path.append(module_path)

import pal.problem.sdd as csdd
import pal.distribution.spline_distribution as spline
from pal.wmi.compute_integral import integrate_distribution
import torch
import torch.nn as nn
from typing import Callable, Generic, Literal, TypeVar
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import time

T = TypeVar("T")


class SimpleFC(Generic[T], nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: list[int],
        final_function: (
            Callable[[torch.Tensor], tuple[torch.Tensor, ...]] | None
        ) = None,
        final_module: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.fcs = []
        for i in range(len(hidden_sizes)):
            if i == 0:
                self.fcs.append(nn.Linear(input_size, hidden_sizes[i]))
            else:
                self.fcs.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
        self.fcs.append(nn.Linear(hidden_sizes[-1], output_size))
        self.fcs = nn.ModuleList(self.fcs)
        self.final_function = final_function
        self.final_module = final_module

    def network(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(len(self.fcs) - 1):
            x = self.fcs[i](x)
            x = nn.functional.relu(x)
        x = self.fcs[-1](x)
        return x

    def forward(self, x) -> T:
        x = self.network(x)
        if self.final_function is not None:
            x = self.final_function(x)
        if self.final_module is not None:
            x = self.final_module(*x)
        return x

    def __call__(self, *args, **kwds) -> T:
        return super().__call__(*args, **kwds)
    

def create_network(
    input_size: int,
    conditional_spline_dist: spline.ConditionalSplineSQ2D,
    net_size: Literal["small", "medium", "large"],
    init_last_layer_positive: bool,
    device: torch.device,
) -> SimpleFC[spline.SplineSQ2D]:
    shape_value, shape_derivative, shape_mixture_weights = (
        conditional_spline_dist.parameter_shape()
    )

    # create the model
    if net_size == "small":
        hidden_size = [512, 512]
    elif net_size == "medium":
        hidden_size = [1024, 1024]
    elif net_size == "large":
        hidden_size = [2048, 2048]

    total_output_size = (
        np.prod(shape_value)
        + np.prod(shape_derivative)
        + np.prod(shape_mixture_weights)
    )

    param_deriv_dens_scale = 0.1

    num_mixture_param = np.prod(shape_mixture_weights)
    num_dens_knots_values = np.prod(shape_value)

    def reparam(
        out_nn: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        param_mixture_weights = out_nn[:, :num_mixture_param].softmax(dim=-1)

        param_dens_value = out_nn[
            :, num_mixture_param: (num_mixture_param + num_dens_knots_values)
        ]
        param_dens_value = param_dens_value.reshape(-1, *shape_value)

        param_dens_derivative = out_nn[:, (num_mixture_param + num_dens_knots_values):]
        param_dens_derivative = param_deriv_dens_scale * param_dens_derivative.reshape(
            -1, *shape_derivative
        )

        return param_dens_value, param_dens_derivative, param_mixture_weights

    model = SimpleFC[spline.SplineSQ2D](
        input_size=input_size,
        output_size=total_output_size,
        hidden_sizes=hidden_size,
        final_function=reparam,
        final_module=conditional_spline_dist,
    ).to(device)

    if init_last_layer_positive:
        with torch.no_grad():
            last_layer = model.fcs[-1]
            pos_sub = 0.1 * torch.abs(
                last_layer.weight.data[
                    num_mixture_param:(num_mixture_param + num_dens_knots_values)
                ]
            )
            last_layer.weight.data[
                num_mixture_param:(num_mixture_param + num_dens_knots_values)
            ] = pos_sub
            last_layer.bias.data = torch.zeros_like(last_layer.bias.data)

    return model


def get_spline_model(
    sdd: csdd.SDDSingleImageTrajectory,
    num_knots: int,
    num_mixtures: int,
    net_size: Literal["small", "medium", "large"],
    init_last_layer_positive: bool,
    device: torch.device,
) -> SimpleFC[spline.SplineSQ2D]:
    """
    Constructs and returns a spline-based neural network model for trajectory prediction.

    Args:
        sdd (csdd.SDDSingleImageTrajectory): An instance of SDDSingleImageTrajectory
            containing trajectory data and constraints.
        num_knots (int): The number of knots for the spline distribution.
        num_mixtures (int): The number of mixture components for the spline distribution.
        net_size (str): The size of the neural network. Options are "small", "medium", or "large".
        init_last_layer_positive (bool): If True, initializes the last layer of the network to be positive.
        device (torch.device): The device on which the model and computations will be performed (e.g., CPU or GPU).

    Returns:
        SimpleFC[spline.SplineSQ2D]: A fully connected neural network model
            configured to output spline-based trajectory distributions.
    """
    # create the constraints
    lra_problem = sdd.create_constraints()

    input_size = np.prod(sdd.get_x_shape())

    spline_distribution_builder = spline.SplineSQ2DBuilder(
        constraints=lra_problem,
        var_positions=sdd.get_y_vars(),
        num_knots=num_knots,
        num_mixtures=num_mixtures,
    )

    spline_distribution_builder.to(device)

    gasp_kwargs = {
        "batch_size": 256,
    }

    # create the distribution
    start_time = time.time()
    conditional_spline_dist = integrate_distribution(
        d=spline_distribution_builder,
        device=device,
        precision=torch.float64,
        gasp_kwargs=gasp_kwargs,
    )
    end_time = time.time()
    print(f"Time to integrate distribution: {end_time - start_time:.2f} seconds")

    # create the model
    model = create_network(
        input_size=input_size,
        conditional_spline_dist=conditional_spline_dist,
        net_size=net_size,
        init_last_layer_positive=init_last_layer_positive,
        device=device,
    )
    return model


def main(args: argparse.Namespace) -> None:
    # check if random seed is set
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    else:
        args.seed = int(time.time())
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        print(f"Random seed set to {args.seed}")

    sdd = csdd.SDDSingleImageTrajectory(
        img_id=args.img_id,
        path="./data/sdd",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_spline_model(
        sdd=sdd,
        num_knots=args.num_knots,
        num_mixtures=args.num_mixtures,
        net_size=args.net_size,
        init_last_layer_positive=args.init_last_layer_positive,
        device=device,
    )

    # create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("Loading dataset")
    dataset = sdd.load_dataset()
    dataset_train = dataset.train
    dataset_val = dataset.val
    dataset_test = dataset.test
    print(
        f"Train size: {len(dataset_train)}, Val size: {len(dataset_val)}, Test size: {len(dataset_test)}"
    )

    model.to(device)
    precision = torch.float64 if args.use_float64 else torch.float32
    model.to(precision)

    batch_size = args.batch_size

    loader = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=False,
        num_workers=10,
    )

    loader_val = DataLoader(dataset_val, batch_size=batch_size)

    loader_test = DataLoader(dataset_test, batch_size=batch_size)

    epochs = args.epochs

    # Create a unique directory for this run
    run_id = f"run_{int(time.time())}"
    checkpoint_dir = os.path.join("./data/sdd_checkpoints", run_id)
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")

    best_val_ll = float("-inf")

    for epoch in tqdm(range(epochs), desc="Epochs"):
        model.train()
        for i, (x, y) in enumerate(tqdm(loader, desc="Training", leave=False)):
            x = x.to(device).to(precision)
            y = y.to(device).to(precision)

            # forward pass
            log_dens = model(x).log_dens(y)

            loss = -log_dens.mean()
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # validation
        model.eval()
        with torch.no_grad():
            val_ll = []
            for i, (x, y) in enumerate(
                tqdm(loader_val, desc="Validation", leave=False)
            ):
                x = x.to(device).to(precision)
                y = y.to(device).to(precision)

                log_dens = model(x).log_dens(y)
                val_ll.append(log_dens.to("cpu"))
            val_ll = torch.cat(val_ll).mean()
            print(f"Epoch {epoch}: Validation log-likelihood: {val_ll:.4f}")

            # Save the best model
            if val_ll > best_val_ll:
                best_val_ll = val_ll
                torch.save(model.state_dict(), best_model_path)
                print(
                    f"Best model saved with validation log-likelihood: {best_val_ll:.4f}"
                )

    # Load the best model for testing
    model.load_state_dict(torch.load(best_model_path))
    print(f"Best model restored for testing from {best_model_path}.")

    # validation
    with torch.no_grad():
        val_ll = []
        for i, (x, y) in enumerate(tqdm(loader_val, desc="Validation", leave=False)):
            x = x.to(device).to(precision)
            y = y.to(device).to(precision)

            log_dens = model(x).log_dens(y)
            val_ll.append(log_dens.to("cpu"))
        val_ll = torch.cat(val_ll).mean()
        print(f"Validation log-likelihood: {val_ll:.4f} (reference: {best_val_ll:.4f})")

    # test
    with torch.no_grad():
        test_ll = []
        for i, (x, y) in enumerate(tqdm(loader_test, desc="Test", leave=False)):
            x = x.to(device).to(precision)
            y = y.to(device).to(precision)

            log_dens = model(x).log_dens(y)
            test_ll.append(log_dens.to("cpu"))
        test_ll = torch.cat(test_ll).mean()
        print(f"Test log-likelihood: {test_ll:.4f}")


def args():
    parser = argparse.ArgumentParser(
        description="Train a MLP model using the SDD dataset"
    )
    parser.add_argument(
        "--img_id",
        type=int,
        default=12,
        help="Image ID to use for the SDD dataset",
    )
    parser.add_argument(
        "--num_knots",
        type=int,
        default=14,
        help="Number of knots to use for the spline distribution",
    )
    parser.add_argument(
        "--num_mixtures",
        type=int,
        default=8,
        help="Number of mixtures to use for the spline distribution",
    )
    parser.add_argument(
        "--net_size",
        type=str,
        choices=["small", "medium", "large"],
        default="large",
        help="Size of the neural network",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for the optimizer",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of epochs to train for",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--use_float64",
        action="store_true",
        help="Use float64 precision instead of float32",
    )
    parser.add_argument(
        "--init_last_layer_positive",
        action="store_true",
        help="Initialize the last layer of the network to be positive",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = args()
    main(args)

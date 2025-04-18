# flake8: noqa: E501
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import numpy as np
from pal.problem.constrained_problem import ConstrainedProblem
from pal.logic.lra_torch import lra_to_torch
from scipy import ndimage
import torch


blue_red6 = LinearSegmentedColormap.from_list(
    "my_gradient",
    (
        # Edit this gradient at https://eltos.github.io/gradient/#0:FFC94C-20:75EB6E-40.1:2B72DE-44:2900B3-50:FFFFFF-56:C70160-75.2:DE5422-100:471000
        (0.000, (1.000, 0.788, 0.298)),
        (0.200, (0.459, 0.922, 0.431)),
        (0.401, (0.169, 0.447, 0.871)),
        (0.430, (0.161, 0.000, 0.702)),
        (0.500, (1.000, 1.000, 1.000)),
        (0.570, (0.780, 0.004, 0.376)),
        (0.752, (0.871, 0.329, 0.133)),
        (1.000, (0.278, 0.063, 0.000)),
    ),
    N=2048,
)

blue_red5 = LinearSegmentedColormap.from_list(
    "my_gradient",
    (
        # Edit this gradient at https://eltos.github.io/gradient/#0:4CFFC8-35:2B72DE-47:2900B3-50:000000-53:C70160-65:DE5422-100:F9FC4A
        (0.000, (1.000, 0.788, 0.298)),
        (0.200, (0.459, 0.922, 0.431)),
        (0.401, (0.169, 0.447, 0.871)),
        (0.470, (0.161, 0.000, 0.702)),
        (0.500, (1.000, 1.000, 1.000)),
        (0.530, (0.780, 0.004, 0.376)),
        (0.752, (0.871, 0.329, 0.133)),
        (1.000, (0.278, 0.063, 0.000)),
    ),
    N=2048,
)

blue_red5_transparent = LinearSegmentedColormap.from_list(
    "my_gradient",
    (
        # Edit this gradient at https://eltos.github.io/gradient/#0:4CFFC8-35:2B72DE-47:2900B3-50:000000-53:C70160-65:DE5422-100:F9FC4A
        (0.000, (1.000, 0.788, 0.298, 1.0)),
        (0.200, (0.459, 0.922, 0.431, 1.0)),
        (0.401, (0.169, 0.447, 0.871, 1.0)),
        (0.470, (0.161, 0.000, 0.702, 1.0)),
        (0.499, (0.161, 0.000, 0.702, 0.0)),
        (0.500, (1.000, 1.000, 1.000, 0.0)),
        (0.501, (0.780, 0.004, 0.376, 0.0)),
        (0.530, (0.780, 0.004, 0.376, 1.0)),
        (0.752, (0.871, 0.329, 0.133, 1.0)),
        (1.000, (0.278, 0.063, 0.000, 1.0)),
    ),
    N=2048,
)

blue_red6 = LinearSegmentedColormap.from_list(
    "my_gradient",
    (
        # Edit this gradient at https://eltos.github.io/gradient/#0:4CFFC8-35:2B72DE-47:2900B3-50:000000-53:C70160-65:DE5422-100:F9FC4A
        (0.000, (1.000, 0.788, 0.298)),
        (0.200, (0.459, 0.922, 0.431)),
        (0.401, (0.169, 0.447, 0.871)),
        (0.480, (0.161, 0.000, 0.702)),
        (0.500, (1.000, 1.000, 1.000)),
        (0.520, (0.780, 0.004, 0.376)),
        (0.752, (0.871, 0.329, 0.133)),
        (1.000, (0.278, 0.063, 0.000)),
    ),
    N=2048,
)

blue_red6_transapernt = LinearSegmentedColormap.from_list(
    "my_gradient",
    (
        # Edit this gradient at https://eltos.github.io/gradient/#0:4CFFC8-35:2B72DE-47:2900B3-50:000000-53:C70160-65:DE5422-100:F9FC4A
        (0.000, (1.000, 0.788, 0.298, 1.0)),
        (0.200, (0.459, 0.922, 0.431, 1.0)),
        (0.401, (0.169, 0.447, 0.871, 1.0)),
        (0.480, (0.161, 0.000, 0.702, 1.0)),
        (0.499, (0.161, 0.000, 0.702, 0.0)),
        (0.500, (1.000, 1.000, 1.000, 0.0)),
        (0.501, (0.780, 0.004, 0.376, 0.0)),
        (0.520, (0.780, 0.004, 0.376, 1.0)),
        (0.752, (0.871, 0.329, 0.133, 1.0)),
        (1.000, (0.278, 0.063, 0.000, 1.0)),
    ),
    N=2048,
)

blue_red8 = LinearSegmentedColormap.from_list(
    "my_gradient",
    (
        # Edit this gradient at https://eltos.github.io/gradient/#0:FFC94C-20:75EB6E-35:2B72DE-41:2900B3-50:FFFFFF-59:C70160-75.2:DE5422-100:471000
        (0.000, (1.000, 0.788, 0.298)),
        (0.200, (0.459, 0.922, 0.431)),
        (0.350, (0.169, 0.447, 0.871)),
        (0.400, (0.161, 0.000, 0.702)),
        (0.500, (1.000, 1.000, 1.000)),
        (0.600, (0.780, 0.004, 0.376)),
        (0.752, (0.871, 0.329, 0.133)),
        (1.000, (0.278, 0.063, 0.000)),
    ),
    N=2048,
)

blue_red8_transparent = LinearSegmentedColormap.from_list(
    "my_gradient",
    (
        # Edit this gradient at https://eltos.github.io/gradient/#0:FFC94C-20:75EB6E-35:2B72DE-41:2900B3-50:FFFFFF-59:C70160-75.2:DE5422-100:471000
        (0.000, (1.000, 0.788, 0.298, 1.0)),
        (0.200, (0.459, 0.922, 0.431, 1.0)),
        (0.350, (0.169, 0.447, 0.871, 1.0)),
        (0.400, (0.161, 0.000, 0.702, 1.0)),
        (0.499, (0.161, 0.000, 0.702, 0.0)),
        (0.500, (1.000, 1.000, 1.000, 0.0)),
        (0.501, (1.000, 1.000, 1.000, 0.0)),
        (0.600, (0.780, 0.004, 0.376, 1.0)),
        (0.752, (0.871, 0.329, 0.133, 1.0)),
        (1.000, (0.278, 0.063, 0.000, 1.0)),
    ),
    N=2048,
)

the_div_colormap = blue_red8


def prepare_meshgrid(problem: ConstrainedProblem, resolution):
    var_dict = problem.get_y_vars()
    constraints = problem.create_constraints()

    y_pos_dict = {i: name for name, i in var_dict.items()}
    # plot the fitted polynomial
    limits = constraints.get_global_limits()
    linspaces = [
        torch.linspace(limits[y_pos_dict[i]][0], limits[y_pos_dict[i]][1], resolution)
        for i in range(len(var_dict))
    ]

    img_extent = [
        limits[y_pos_dict[0]][0],
        limits[y_pos_dict[0]][1],
        limits[y_pos_dict[1]][0],
        limits[y_pos_dict[1]][1],
    ]

    mesh_grid = torch.meshgrid(*linspaces)
    mesh = torch.stack(mesh_grid, dim=-1).reshape(-1, len(var_dict))

    assert constraints.expression is not None
    pytorch_constraints = lra_to_torch(
        constraints.expression, var_dict=problem.get_y_vars()
    )

    valid = pytorch_constraints(mesh).detach().reshape(resolution, resolution)

    return mesh, mesh_grid, img_extent, valid


def prepare_border(problem: ConstrainedProblem, meshgrid):
    constraints_generic = problem.create_constraints()
    assert constraints_generic.expression is not None
    pytorch_constraints = lra_to_torch(
        constraints_generic.expression, var_dict=problem.get_y_vars()
    )

    var_dict = problem.get_y_vars()

    # check if meshgrid element is torch tensor
    if not isinstance(meshgrid[0], torch.Tensor):
        meshgrid = [torch.tensor(m) for m in meshgrid]
    ys = torch.stack(meshgrid, dim=-1).reshape(-1, len(var_dict))
    valid_ys: torch.Tensor = pytorch_constraints(ys).detach()

    shape_image = valid_ys.reshape(meshgrid[0].shape[0], meshgrid[1].shape[0])

    # Convert the shape image to grayscale if it's a color image
    if shape_image.ndim == 3:
        shape_image = np.mean(shape_image, axis=2)

    # Apply a binary threshold
    binary = shape_image > 0.5

    # Find the edges using binary erosion
    eroded_image = ndimage.binary_erosion(~binary)
    border = binary & ~eroded_image

    return border


def plot_in_paper_diverging_theme(
    the_ax,
    data,
    valid,
    border,
    img_extent,
    norm=None,
    plot_max_value=None,
    the_cmap=the_div_colormap,
    with_contour=False,
    with_valid=False,
    contour_lw=0.5,
    spine_lw=1.5,
    hatched=False,
    plot_borders_below=False,
):
    # plt.figure()
    # f = plt.gcf()
    # the_ax = plt.gca()
    if plot_max_value is None:
        the_max = data.max()
    else:
        the_max = plot_max_value

    if norm is not None:
        the_max = norm.vmax

    if with_contour and plot_borders_below:
        if hatched:
            contour = the_ax.contourf(
                valid.T.to(float),
                levels=[-0.5, 0.5],
                colors="None",
                hatches=["//", None],
                ls=None,
                extent=img_extent,
                origin="upper",
                zorder=0,
            )
            for collection in contour.collections:
                collection.set_edgecolor("pink")
                collection.set_linewidth(0)  # Remove contour lines
        the_ax.contour(
            border,
            colors="pink",
            linewidths=contour_lw,
            extent=img_extent,
            origin="upper",
            zorder=0,
        )
    the_data = data.copy()

    # negative values are valid, positve values are invalid
    # I know this is stupid, but it is what it is
    the_data[valid.T] = (-1) * the_data[valid.T]
    # the_data[~valid] = (-1) * the_data[~valid]

    the_ax.imshow(
        the_data,
        extent=img_extent,
        origin="upper",
        cmap=the_cmap,
        vmin=-the_max,
        vmax=the_max,
        interpolation="nearest",
    )

    if with_valid:
        # plot valid tensor with true being pink and false transparent
        white_cmap = ListedColormap(["white", "none"])

        the_ax.imshow(valid.T, extent=img_extent, origin="upper", cmap=white_cmap)

    if with_contour and not plot_borders_below:
        if hatched:
            contour = the_ax.contourf(
                valid.T.to(float),
                levels=[-0.5, 0.5],
                colors="None",
                hatches=["//", None],
                ls=None,
                extent=img_extent,
                origin="upper",
            )
            for collection in contour.collections:
                collection.set_edgecolor("pink")
                collection.set_linewidth(0)  # Remove contour lines
        the_ax.contour(
            border,
            colors="pink",
            linewidths=contour_lw,
            extent=img_extent,
            origin="upper",
        )

    # black frame, nothing else
    the_ax.set_xticks([])
    the_ax.set_yticks([])
    the_ax.set_xticklabels([])
    the_ax.set_yticklabels([])

    # frame a bit thicker
    for spine in the_ax.spines.values():
        spine.set_linewidth(spine_lw)


plot_in_paper_theme = plot_in_paper_diverging_theme

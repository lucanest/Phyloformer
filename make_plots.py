# Ignore warnings
import warnings

import matplotlib.pyplot as plt
import matplotlib.text as mtext
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Rename some PF methods
RENAMER = {
    "PF_gaps+FastME": "PF_Indel+FastME",
    "PF+FastME": "PF_Base+FastME",
    "PF_MRE+FastME": "PF+FastME",
    "PF_cherry+FastME": "PF_Cherry+FastME",
    "PF_pastek+FastME": "PF_SelReg+FastME",
}

# All possible methods -> corresponding color in the 'tab10' palette
METHODS = [
    "IQTree_LG+GC",  # Blue
    "FastTree",  # Orange
    "FastME",  # Green
    "PF+FastME",  # Red
    "PF_Cherry+FastME",  # Purple
    "IQTree_MF",  # Brown
    "PF_SelReg+FastME",  # Pink
    "Hamming+FastME",  # Gray
    "PF_Base+FastME",  # Khaki
    "PF_Indel+FastME",  # Cyan
    "PF_CPU+FastME",
    # "BioNJ",
]

MARKERS = {
    "IQTree_LG+GC": "o",
    "IQTree_MF": "^",
    "FastME": "X",
    "FastTree": "s",
    "PF+FastME": "o",
    "PF_Cherry+FastME": "s",
    "PF_SelReg+FastME": "X",
    "PF_Indel+FastME": "v",
    "PF_Base+FastME": "^",
    "PF_CPU+FastME": "^",
    "Hamming+FastME": "v",
}

LINESTYLES = {
    "IQTree_LG+GC": "-",
    "IQTree_MF": "-",
    "FastME": ":",
    "FastTree": "-",
    "PF+FastME": "--",
    "PF_Cherry+FastME": "--",
    "PF_SelReg+FastME": "--",
    "PF_Indel+FastME": "--",
    "PF_Base+FastME": "--",
    "PF_CPU+FastME": ":",
    "Hamming+FastME": ":",
}

LGGC_METHODS = sorted(
    [
        "IQTree_LG+GC",
        "FastTree",
        "FastME",
        "Hamming+FastME",
        "PF+FastME",
    ]
)

LGGC_METHODS_NO_HAMMING = sorted(
    [
        "IQTree_LG+GC",
        "FastTree",
        "FastME",
        "PF+FastME",
    ]
)

FINE_TUNE_METHODS = sorted(
    [
        "IQTree_MF",
        "FastTree",
        "FastME",
        "PF+FastME",
    ]
)


TIPS_TICKS = [i for i in range(10, 110, 10)]

# Unified styles
STYLES = {
    k: (col, LINESTYLES[k], MARKERS[k])
    for k, col in zip(METHODS, sns.color_palette("tab10", n_colors=len(METHODS)))
}

# Make both IQTree versions have the same color
STYLES["IQTree_MF"] = (
    STYLES["IQTree_LG+GC"][0],
    STYLES["IQTree_MF"][1],
    STYLES["IQTree_MF"][2],
)
# And LG versions of PF
STYLES["PF_CPU+FastME"] = (
    STYLES["PF+FastME"][0],
    STYLES["PF_CPU+FastME"][1],
    STYLES["PF_CPU+FastME"][2],
)
STYLES["PF_Base+FastME"] = (
    STYLES["PF+FastME"][0],
    STYLES["PF_Base+FastME"][1],
    STYLES["PF_Base+FastME"][2],
)


# To add titles to legends
class LegendTitle(object):
    def __init__(self, text_props=None):
        self.text_props = text_props or {}
        super(LegendTitle, self).__init__()

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        title = mtext.Text(x0, y0, orig_handle, **self.text_props)
        handlebox.add_artist(title)
        return title


def label_axes(axes, fig, uppercase=False):
    """
    Add letter labels in the upper left corner of each subplot
    """

    letters = "abcdefghijklmnopqrstuvwxyz"
    if uppercase:
        letters = letters.upper()
    labels = [f"{l})" for l in letters]

    # Set axis labels
    trans = mtransforms.ScaledTranslation(10 / 72, -5 / 72, fig.dpi_scale_trans)
    ctx = sns.plotting_context()
    sty = sns.axes_style()
    for ax, label in zip(axes, labels):
        ax.text(
            0.0,
            1.0,
            label,
            transform=ax.transAxes + trans,
            fontsize=ctx["axes.titlesize"],
            verticalalignment="top",
            fontfamily=sty["font.family"][0],
            fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="none", pad=3.0),
        )


def group_elapsed(path, rename_pf_cpu: bool = False):
    # Reading and parsing execution metadata
    exec = pd.read_csv(path)

    if rename_pf_cpu:
        exec["marker"] = exec.apply(
            lambda row: (
                "PF_CPU+FastME_outer"
                if row["marker"] == "PF_CPU+FastME" and row["timer"] == "times_outer"
                else row["marker"]
            ),
            axis=1,
        )

    # Sum times and memory for 2 part methods (e.g. PF or Hamming)
    grouped = (
        exec[exec["timer"] != "times_pf_old"]  # Temp
        .groupby(["marker", "id"])[["elapsed_sec", "MaxRSS_kb"]]
        .sum()
        .reset_index()
    )
    grouped["n_tips"] = grouped["id"].apply(lambda x: x.split("_")[1]).astype(int)
    grouped["length"] = grouped["id"].apply(lambda x: x.split("_")[-1]).astype(int)

    return grouped


def plot_line(
    df,
    method,
    ax,
    xvar="n_tips",
    yvar="norm_rf",
    label=True,
    alpha=1.0,
    offset=0.0,
):
    """
    Call sns.lineplot on a particular subset of df and show in on a given Axes object
    """
    col, ls, marker = STYLES[method]
    sub = df[df["marker"] == method]
    sns.lineplot(
        x=sub[xvar],
        y=sub[yvar] + offset,
        color=col,
        ls=ls,
        label=method if label else None,
        alpha=alpha,
        ax=ax,
        marker=marker,
    )


def build_plot(df, methods, metric, label, figsize, xticks, ymajor, yminor):
    """Build the 3 subplots for different alignment lengths on LG+GC"""

    # Setup figure and layout
    fig = plt.figure(layout="constrained", figsize=figsize)
    gs = GridSpec(3, 4, figure=fig, top=0.95, bottom=0.05, right=0.5, left=0.05)
    ax1 = fig.add_subplot(gs[:2, :])
    ax2 = fig.add_subplot(gs[-1, :2], sharey=ax1)
    ax3 = fig.add_subplot(gs[-1, 2:], sharey=ax1)

    for length, ax in zip([500, 250, 1000], [ax1, ax2, ax3]):
        # Plot data to subplot
        for method in methods:
            plot_line(df[df["length"] == length], method, ax, "n_tips", metric)
        ax.set_title(f"Alignment length = {length}")
        ax.set_xlabel("Number of leaves")
        # Supress ylabel of lower right subplot
        if length == 1000:
            ax.set_ylabel(None)
        else:
            ax.set_ylabel(label)

    plt.setp(ax3.get_yticklabels(), visible=False)

    # Custom ticks for top subplot
    if xticks is not None:
        ax1.set_xticks(xticks)
    if ymajor is not None:
        ax1.set_yticks(ymajor)
    if yminor is not None:
        ax1.set_yticks(yminor, minor=True)
    ax1.minorticks_on()
    ax1.grid(which="major", ls="-", linewidth=1.5, color="white")
    ax1.grid(which="minor", ls="-", linewidth=0.5, color="white", axis="y")

    # Add labels to sunplots
    label_axes([ax1, ax2, ax3], fig)

    # Add legends
    ax1.legend(
        loc="upper left",
        title="Tree inference\nmethod",
        bbox_to_anchor=(1, 1),
    )
    ax2.get_legend().remove()
    ax3.get_legend().remove()

    return fig


def side_by_side(df, methods, metric, label, figsize):
    fig = plt.figure(layout="constrained", figsize=figsize)
    gs = GridSpec(4, 9, figure=fig, top=0.95, bottom=0.05, right=0.5, left=0.05)
    ax1 = fig.add_subplot(gs[:3, :3])
    ax2 = fig.add_subplot(gs[:3, 3:6], sharey=ax1)
    ax3 = fig.add_subplot(gs[:3, -3:], sharey=ax1)
    ax_legend = fig.add_subplot(gs[-1, :])

    axes = [ax1, ax2, ax3]

    for length, ax in zip([250, 500, 1000], axes):
        # Plot data to subplot
        for method in methods:
            plot_line(df[df["length"] == length], method, ax, "n_tips", metric)
        ax.set_title(f"Alignment length = {length}")
        ax.set_xlabel("Number of leaves")

        # Set Y axis labels
        if length == 250:
            ax.set_ylabel(label)
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.set_ylabel("")

    h, l = ax1.get_legend_handles_labels()
    for ax in axes:
        ax.get_legend().remove()

    ax_legend.set_axis_off()
    ax_legend.set_ylabel(None)
    ax_legend.set_xlabel(None)
    ax_legend.legend(h, l, loc="center", bbox_to_anchor=(0.5, 0.5), ncol=4)

    return fig


def build_LGGC_normRF(df, figsize):
    label = "Normalized Robinson-Foulds distance"
    ticks = [float(f"0.{i:02}") for i in range(21)]
    major = [0, 0.05, 0.1, 0.15, 0.2]
    minor = [x for x in ticks if x not in major]

    return side_by_side(df, LGGC_METHODS_NO_HAMMING, "norm_rf", label, figsize)

    return build_plot(
        df, LGGC_METHODS_NO_HAMMING, "norm_rf", label, figsize, TIPS_TICKS, major, minor
    )


def build_LGGC_KFscore(df, figsize):
    label = "Kuhner-Felsenstein distance"
    ticks = [round(i / 10, 1) for i in range(35)]
    major = [0, 0.5, 1, 1.5, 2, 2.5, 3]
    minor = [x for x in ticks if x not in major]

    return side_by_side(df, LGGC_METHODS_NO_HAMMING, "kf_score", label, figsize)

    return build_plot(
        df,
        LGGC_METHODS_NO_HAMMING,
        "kf_score",
        label,
        figsize,
        TIPS_TICKS,
        major,
        minor,
    )


def build_LGGC_wRF(df, figsize):
    label = "weighted Robinson-Foulds distance"
    ticks = list(range(16))
    major = list(range(0, 16, 5))
    minor = [x for x in ticks if x not in major]

    return side_by_side(df, LGGC_METHODS_NO_HAMMING, "weighted_rf", label, figsize)

    return build_plot(
        df,
        LGGC_METHODS_NO_HAMMING,
        "weighted_rf",
        label,
        figsize,
        TIPS_TICKS,
        major,
        minor,
    )


def build_LGGC_lik(df, figsize):
    label = "Log-likelihood Ratio"
    fig = side_by_side(df, LGGC_METHODS_NO_HAMMING, "ratio", label, figsize)

    # fig = build_plot(
    #     df, LGGC_METHODS_NO_HAMMING, "ratio", label, figsize, TIPS_TICKS, None, None
    # )

    for ax in fig.axes[:-1]:
        ax.axhline(y=1, ls=":", color="gray")

    return fig


def single_LGGC_normRF(df, figsize):

    sub = df[df["length"] == 500]

    # Single RF plot for LG+GC 500 AAs
    fig, ax = plt.subplots(1, figsize=figsize, layout="constrained")
    for method in LGGC_METHODS_NO_HAMMING:
        plot_line(sub, method, ax, "n_tips", "norm_rf")
    ax.set_xlabel("Number of leaves")
    ax.set_ylabel("Normalized Robinson-Foulds distance")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=3,
    )

    return fig


def single_LGGC_KFscore(df, figsize):
    sub = df[df["length"] == 500]

    # Single RF plot for LG+GC 500 AAs
    fig, ax = plt.subplots(1, figsize=figsize, layout="constrained")
    for method in LGGC_METHODS_NO_HAMMING:
        plot_line(sub, method, ax, "n_tips", "kf_score")
    ax.set_xlabel("Number of leaves")
    ax.set_ylabel("Kuhner-Felsenstein distance")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=3,
    )

    return fig


def single_LGGC_wRF(df, figsize):
    sub = df[df["length"] == 500]

    # Single RF plot for LG+GC 500 AAs
    fig, ax = plt.subplots(1, figsize=figsize, layout="constrained")
    for method in LGGC_METHODS_NO_HAMMING:
        plot_line(sub, method, ax, "n_tips", "weighted_rf")
    ax.set_xlabel("Number of leaves")
    ax.set_ylabel("weighted Robinson-Foulds distance")

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=3,
    )

    return fig


def single_LGGC_mae(df, figsize):
    sub = df[df["length"] == 500]

    # Single RF plot for LG+GC 500 AAs
    fig, ax = plt.subplots(1, figsize=figsize, layout="constrained")
    for method in LGGC_METHODS_NO_HAMMING:
        plot_line(sub, method, ax, "n_tips", "MAE")
    ax.set_xlabel("Number of leaves")
    ax.set_ylabel("Mean Absolute Error on pairwise distance")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=3,
    )

    return fig


def single_LGGC_mre(df, figsize):
    sub = df[df["length"] == 500]

    # Single RF plot for LG+GC 500 AAs
    fig, ax = plt.subplots(1, figsize=figsize, layout="constrained")
    for method in LGGC_METHODS_NO_HAMMING:
        plot_line(sub, method, ax, "n_tips", "MRE")
    ax.set_xlabel("Number of leaves")
    ax.set_ylabel("Mean Relative Error on pairwise distance")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
    )

    return fig


def single_LGGC_mrd(df, figsize):
    sub = df[df["length"] == 500]

    # Single RF plot for LG+GC 500 AAs
    fig, ax = plt.subplots(1, figsize=figsize, layout="constrained")
    for method in LGGC_METHODS_NO_HAMMING:
        plot_line(sub, method, ax, "n_tips", "MRD")
    ax.set_xlabel("Number of leaves")
    ax.set_ylabel("Mean Relative Difference on pairwise distance")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
    )

    return fig


def single_LGGC_quantiles_mae(sub, figsize):

    # Single RF plot for LG+GC 500 AAs
    fig, ax = plt.subplots(1, figsize=figsize, layout="constrained")
    for method in sorted(LGGC_METHODS_NO_HAMMING + ["PF_Base+FastME"]):
        plot_line(sub, method, ax, "percentile", "MAE")
    ax.set_xlabel("Reference Pairwise Distance Percentile")
    ax.set_ylabel("Mean Absolute Error on pairwise distance")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
    )
    ax.set_yscale("log")
    ax.set_xscale("log")

    return fig


def single_LGGC_quantiles_mre(sub, figsize):

    # Single RF plot for LG+GC 500 AAs
    fig, ax = plt.subplots(1, figsize=figsize, layout="constrained")
    for method in sorted(LGGC_METHODS_NO_HAMMING + ["PF_Base+FastME"]):
        plot_line(sub, method, ax, "percentile", "MRE")
    ax.set_xlabel("Reference Pairwise Distance Percentile")
    ax.set_ylabel("Mean Relative Error on pairwise distance")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
    )
    ax.set_xscale("log")

    return fig


def single_LGGC_quantiles_mrd(sub, figsize):

    # Single RF plot for LG+GC 500 AAs
    fig, ax = plt.subplots(1, figsize=figsize, layout="constrained")
    for method in sorted(LGGC_METHODS_NO_HAMMING + ["PF_Base+FastME"]):
        plot_line(sub, method, ax, "percentile", "MRD")
    ax.set_xlabel("Reference Pairwise Distance Percentile")
    ax.set_ylabel("Mean Relative Difference on pairwise distance")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
    )
    ax.set_xscale("log")

    return fig


def single_LGGC_binned_mae(sub, figsize):

    # Single RF plot for LG+GC 500 AAs
    fig, ax = plt.subplots(1, figsize=figsize, layout="constrained")
    for method in sorted(LGGC_METHODS_NO_HAMMING + ["PF_Base+FastME"]):
        plot_line(sub, method, ax, "binned", "MAE")
    ax.set_xlabel("Binned Reference Pairwise Distance")
    ax.set_ylabel("Mean Absolute Error on pairwise distance")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
    )
    ax.set_yscale("log")
    ax.set_xscale("log")

    return fig


def single_LGGC_binned_mre(sub, figsize):

    # Single RF plot for LG+GC 500 AAs
    fig, ax = plt.subplots(1, figsize=figsize, layout="constrained")
    for method in sorted(LGGC_METHODS_NO_HAMMING + ["PF_Base+FastME"]):
        plot_line(sub, method, ax, "binned", "MRE")
    ax.set_xlabel("Binned Reference Pairwise Distance Percentile")
    ax.set_ylabel("Mean Relative Error on pairwise distance")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
    )
    ax.set_xscale("log")

    return fig


def single_LGGC_binned_mrd(sub, figsize):

    # Single RF plot for LG+GC 500 AAs
    fig, ax = plt.subplots(1, figsize=figsize, layout="constrained")
    for method in sorted(LGGC_METHODS_NO_HAMMING + ["PF_Base+FastME"]):
        plot_line(sub, method, ax, "binned", "MRD")
    ax.set_xlabel("Binned Reference Pairwise Distance Percentile")
    ax.set_ylabel("Mean Relative Difference on pairwise distance")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
    )
    ax.set_xscale("log")

    return fig


def single_LGGC_elapsed(df, figsize, model_load_time=None):
    sub = df[df["length"] == 500]

    # Single RF plot for LG+GC 500 AAs
    fig, ax = plt.subplots(1, figsize=figsize, layout="constrained")
    if model_load_time is not None:
        plot_line(
            sub,
            "PF+FastME",
            ax,
            "n_tips",
            "elapsed_sec",
            offset=model_load_time,
            alpha=0.2,
            label=False,
        )
    for method in LGGC_METHODS_NO_HAMMING:
        plot_line(sub, method, ax, "n_tips", "elapsed_sec")

    ax.set_yscale("log")
    ax.grid(
        which="minor",
        ls=sns.axes_style()["grid.linestyle"],
        linewidth=0.4 * sns.plotting_context()["grid.linewidth"],
        color=sns.axes_style()["grid.color"],
        axis="y",
    )

    ax.set_xlabel("Number of leaves")
    ax.set_ylabel("Elapsed time (sec)")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=3,
    )

    return fig


def paper_elapsed(df, figsize):

    # Single RF plot for LG+GC 500 AAs
    fig, ax = plt.subplots(1, figsize=figsize, layout="constrained")
    for method in sorted(LGGC_METHODS_NO_HAMMING + ["IQTree_MF"]):
        plot_line(df, method, ax, "n_tips", "elapsed_sec")

    ax.set_yscale("log")
    ax.grid(
        which="minor",
        ls=sns.axes_style()["grid.linestyle"],
        linewidth=0.4 * sns.plotting_context()["grid.linewidth"],
        color=sns.axes_style()["grid.color"],
        axis="y",
    )

    ax.set_xlabel("Number of leaves")
    ax.set_ylabel("Elapsed time (sec)")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=3,
    )

    return fig


def single_LGGC_mem(df, figsize):
    sub = df[df["length"] == 500]

    # Single RF plot for LG+GC 500 AAs
    fig, ax = plt.subplots(1, figsize=figsize, layout="constrained")
    for method in LGGC_METHODS_NO_HAMMING:
        plot_line(sub, method, ax, "n_tips", "MaxRSS_kb")

    ax.set_yscale("log")
    ax.grid(
        which="minor",
        ls=sns.axes_style()["grid.linestyle"],
        linewidth=0.4 * sns.plotting_context()["grid.linewidth"],
        color=sns.axes_style()["grid.color"],
        axis="y",
    )

    ax.set_xlabel("Number of leaves")
    ax.set_ylabel("Maximum RSS (kB)")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=3,
    )

    return fig


def single_LGGC_lik(df, figsize):

    sub = df[df["length"] == 500]

    # Single RF plot for LG+GC 500 AAs
    fig, ax = plt.subplots(1, figsize=figsize, layout="constrained")
    for method in LGGC_METHODS_NO_HAMMING:
        plot_line(sub, method, ax, "n_tips", "ratio")
    ax.set_xlabel("Number of leaves")
    ax.set_ylabel("Log-likelihood Ratio")
    ax.axhline(y=1, ls=":", color="gray", label="True Tree")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=3,
    )

    return fig


def cherry_pastek_plots(
    cherry_df,
    pastek_df,
    figsize,
    yvar,
    ylabel,
    log=False,
    include_PF=True,
    sharey=False,
):
    handles = []
    fig = plt.figure(layout="constrained", figsize=figsize)
    gs = GridSpec(7, 6, figure=fig)
    kw = dict()
    ax_cherry = fig.add_subplot(gs[:6, :3])
    if sharey:
        kw["sharey"] = ax_cherry
    ax_pastek = fig.add_subplot(gs[:6, -3:], **kw)
    ax_dummy = fig.add_subplot(gs[-1, :])

    ms = sns.plotting_context()["lines.markersize"]
    marker_kwargs = dict(
        markersize=ms,
        markeredgecolor="w",
        markeredgewidth=0.75,
    )

    methods = FINE_TUNE_METHODS if include_PF else FINE_TUNE_METHODS[:-1]

    # Cherry plot
    for method in methods + ["PF_Cherry+FastME"]:
        plot_line(cherry_df, method, ax_cherry, "n_tips", yvar)
        col, ls, marker = STYLES[method]
        col, ls, marker = STYLES[method]
        handles.append(
            Line2D(
                [0], [0], color=col, ls=ls, label=method, marker=marker, **marker_kwargs
            )
        )

    ax_cherry.set_title("Cherry")

    # Pastek plot
    for method in methods + ["PF_SelReg+FastME"]:
        plot_line(pastek_df, method, ax_pastek, "n_tips", yvar)
    col, ls, marker = STYLES["PF_SelReg+FastME"]
    handles.append(
        Line2D(
            [0],
            [0],
            color=col,
            ls=ls,
            label="PF_SelReg+FastME",
            marker=marker,
            **marker_kwargs,
        )
    )
    ax_pastek.set_title("SelReg")

    # Set log axis if needed
    if log:
        for ax in [ax_cherry, ax_pastek]:
            ax.set_yscale("log")
            ax.grid(
                which="minor",
                ls=sns.axes_style()["grid.linestyle"],
                linewidth=0.4 * sns.plotting_context()["grid.linewidth"],
                color=sns.axes_style()["grid.color"],
                axis="y",
            )

    # Add subplot label
    label_axes([ax_cherry, ax_pastek], fig)

    # Set axis labels
    ax_cherry.set_ylabel(ylabel)
    for ax in [ax_cherry, ax_pastek]:
        ax.set_xlabel("Number of leaves")
        ax.get_legend().remove()

    ax_pastek.set_ylabel(None)
    if sharey:
        plt.setp(ax_pastek.get_yticklabels(), visible=False)

    ax_dummy.set_axis_off()
    ax_dummy.set_ylabel(None)
    ax_dummy.set_xlabel(None)
    ax_dummy.legend(handles=handles, loc="center", ncol=3)

    return fig


def fine_tuned_plot(
    gaps_df,
    cherry_df,
    pastek_df,
    figsize,
    yvar,
    ylabel,
    log=False,
    include_PF=True,
    exclude_pf_gaps=True,
    sharey=False,
):
    handles = []
    fig = plt.figure(layout="constrained", figsize=figsize)
    gs = GridSpec(7, 6, figure=fig)
    ax_gaps = fig.add_subplot(gs[:6, :2])
    kw = dict()
    if sharey:
        kw["sharey"] = ax_gaps
    ax_cherry = fig.add_subplot(gs[:6, 2:-2], **kw)
    ax_pastek = fig.add_subplot(gs[:6, -2:], **kw)
    ax_dummy = fig.add_subplot(gs[-1, :])

    ms = sns.plotting_context()["lines.markersize"]
    marker_kwargs = dict(
        markersize=ms,
        markeredgecolor="w",
        markeredgewidth=0.75,
    )

    methods = FINE_TUNE_METHODS if include_PF else FINE_TUNE_METHODS[:-1]

    # Gaps plot
    for method in methods + ["PF_Indel+FastME"]:
        if exclude_pf_gaps and method == "PF+FastME":
            continue
        plot_line(gaps_df, method, ax_gaps, "n_tips", yvar)
        col, ls, marker = STYLES[method]
        handles.append(
            Line2D(
                [0], [0], color=col, ls=ls, label=method, marker=marker, **marker_kwargs
            )
        )
    ax_gaps.set_title("LG+GC+Gaps")

    # Cherry plot
    for method in methods + ["PF_Cherry+FastME"]:
        plot_line(cherry_df, method, ax_cherry, "n_tips", yvar)
        col, ls, marker = STYLES[method]
    col, ls, marker = STYLES["PF_Cherry+FastME"]
    handles.append(
        Line2D(
            [0],
            [0],
            color=col,
            ls=ls,
            label="PF_SelReg+FastME",
            marker=marker,
            **marker_kwargs,
        )
    )
    ax_cherry.set_title("Cherry")

    # Pastek plot
    for method in methods + ["PF_SelReg+FastME"]:
        plot_line(pastek_df, method, ax_pastek, "n_tips", yvar)
    col, ls, marker = STYLES["PF_SelReg+FastME"]
    handles.append(
        Line2D(
            [0],
            [0],
            color=col,
            ls=ls,
            label="PF_SelReg+FastME",
            marker=marker,
            **marker_kwargs,
        )
    )
    ax_pastek.set_title("SelReg")

    # Set log axis if needed
    if log:
        for ax in [ax_gaps, ax_cherry, ax_pastek]:
            ax.set_yscale("log")
            ax.grid(
                which="minor",
                ls=sns.axes_style()["grid.linestyle"],
                linewidth=0.4 * sns.plotting_context()["grid.linewidth"],
                color=sns.axes_style()["grid.color"],
                axis="y",
            )

    # Add subplot labels
    label_axes([ax_gaps, ax_cherry, ax_pastek], fig)

    # Set axis labels
    ax_gaps.set_ylabel(ylabel)
    for ax in [ax_gaps, ax_cherry, ax_pastek]:
        ax.set_xlabel("Number of leaves")
        ax.get_legend().remove()

    ax_cherry.set_ylabel(None)
    ax_pastek.set_ylabel(None)
    if sharey:
        plt.setp(ax_cherry.get_yticklabels(), visible=False)
        plt.setp(ax_pastek.get_yticklabels(), visible=False)

    ax_dummy.set_axis_off()
    ax_dummy.set_ylabel(None)
    ax_dummy.set_xlabel(None)
    ax_dummy.legend(handles=handles, loc="center", ncol=3)

    return fig


def fine_tuned_normRF(gaps_df, cherry_df, pastek_df, figsize):
    return fine_tuned_plot(
        gaps_df,
        cherry_df,
        pastek_df,
        figsize,
        "norm_rf",
        "Normalized Robinson-Foulds distance",
    )


def cherry_pastek_normRF(cherry_df, pastek_df, figsize):
    return cherry_pastek_plots(
        cherry_df,
        pastek_df,
        figsize,
        "norm_rf",
        "Normalized Robinson-Foulds distance",
        sharey=True,
    )


def fine_tuned_KFscore(gaps_df, cherry_df, pastek_df, figsize):
    return fine_tuned_plot(
        gaps_df,
        cherry_df,
        pastek_df,
        figsize,
        "kf_score",
        "Kuhner-Felsenstein distance",
    )


def cherry_pastek_KFscore(cherry_df, pastek_df, figsize):
    return cherry_pastek_plots(
        cherry_df,
        pastek_df,
        figsize,
        "kf_score",
        "Kuhner-Felsenstein distance",
        sharey=True,
    )


def fine_tuned_wRF(gaps_df, cherry_df, pastek_df, figsize):
    return fine_tuned_plot(
        gaps_df,
        cherry_df,
        pastek_df,
        figsize,
        "weighted_rf",
        "weighted Robinson-Foulds distance",
    )


def cherry_pastek_wRF(cherry_df, pastek_df, figsize):
    return cherry_pastek_plots(
        cherry_df,
        pastek_df,
        figsize,
        "weighted_rf",
        "weighted Robinson-Foulds distance",
        sharey=True,
    )


def fine_tuned_elapsed(gaps_df, cherry_df, pastek_df, figsize):
    return fine_tuned_plot(
        gaps_df,
        cherry_df,
        pastek_df,
        figsize,
        "elapsed_sec",
        "Elapsed time (sec)",
        log=True,
        include_PF=False,
        sharey=True,
    )


def fine_tuned_mem(gaps_df, cherry_df, pastek_df, figsize):
    return fine_tuned_plot(
        gaps_df,
        cherry_df,
        pastek_df,
        figsize,
        "MaxRSS_kb",
        "Maximum RSS (kB)",
        log=True,
        include_PF=False,
        sharey=True,
    )


def fine_tuned_mae(gaps_df, cherry_df, pastek_df, figsize):
    return fine_tuned_plot(gaps_df, cherry_df, pastek_df, figsize, "MAE", "MAE")


def cherry_pastek_topos(
    cherry_df,
    pastek_df,
    figsize,
    include_PF=False,
):
    # Setup figure and layout
    fig = plt.figure(layout="constrained", figsize=figsize)
    gs = GridSpec(7, 6, figure=fig, top=0.95, bottom=0.05, right=0.5, left=0.05)

    rfc_ax = fig.add_subplot(gs[:3, :3])
    rfp_ax = fig.add_subplot(gs[:3, -3:], sharey=rfc_ax)
    kfc_ax = fig.add_subplot(gs[3:-1, :3])
    kfp_ax = fig.add_subplot(gs[3:-1, -3:], sharey=kfc_ax)
    legend_ax = fig.add_subplot(gs[-1, :])

    handles = []

    ms = sns.plotting_context()["lines.markersize"]
    marker_kwargs = dict(
        markersize=ms,
        markeredgecolor="w",
        markeredgewidth=0.75,
    )

    methods = FINE_TUNE_METHODS if include_PF else FINE_TUNE_METHODS[:-1]

    # Cherry plot
    for method in methods + ["PF_Cherry+FastME"]:
        plot_line(cherry_df, method, rfc_ax, "n_tips", "norm_rf")
        plot_line(cherry_df, method, kfc_ax, "n_tips", "kf_score")
        col, ls, marker = STYLES[method]
        col, ls, marker = STYLES[method]
        handles.append(
            Line2D(
                [0], [0], color=col, ls=ls, label=method, marker=marker, **marker_kwargs
            )
        )

    rfc_ax.set_title("Cherry")

    # Pastek plot
    for method in methods + ["PF_SelReg+FastME"]:
        plot_line(pastek_df, method, rfp_ax, "n_tips", "norm_rf")
        plot_line(pastek_df, method, kfp_ax, "n_tips", "kf_score")

    col, ls, marker = STYLES["PF_SelReg+FastME"]
    handles.append(
        Line2D(
            [0],
            [0],
            color=col,
            ls=ls,
            label="PF_SelReg+FastME",
            marker=marker,
            **marker_kwargs,
        )
    )
    rfp_ax.set_title("SelReg")

    # Add subplot label
    label_axes([rfc_ax, rfp_ax, kfc_ax, kfp_ax], fig)

    # Remove legends
    for ax in [kfc_ax, kfp_ax, rfc_ax, rfp_ax]:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.get_legend().remove()

    # Set X axis labels
    for ax in [kfc_ax, kfp_ax]:
        ax.set_title("")
        ax.set_xlabel("Number of leaves")

    # Set Y axis labels
    rfc_ax.set_ylabel("Normalized Robinson-Foulds distance")
    kfc_ax.set_ylabel("Kuhner-Felsenstein distance")
    for ax in [rfp_ax, kfp_ax]:
        ax.set_ylabel("")
        plt.setp(ax.get_yticklabels(), visible=False)

    legend_ax.set_axis_off()
    legend_ax.set_ylabel("")
    legend_ax.set_xlabel("")
    legend_ax.legend(handles=handles, loc="center", ncol=3)
    return fig


def base_vs_ft(
    topo_df,
    dists_df,
    figsize,
):
    # Setup figure and layout
    fig = plt.figure(layout="constrained", figsize=figsize)
    gs = GridSpec(7, 6, figure=fig, top=0.95, bottom=0.05, right=0.5, left=0.05)

    kf_ax = fig.add_subplot(gs[:3, :3])
    rf_ax = fig.add_subplot(gs[:3, -3:])
    mae_ax = fig.add_subplot(gs[3:-1, :3])
    mre_ax = fig.add_subplot(gs[3:-1, -3:])
    legend_ax = fig.add_subplot(gs[-1, :])

    methods = ["PF+FastME", "PF_Base+FastME"]
    kws = dict(x="n_tips", hue="marker", hue_order=methods, marker="o")

    # Plot figures
    sns.lineplot(data=topo_df, y="kf_score", ax=kf_ax, **kws)
    sns.lineplot(data=topo_df, y="norm_rf", ax=rf_ax, **kws)
    sns.lineplot(data=dists_df, y="MAE", ax=mae_ax, **kws)
    sns.lineplot(data=dists_df, y="MRE", ax=mre_ax, **kws)

    handles, labels = mre_ax.get_legend_handles_labels()

    # Add subplot label
    label_axes([kf_ax, rf_ax, mae_ax, mre_ax], fig)

    # Remove legends
    for ax in [mae_ax, mre_ax, kf_ax, rf_ax]:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.get_legend().remove()

    # Set X axis labels
    for ax in [mae_ax, mre_ax]:
        ax.set_title("")
        ax.set_xlabel("Number of leaves")

    # Set Y axis labels
    rf_ax.set_ylabel("Normalized Robinson-Foulds distance")
    kf_ax.set_ylabel("Kuhner-Felsenstein distance")
    mae_ax.set_ylabel("Mean Absolute Error")
    mre_ax.set_ylabel("Mean Relative Error")

    for ax in [rf_ax, kf_ax]:
        ax.set_xlabel("")
        plt.setp(ax.get_xticklabels(), visible=False)

    legend_ax.set_axis_off()
    legend_ax.set_ylabel("")
    legend_ax.set_xlabel("")
    legend_ax.legend(handles, labels, loc="center", ncol=2)
    return fig


def dataset_plot(df, mae_df, figsize, methods):
    # Setup figure and layout
    fig = plt.figure(layout="constrained", figsize=figsize)
    gs = GridSpec(4, 5, figure=fig, top=0.95, bottom=0.05, right=0.5, left=0.05)

    kf_ax = fig.add_subplot(gs[:3, :3])
    mae_ax = fig.add_subplot(gs[:2, -2:])
    rf_ax = fig.add_subplot(gs[-2:, -2:], sharex=mae_ax)
    legend_ax = fig.add_subplot(gs[-1, :3])

    # Plot lines
    for method in methods:
        plot_line(df, method, kf_ax, yvar="kf_score")
        plot_line(df, method, rf_ax, yvar="norm_rf")
        plot_line(mae_df, method, mae_ax, yvar="MAE")

    # Add legend
    legend_ax.set_axis_off()
    legend_ax.set_ylabel("")
    legend_ax.set_xlabel("")
    handles, labels = kf_ax.get_legend_handles_labels()
    legend_ax.legend(handles, labels, loc="center", ncol=3)

    # Set axis labels
    plot_axes = [kf_ax, mae_ax, rf_ax]
    label_axes(plot_axes, fig)
    for ax in plot_axes:
        ax.get_legend().remove()
        ax.set_xlabel("Number of leaves")

    kf_ax.set_ylabel("Kuhner-Felsenstein distance")
    mae_ax.set_ylabel("Mean absolute error")
    rf_ax.set_ylabel("Normalized Robinson-Foulds distance")

    return fig


def hist_LGGC(dists, figsize):
    return hist_4x4(dists, figsize, LGGC_METHODS_NO_HAMMING)


def hist_cherry_4x4(dists, figsize):
    return hist_4x4(
        dists, figsize, sorted(FINE_TUNE_METHODS[:-1] + ["PF_Cherry+FastME"])
    )


def hist_pastek_4x4(dists, figsize):
    return hist_4x4(
        dists, figsize, sorted(FINE_TUNE_METHODS[:-1] + ["PF_SelReg+FastME"])
    )


def hist_4x4(dists, figsize, methods):

    assert len(methods) == 4  # So we dont break the layout

    # Safety barrier
    s = dists[(dists["ref_dist"] > 0) & (dists["cmp_dist"] > 0)]

    # Compute n of bins and common color scale
    _, bin_edges = np.histogram(
        np.log10(s.loc[s["marker"].isin(methods), "ref_dist"]),
        bins="auto",
    )
    bin_nr = len(bin_edges) - 1
    vmin_list, vmax_list = [], []
    for c_type in methods:
        arr, _, _ = np.histogram2d(
            np.log10(s.loc[s.marker == c_type, "ref_dist"]),
            np.log10(s.loc[s.marker == c_type, "cmp_dist"]),
            bins=bin_nr,
        )
        # To get Frequencies instead of counts
        vmin_list.append(arr.min() / (s.marker == c_type).sum())
        vmax_list.append(arr.max() / (s.marker == c_type).sum())

    # find lowest and highest counts for all subplots
    vmin_all = min(vmin_list)
    vmax_all = max(vmax_list)

    fig = plt.figure(layout="constrained", figsize=figsize)
    base = 15
    gs = GridSpec(2 * base, 2 * base + 1, figure=fig)
    ax1 = fig.add_subplot(gs[:base, :base])
    ax2 = fig.add_subplot(gs[:base, base:-1], sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(gs[-base:, :base], sharex=ax1, sharey=ax1)
    ax4 = fig.add_subplot(gs[-base:, base:-1], sharex=ax1, sharey=ax1)
    cbax = fig.add_subplot(gs[:, -1:])

    trans = mtransforms.ScaledTranslation(10 / 72, -5 / 72, fig.dpi_scale_trans)
    ctx = sns.plotting_context()
    sty = sns.axes_style()
    for method, ax in zip(methods, [ax1, ax2, ax3, ax4]):
        sns.histplot(
            data=s[s["marker"] == method],
            x="ref_dist",
            y="cmp_dist",
            log_scale=True,
            stat="proportion",
            bins=bin_nr,
            vmin=vmin_all,
            vmax=vmax_all,
            cbar=True,
            cbar_ax=cbax,
            ax=ax,
        )
        ax.text(
            0.0,
            1.0,
            method,
            transform=ax.transAxes + trans,
            fontsize=ctx["axes.titlesize"],
            verticalalignment="top",
            fontfamily=sty["font.family"][0],
            fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="none", pad=3.0),
        )

        # Label Y axes
        if ax in [ax1, ax3]:
            ax.set_ylabel("Predicted distance")
        else:
            ax.set_ylabel(None)
            ax.tick_params(axis="y", which="both", left=False, labelleft=False)

        # Lavel X axes
        if ax in [ax3, ax4]:
            ax.set_xlabel("Reference distance")
        else:
            ax.set_xlabel(None)
            ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

        # Plot reference line
        ax.axline(
            xy1=(1e-2, 1e-2),
            xy2=(1, 1),
            ls=":",
            color="red",
            zorder=0.5,
            lw=ctx["grid.linewidth"],
        )

    return fig


def hist_ft(dists, figsize, methods):

    # methods = FINE_TUNE_METHODS + ["PF_SelReg+FastME"]

    # Safety barrier
    s = dists[(dists["ref_dist"] > 0) & (dists["cmp_dist"] > 0)]

    # Compute n of bins and common color scale
    _, bin_edges = np.histogram(
        np.log10(s.loc[s["marker"].isin(methods), "ref_dist"]),
        bins="auto",
    )
    bin_nr = len(bin_edges) - 1
    vmin_list, vmax_list = [], []
    for c_type in methods:
        arr, _, _ = np.histogram2d(
            np.log10(s.loc[s.marker == c_type, "ref_dist"]),
            np.log10(s.loc[s.marker == c_type, "cmp_dist"]),
            bins=bin_nr,
        )
        vmin_list.append(arr.min())
        vmax_list.append(arr.max())

    # find lowest and highest counts for all subplots
    vmin_all = min(vmin_list)
    vmax_all = max(vmax_list)

    fig = plt.figure(layout="constrained", figsize=figsize)
    base = 15
    gs = GridSpec(3 * base, 2 * base + 1, figure=fig)
    iqax = fig.add_subplot(gs[:base, :base])
    ftax = fig.add_subplot(gs[:base, base:-1], sharex=iqax, sharey=iqax)
    fmax = fig.add_subplot(gs[base:-base, :base], sharex=iqax, sharey=iqax)
    pfax = fig.add_subplot(gs[base:-base, base:-1], sharex=iqax, sharey=iqax)
    ptax = fig.add_subplot(gs[-base:, :base], sharex=iqax, sharey=iqax)
    cbax = fig.add_subplot(gs[1:-1, -1:])

    trans = mtransforms.ScaledTranslation(10 / 72, -5 / 72, fig.dpi_scale_trans)
    ctx = sns.plotting_context()
    sty = sns.axes_style()
    for method, ax in zip(methods, [iqax, ftax, fmax, pfax, ptax]):
        sns.histplot(
            data=s[s["marker"] == method],
            x="ref_dist",
            y="cmp_dist",
            log_scale=True,
            stat="count",
            bins=bin_nr,
            vmin=vmin_all,
            vmax=vmax_all,
            cbar=True,
            cbar_ax=cbax,
            ax=ax,
        )
        ax.text(
            0.0,
            1.0,
            method,
            transform=ax.transAxes + trans,
            fontsize=ctx["axes.titlesize"],
            verticalalignment="top",
            fontfamily=sty["font.family"][0],
            fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="none", pad=3.0),
        )
        ax.axline(
            xy1=(1e-2, 1e-2), xy2=(1, 1), ls=":", color="white", zorder=2, alpha=0.5
        )

    # Set axes
    for ax in [iqax, fmax, ptax]:
        ax.set_ylabel("Predicted distance")
    for ax in [ftax, pfax]:
        ax.set_ylabel(None)
        ax.tick_params(axis="y", which="both", left=False, labelleft=False)

    for ax in [pfax, ptax]:
        ax.set_xlabel("Reference distance")
    for ax in [iqax, fmax, ftax]:
        ax.set_xlabel(None)
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    return fig


def misspecification(means, figsize, datasets, methods):
    ax_titles = dict(
        kf_score="Kuhner-Felsenstein distance",
        norm_rf="Normalized Robinson-Foulds distance",
        MAE="Mean Absolute Error",
        MRE="Mean Relative Error",
    )

    def get_heatmap(df, var):
        return df.reset_index().pivot_table(
            index="dataset", columns="marker", values=var
        )

    fig, axes = plt.subplots(
        2, 2, sharex=True, sharey=True, figsize=figsize, layout="constrained"
    )
    for ax, var in zip(axes.flatten(), ax_titles):
        sns.heatmap(
            get_heatmap(means, var).loc[datasets, methods],
            ax=ax,
            square=True,
            fmt=".1e",
            annot=True,
            annot_kws=dict(fontsize="small"),
        )
        # Set titles
        ax.set_title(ax_titles[var])
        ax.set_xlabel(None)
        ax.set_ylabel(None)

    # Remove ticks for internal axes
    for ax in axes[0, :]:
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    for ax in axes[:, 1]:
        ax.tick_params(axis="y", which="both", left=False, labelleft=False)

    # Rotate ticks for external axes
    for ax in axes[1, :]:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    for ax in axes[:, 0]:
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    return fig


def plot_brlen_dists(sub, figsize):
    tip_order = sorted(sub[legend_name].unique())
    pal = sns.cubehelix_palette(len(tip_order))

    # Distribution of branch lengths per tree size in testing set
    fig, axes = plt.subplots(
        ncols=2, layout="constrained", figsize=figsize, sharey=True
    )
    sns.kdeplot(
        data=sub,
        x="ref_len",
        hue=legend_name,
        hue_order=tip_order,
        common_norm=False,
        log_scale=True,
        ax=axes[0],
    )
    axes[0].set_ylabel("Density")
    axes[0].set_xlabel("Branch Length")
    axes[0].get_legend().remove()

    # Distribution of well and wrongly predicted branches
    for t, ls in zip(["common", "ref_unique"], ["-", "--"]):
        sns.kdeplot(
            data=sub[sub["type"] == t],
            x="ref_len",
            hue=legend_name,
            hue_order=tip_order,
            palette=pal,
            common_norm=False,
            log_scale=True,
            ls=ls,
            ax=axes[1],
        )

    # axes[1].set_ylabel("Density")
    axes[1].set_xlabel("Branch Length")
    axes[1].get_legend().remove()
    axes[1].tick_params(axis="y", which="both", left=False, labelleft=False)

    label_axes(axes, fig)

    fs = sns.plotting_context()["legend.title_fontsize"]
    lines, labels = [legend_name], [""]
    for n, col in zip(tip_order, pal):
        lines.append(Line2D([0], [0], c=col, ls="-"))
        labels.append(f"{n}")
    lines.extend(["", "Branch Inferred ?"])
    labels.extend(["", ""])
    for ls, lab in zip(["-", "--"], ["Yes", "No"]):
        lines.append(Line2D([0], [0], c="gray", ls=ls))
        labels.append(lab)
    axes[1].legend(
        lines,
        labels,
        handler_map={str: LegendTitle({"fontsize": fs})},
        loc="upper left",
        bbox_to_anchor=(1, 1),
    )

    return fig


if __name__ == "__main__":

    # Set general plotting options
    sns.set_context("notebook")
    sns.set_style("darkgrid")

    with tqdm(bar_format="[{elapsed}] {desc}", maxinterval=1) as pbar:
        ####################
        # TOPOLOGY METRICS #
        ####################

        # Parse LG+GC dataset
        pbar.set_description("Parsing LGGC topological results")
        lggc = pd.read_csv("./data/topos_lggc.csv")
        lggc["length"] = lggc["id"].apply(lambda x: x.split("_")[-1]).astype(int)
        lggc["marker"] = lggc["marker"].apply(lambda x: RENAMER.get(x, x))
        lggc["dataset"] = "LG+GC"
        pbar.update(1)

        # Choose figure size while keeping aspect ratio
        mult = 1.5
        # figsize = (4 * mult + 1, 3 * mult + 1)
        figsize = (6 * mult + 1, 3 * mult)

        # Norm RF for all aln lengths
        pbar.set_description("Plotting LG+GC topological metrics")
        fig = build_LGGC_normRF(lggc, figsize)
        plt.savefig("./figures/combined_LGGC_rf.pdf")
        plt.clf()
        plt.cla()
        pbar.update(1)

        # KF score for all aln lengths
        fig = build_LGGC_KFscore(lggc, figsize)
        plt.savefig("./figures/combined_LGGC_kf.pdf")
        plt.clf()
        plt.cla()
        pbar.update(1)

        # wRF score for all aln lengths
        fig = build_LGGC_wRF(lggc, figsize)
        plt.savefig("./figures/combined_LGGC_wrf.pdf")
        plt.clf()
        plt.cla()
        pbar.update(1)

        mult = 1.5
        figsize = (4 * mult + 1, 3 * mult + 1)

        # Single RF 500 length lGGC
        fig = single_LGGC_normRF(lggc, figsize)
        plt.savefig("./figures/LGGC_500_rf.pdf")
        plt.clf()
        plt.cla()
        pbar.update(1)

        # Single KF 500 length LGGC
        fig = single_LGGC_KFscore(lggc, figsize)
        plt.savefig("./figures/LGGC_500_kf.pdf")
        plt.clf()
        plt.cla()
        pbar.update(1)

        # Single wRF 500 length LGGC
        fig = single_LGGC_wRF(lggc, figsize)
        plt.savefig("./figures/LGGC_500_wrf.pdf")
        plt.clf()
        plt.cla()
        pbar.update(1)

        # Read fine tuned
        pbar.set_description("Parsing Pastek and Cherry Topological results")
        cherry = pd.read_csv("./data/topos_cherry.csv")
        pastek = pd.read_csv("./data/topos_pastek.csv")
        gaps = pd.read_csv("./data/topos_gaps.csv")

        for df, ds in zip([cherry, pastek, gaps], ["Cherry", "SelReg", "Indels"]):
            df["marker"] = df["marker"].apply(lambda x: RENAMER.get(x, x))
            df["dataset"], df["length"] = ds, 500

        pbar.update(1)

        # mult = 1.5
        figsize = (6 * mult + 2, 3 * mult + 1)

        # Fine tune RF
        pbar.set_description("Plotting Cherry+Pastek topological metrics")
        fig = cherry_pastek_normRF(cherry, pastek, figsize)
        plt.savefig("./figures/cherry_pastek_rf.pdf")
        plt.clf()
        plt.cla()
        pbar.update(1)

        # Fine tune KF
        fig = cherry_pastek_KFscore(cherry, pastek, figsize)
        plt.savefig("./figures/cherry_pastek_kf.pdf")
        plt.clf()
        plt.cla()
        pbar.update(1)

        # Fine tune wRF
        fig = cherry_pastek_wRF(cherry, pastek, figsize)
        plt.savefig("./figures/cherry_pastek_wrf.pdf")
        plt.clf()
        plt.cla()
        pbar.update(1)

        # Topological metrics for Cherry + Pastek
        figsize = (4.5 * mult + 1, 4.5 * mult + 1)
        fig = cherry_pastek_topos(cherry, pastek, figsize)
        plt.savefig("./figures/cherry_pastek_topos.pdf")
        plt.clf()
        plt.cla()
        pbar.update(1)

        # mult = 1.5
        figsize = (9 * mult + 2, 3 * mult + 1)

        # Fine tune RF
        pbar.set_description("Plotting fine-tuned topological metrics")
        fig = fine_tuned_normRF(gaps, cherry, pastek, figsize)
        plt.savefig("./figures/fine_tune_rf.pdf")
        plt.clf()
        plt.cla()
        pbar.update(1)

        # Fine tune KF
        fig = fine_tuned_KFscore(gaps, cherry, pastek, figsize)
        plt.savefig("./figures/fine_tune_kf.pdf")
        plt.clf()
        plt.cla()
        pbar.update(1)

        # Fine tune wRF
        fig = fine_tuned_wRF(gaps, cherry, pastek, figsize)
        plt.savefig("./figures/fine_tune_wrf.pdf")
        plt.clf()
        plt.cla()
        pbar.update(1)

        ######################
        # EXECUTION METADATA #
        ######################

        with open("./data/model_load_times.txt", "r") as f:
            times = [float(line.strip()) for line in f]
        load_time = sum(times) / len(times)

        mult = 1.5
        figsize = (4 * mult + 1, 3 * mult + 1)

        pbar.set_description("Parsing LGGC execution metadata")
        grouped_lggc = group_elapsed("./data/execution_lggc.csv")
        grouped_lggc["marker"] = grouped_lggc["marker"].apply(
            lambda x: RENAMER.get(x, x)
        )

        pbar.update(1)

        # Memory usage 500 length LGGC
        pbar.set_description("Plotting LGGC execution metadata")
        fig = single_LGGC_elapsed(grouped_lggc, figsize, load_time)
        plt.savefig("./figures/LGGC_500_elapsed.pdf")
        plt.clf()
        plt.cla()
        pbar.update(1)

        # Memory usage 500 length LGGC
        fig = single_LGGC_mem(grouped_lggc, figsize)
        plt.savefig("./figures/LGGC_500_mem.pdf")
        plt.clf()
        plt.cla()
        pbar.update(1)

        # Plot Time for pastek and cherry
        pbar.set_description("Parsing fine-tuned execution metadata")
        grouped_gaps = group_elapsed("./data/execution_gaps.csv")
        grouped_cherry = group_elapsed("./data/execution_cherry.csv")
        grouped_pastek = group_elapsed("./data/execution_pastek.csv")

        grouped_gaps["marker"] = grouped_gaps["marker"].apply(
            lambda x: RENAMER.get(x, x)
        )
        grouped_cherry["marker"] = grouped_cherry["marker"].apply(
            lambda x: RENAMER.get(x, x)
        )
        grouped_pastek["marker"] = grouped_pastek["marker"].apply(
            lambda x: RENAMER.get(x, x)
        )

        pbar.update(1)

        mult = 2
        figsize = (5 * mult + 1, 3 * mult)

        # Fine tune elapsed
        pbar.set_description("Plotting fine-tuned execution metadata")
        fig = fine_tuned_elapsed(grouped_gaps, grouped_cherry, grouped_pastek, figsize)
        plt.savefig("./figures/fine_tune_elapsed.pdf")
        plt.clf()
        plt.cla()
        pbar.update(1)

        # Fine tune mem
        fig = fine_tuned_mem(grouped_gaps, grouped_cherry, grouped_pastek, figsize)
        plt.savefig("./figures/fine_tune_mem.pdf")
        plt.clf()
        plt.cla()
        pbar.update(1)

        # Mixed dataset execution time
        mult = 1.5
        figsize = (4 * mult + 1, 3 * mult + 1)
        fig = paper_elapsed(
            pd.concat(
                [
                    grouped_lggc[grouped_lggc["length"] == 500],
                    grouped_cherry[grouped_cherry["marker"] == "IQTree_MF"],
                ]
            ),
            figsize,
        )
        plt.savefig("./figures/elapsed.pdf")
        plt.clf()
        plt.cla()
        pbar.update(1)

        ###########
        # MAE     #
        ###########

        # Percent of pairwise distances to plot
        SAMPLING_FRAC = 0.05

        mult = 1.5
        figsize = (4 * mult + 1, 3 * mult + 1)

        pbar.set_description("Parsing LGGC distance results")
        dists_lggc = pd.read_csv("./data/dists_lggc.csv").sample(frac=SAMPLING_FRAC)
        dists_lggc["n_tips"] = (
            dists_lggc["id"].apply(lambda x: x.split("_")[1]).astype(int)
        )
        dists_lggc["length"] = (
            dists_lggc["id"].apply(lambda x: x.split("_")[-1]).astype(int)
        )
        dists_lggc["MAE"] = (dists_lggc["ref_dist"] - dists_lggc["cmp_dist"]).abs()
        dists_lggc["MRE"] = dists_lggc["MAE"] / dists_lggc["ref_dist"]
        dists_lggc["MRD"] = (
            dists_lggc["ref_dist"] - dists_lggc["cmp_dist"]
        ) / dists_lggc["ref_dist"]
        dists_lggc["marker"] = dists_lggc["marker"].apply(lambda x: RENAMER.get(x, x))
        dists_lggc["dataset"] = "LG+GC"
        pbar.update(1)

        # MRE for aln 500 LG+GC
        pbar.set_description("Plotting LGGC distance results")
        fig = single_LGGC_mre(dists_lggc, figsize)
        plt.savefig("./figures/LGGC_500_mre.pdf")
        plt.clf()
        plt.cla()
        pbar.update(1)

        # MAE for aln 500 LG+GC
        fig = single_LGGC_mae(dists_lggc, figsize)
        plt.savefig("./figures/LGGC_500_mae.pdf")
        plt.clf()
        plt.cla()
        pbar.update(1)

        # Quantile plots
        sub = dists_lggc[dists_lggc["length"] == 500]
        sub["percentile"] = pd.qcut(sub["ref_dist"], 100).apply(lambda x: x.right)

        # Bin distances logarithmically
        bins = np.logspace(
            np.trunc(np.log10(sub["ref_dist"].min())),
            np.ceil(np.log10(sub["ref_dist"].max())),
            100,
        )
        sub["binned"] = pd.cut(sub["ref_dist"], bins=bins).apply(lambda x: x.right)

        # Distance percentile vs MAE
        single_LGGC_quantiles_mae(sub, figsize)
        plt.savefig("./figures/LGGC_500_quantile_mae.pdf")
        plt.clf()
        plt.cla()
        pbar.update(1)

        # Distance percentile vs MRE
        single_LGGC_quantiles_mre(sub, figsize)
        plt.savefig("./figures/LGGC_500_quantile_mre.pdf")
        plt.clf()
        plt.cla()
        pbar.update(1)

        # Distance percentile vs MRE
        single_LGGC_quantiles_mrd(sub, figsize)
        plt.savefig("./figures/LGGC_500_quantile_mrd.pdf")
        plt.clf()
        plt.cla()
        pbar.update(1)

        # Binned Distance vs MAE
        single_LGGC_binned_mae(sub, figsize)
        plt.savefig("./figures/LGGC_500_binned_mae.pdf")
        plt.clf()
        plt.cla()
        pbar.update(1)

        # Binned Distance vs MRE
        single_LGGC_binned_mre(sub, figsize)
        plt.savefig("./figures/LGGC_500_binned_mre.pdf")
        plt.clf()
        plt.cla()
        pbar.update(1)

        # Binned Distance vs MRE
        single_LGGC_binned_mrd(sub, figsize)
        plt.savefig("./figures/LGGC_500_binned_mrd.pdf")
        plt.clf()
        plt.cla()
        pbar.update(1)

        # Distribution of pairwise distances in test set trees
        legend_name = "Number of leaves"
        sub = dists_lggc[
            (dists_lggc["length"] == 500) & (dists_lggc["marker"] == "PF+FastME")
        ].rename({"n_tips": legend_name}, axis=1)
        tip_order = sorted(sub[legend_name].unique())
        fig, ax = plt.subplots(1, layout="constrained")
        sns.kdeplot(
            data=sub,
            x="ref_dist",
            hue=legend_name,
            hue_order=tip_order,
            common_norm=False,
            log_scale=True,
            ax=ax,
        )
        ax.set_ylabel("Density")
        ax.set_xlabel("Pairwise Distance")
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        plt.savefig("./figures/pairwise_dist_testset.pdf")
        plt.clf()
        plt.cla()
        pbar.update(1)

        fig = base_vs_ft(
            lggc[lggc["length"] == 500], dists_lggc[dists_lggc["length"] == 500], (9, 8)
        )
        plt.savefig("./figures/base_vs_mre.pdf")
        plt.clf()
        plt.cla()
        pbar.update(1)

        mult = 2
        figsize = (5 * mult + 1, 3 * mult)
        # Fine tune MAE
        pbar.set_description("Parsing fine-tuned distance results")
        dists_cherry = pd.read_csv("./data/dists_cherry.csv").sample(frac=SAMPLING_FRAC)
        dists_pastek = pd.read_csv("./data/dists_pastek.csv").sample(frac=SAMPLING_FRAC)
        dists_gaps = pd.read_csv("./data/dists_gaps.csv").sample(frac=SAMPLING_FRAC)
        for df, ds in zip(
            [dists_cherry, dists_pastek, dists_gaps], ["Cherry", "SelReg", "Indels"]
        ):
            df["marker"] = df["marker"].apply(lambda x: RENAMER.get(x, x))
            df["n_tips"] = df["id"].apply(lambda x: x.split("_")[1]).astype(int)
            df["MAE"] = (df["ref_dist"] - df["cmp_dist"]).abs()
            df["MRE"] = df["MAE"] / df["ref_dist"]
            df["dataset"], df["length"] = ds, 500  # Needed for mis-specification plots
        pbar.update(1)

        pbar.set_description("Plotting fine-tuned distance results")
        fig = fine_tuned_mae(dists_gaps, dists_cherry, dists_pastek, figsize)
        plt.savefig("./figures/fine_tune_mae.pdf")
        plt.clf()
        plt.cla()
        pbar.update(1)

        figsize = (6.5, 6)
        pbar.set_description("Plotting Histograms on distances in LGGC")
        fig = hist_LGGC(dists_lggc[dists_lggc["length"] == 500], figsize)
        plt.savefig("./figures/dist_hist_LGGC.png", dpi=150)
        plt.clf()
        plt.cla()
        pbar.update(1)

        # figsize = (10.5, 15)
        pbar.set_description("Plotting Histograms on distances in Pastek")
        fig = hist_cherry_4x4(dists_cherry, figsize)
        plt.savefig("./figures/dist_hist_cherry.png", dpi=150)
        plt.clf()
        plt.cla()
        pbar.update(1)

        pbar.set_description("Plotting Histograms on distances in Pastek")
        fig = hist_pastek_4x4(dists_pastek, figsize)
        plt.savefig("./figures/dist_hist_pastek.png", dpi=150)
        plt.clf()
        plt.cla()
        pbar.update(1)

        ###############
        # ALL GROUPED #
        ###############

        mult = 2
        figsize = (5 * mult + 1, 3 * mult)

        pbar.set_description("Plotting all metrics for LG+GC")
        fig = dataset_plot(
            lggc[lggc["length"] == 500],
            dists_lggc[dists_lggc["length"] == 500],
            figsize,
            LGGC_METHODS_NO_HAMMING,
        )
        plt.savefig("./figures/lggc_all.pdf")
        plt.clf()
        plt.cla()
        pbar.update(1)

        pbar.set_description("Plotting all metrics for cherry")
        fig = dataset_plot(
            cherry,
            dists_cherry,
            figsize,
            sorted(sorted(["IQTree_LG+GC"] + FINE_TUNE_METHODS) + ["PF_Cherry+FastME"]),
        )
        plt.savefig("./figures/cherry_all.pdf")
        plt.clf()
        plt.cla()
        pbar.update(1)

        pbar.set_description("Plotting all metrics for pastek")
        fig = dataset_plot(
            pastek,
            dists_pastek,
            figsize,
            sorted(sorted(["IQTree_LG+GC"] + FINE_TUNE_METHODS) + ["PF_SelReg+FastME"]),
        )
        plt.savefig("./figures/pastek_all.pdf")
        plt.clf()
        plt.cla()
        pbar.update(1)

        pbar.set_description("Plotting all metrics for gaps")
        fig = dataset_plot(
            gaps,
            dists_gaps,
            figsize,
            # Removing PF_MRE because it crushes everything
            sorted(
                [x for x in LGGC_METHODS_NO_HAMMING if x != "PF+FastME"]
                + ["PF_Indel+FastME"]
            ),
        )
        plt.savefig("./figures/gaps_all.pdf")
        plt.clf()
        plt.cla()
        pbar.update(1)

        # Model mis-specification plots
        PFS = ["PF+FastME", "PF_Indel+FastME", "PF_Cherry+FastME", "PF_SelReg+FastME"]

        # Subset and collect PF results
        dfs = []
        for df in [lggc, cherry, gaps, pastek]:
            dfs.append(df[(df["marker"].isin(PFS)) & (df["length"] == 500)])
        topo = pd.concat(dfs)
        dfs = []
        for df in [dists_lggc, dists_cherry, dists_gaps, dists_pastek]:
            dfs.append(df[(df["marker"].isin(PFS)) & (df["length"] == 500)])
        dists = pd.concat(dfs)
        for df in [topo, dists]:
            df["marker"] = df["marker"].apply(lambda x: x.removesuffix("+FastME"))

        means_alltips = (
            topo.groupby(["dataset", "marker"])[["norm_rf", "kf_score"]]
            .mean()
            .join(dists.groupby(["dataset", "marker"])[["MAE", "MRE"]].mean())
        )

        means_50 = (
            topo[topo["n_tips"] == 50]
            .groupby(["dataset", "marker"])[["norm_rf", "kf_score"]]
            .mean()
            .join(
                dists[dists["n_tips"] == 50]
                .groupby(["dataset", "marker"])[["MAE", "MRE"]]
                .mean()
            )
        )

        datasets = ["LG+GC", "Indels", "Cherry", "SelReg"]
        pf_order = [x.removesuffix("+FastME") for x in PFS]

        mult = 1.5
        figsize = (mult * 5, mult * 5)

        # 50 tips plot
        fig = misspecification(means_50, figsize, datasets, pf_order)
        plt.savefig("./figures/misspecification_50tips.pdf")
        plt.clf()
        plt.cla()

        # All tips plot
        fig = misspecification(means_alltips, figsize, datasets, pf_order)
        plt.savefig("./figures/misspecification_alltips.pdf")
        plt.clf()
        plt.cla()

        ##############
        # LIKELIHOOD #
        ##############

        pbar.set_description("Reading and plotting LGGC likelihoods")
        lik_lggc = pd.read_csv("./data/likelihoods_lggc.csv")
        lik_lggc["marker"] = lik_lggc["marker"].apply(lambda x: RENAMER.get(x, x))

        mult = 1.5
        figsize = (6 * mult + 1, 3 * mult)

        # Norm RF for all aln lengths
        fig = build_LGGC_lik(lik_lggc, figsize)
        plt.savefig("./figures/combined_LGGC_lik.pdf")
        plt.clf()
        plt.cla()

        mult = 1.5
        figsize = (4 * mult + 1, 3 * mult + 1)

        # Single RF 500 length lGGC
        fig = single_LGGC_lik(lik_lggc, figsize)
        plt.savefig("./figures/LGGC_500_lik.pdf")
        plt.clf()
        plt.cla()
        pbar.update(1)

        ##################
        # BRANCH LENGTHS #
        ##################

        brlens = pd.read_csv("./data/brlens_lggc.csv")
        legend_name = "Number of leaves"
        brlens["length"] = brlens["id"].apply(lambda x: x.split("_")[-1]).astype(int)
        brlens[legend_name] = brlens["id"].apply(lambda x: x.split("_")[1]).astype(int)
        brlens["type"] = (
            brlens["ref_len"].isna() * 10 + brlens["cmp_len"].isna()
        ).apply({0: "common", 1: "ref_unique", 10: "cmp_unique"}.get)
        sub = brlens[(brlens["length"] == 500) & (brlens["marker"] == "PF+FastME")]

        fig = plot_brlen_dists(sub, (10, 4))
        plt.savefig("./figures/branch_length_errors.pdf")
        plt.savefig("./figures/branch_length_errors.svg")
        plt.clf()
        plt.cla()
        pbar.update(1)

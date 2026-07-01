import pandas as pd

csv_path = "res/microsoft_phi-3-mini-4k-instruct_xp1_layers.csv"

df = pd.read_csv(csv_path)

metric_names = {
    "cos": "Cosine",
    "inv_endpoint": "Endpoint",
    "inv_haus": "Hausdorff",
    "inv_cham": "Chamfer",
    "inv_l2": r"$L^2$",
    "inv_h1": r"$H^1$-type",
    "inv_linf": r"$L^\infty$",
    "inv_dtw": "DTW",
}

metric_order = [
    "cos",
    "inv_endpoint",
    "inv_haus",
    "inv_cham",
    "inv_l2",
    "inv_h1",
    "inv_linf",
    "inv_dtw",
]

unaligned_metrics = [
    "cos",
    "inv_endpoint",
    "inv_haus",
    "inv_cham",
]

aligned_metrics = [
    "inv_l2",
    "inv_h1",
    "inv_linf",
    "inv_dtw",
]

import matplotlib.pyplot as plt

def format_value(x):
    return f"{x:.3f}".rstrip("0").rstrip(".")

for dataset in ["stsb", "paws", "sick"]:
    df_dataset = df[df["dataset"] == dataset].copy()

    # Keep layer and metrics only
    df_dataset = df_dataset[["layer"] + metric_order]

    # Reverse layer order for the table
    df_dataset = df_dataset.sort_values("layer", ascending=False)

    # Find the single maximum value in the whole table
    values = df_dataset[metric_order]
    max_row, max_col = values.stack().idxmax()

    # Format values and bold only the global maximum
    df_latex = df_dataset.copy()

    for col in metric_order:
        df_latex[col] = [
            rf"\textbf{{{format_value(value)}}}" if i == max_row and col == max_col else format_value(value)
            for i, value in df_latex[col].items()
        ]

    # Rename metric columns for LaTeX
    df_latex = df_latex.rename(columns=metric_names)

    tex_path = f"phi_{dataset}_layers_table.tex"

    latex_table = df_latex.to_latex(
        index=False,
        escape=False,
        longtable=True,
        caption=f"Layer-wise results on {dataset.upper()} for Qween",
        label=f"tab:phi_{dataset}_layers",
    )

    with open("tex/"+tex_path, "w") as f:
        f.write(latex_table)

    print(f"Wrote {tex_path}")

    unaligned_metrics = [
        "cos",
        "inv_endpoint",
        "inv_haus",
        "inv_cham",
    ]

    aligned_metrics = [
        "inv_l2",
        "inv_h1",
        "inv_linf",
        "inv_dtw",
    ]

    # Plot one line per metric + group averages
    df_plot = df_dataset.sort_values("layer", ascending=True).copy()

    # Average over unaligned and aligned metrics separately
    df_plot["Unaligned average"] = df_plot[unaligned_metrics].mean(axis=1)
    df_plot["Aligned average"] = df_plot[aligned_metrics].mean(axis=1)

    plt.figure(figsize=(8, 5))

    for metric in metric_order:
        plt.plot(
            df_plot["layer"],
            df_plot[metric],
            marker="o",
            label=metric_names[metric],
        )

    plt.plot(
        df_plot["layer"],
        df_plot["Unaligned average"],
        marker="o",
        linewidth=3,
        linestyle="--",
        label="Unaligned average",
    )

    plt.plot(
        df_plot["layer"],
        df_plot["Aligned average"],
        marker="o",
        linewidth=3,
        linestyle="--",
        label="Aligned average",
    )

    plt.xlabel("Layer")
    plt.ylabel("Score")
    plt.title(f"{dataset.upper()} layer-wise results")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = f"phi_{dataset}_layers_plot.pdf"
    plt.savefig("plots/"+plot_path, bbox_inches="tight")
    plt.close()

    print(f"Wrote {plot_path}")

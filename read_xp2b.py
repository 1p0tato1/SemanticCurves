import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = "res/microsoft_phi-3-mini-4k-instruct_xp2_sigma.csv"

df = pd.read_csv(path)
df = df.rename(columns={"energy": "correlation"})
df = df.rename(columns={"tau": "sigma"})
df = df.sort_values(["layer", "sigma"])

dataset_name = df["dataset"].iloc[0]

plt.figure(figsize=(7, 5))

for layer, layer_df in df.groupby("layer"):
    layer_df = layer_df.sort_values("sigma")

    x = layer_df["sigma"].to_numpy()
    y = layer_df["correlation"].to_numpy()

    line, = plt.plot(
        x,
        y,
        linewidth=2,
        label=f"Layer {layer}"
    )

        # Find the maximum correlation for this layer
    max_idx = layer_df["correlation"].idxmax()
    max_sigma = layer_df.loc[max_idx, "sigma"]
    max_corr = layer_df.loc[max_idx, "correlation"]

    color = line.get_color()

    plt.axvline(
        x=max_sigma,
        color=color,
        linestyle=":",
        linewidth=1.5,
        alpha=0.8
    )

    plt.scatter(
        max_sigma,
        max_corr,
        color=color,
        s=45,
        zorder=3
    )

    # Move annotation slightly to the right
    x_offset = 0.01 * (df["sigma"].max() - df["sigma"].min())
    y_offset = -0.1 * (df["correlation"].max() - df["correlation"].min())

    plt.text(
        max_sigma + x_offset,
        max_corr + y_offset,
        rf"$\sigma$={max_sigma:.3g}" + f"\nmax={max_corr:.3g}",
        color=color,
        fontsize=9,
        ha="left",
        va="bottom"
    )


plt.xlabel(r"$\sigma$")
plt.ylabel("Correlation")
plt.title(f"Correlation vs sigma on {dataset_name}")
plt.legend(title="Layer")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("plots/correlation_across_layers.pdf", format="pdf", bbox_inches="tight")
plt.show()

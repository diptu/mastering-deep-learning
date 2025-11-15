import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle


class NeuralNetwork:
    def __init__(
        self,
        layers,
        max_display=31,
        max_arrows=5,
        top_n=5,
        bottom_n=5,
        show_ellipsis=True,
    ):
        self.layers = layers
        self.max_display = max_display
        self.max_arrows = max_arrows
        self.top_n = top_n
        self.bottom_n = bottom_n
        self.show_ellipsis = show_ellipsis
        self.positions = None

    def _compute_positions(self):
        """Compute neuron positions, centered vertically for pyramid effect"""
        positions = []
        for n in self.layers:
            if n <= self.max_display:
                y_positions = (
                    np.linspace(-(n - 1) / 2, (n - 1) / 2, n) / max(self.layers) * 1.8
                )
            else:
                top_y = (
                    np.linspace(
                        0.9, 0.9 - 0.4 * (self.top_n - 1) / (self.top_n - 1), self.top_n
                    )
                    if self.top_n > 1
                    else [0.9]
                )
                bottom_y = (
                    np.linspace(
                        -0.9 + 0.4 * (self.bottom_n - 1) / (self.bottom_n - 1),
                        -0.9,
                        self.bottom_n,
                    )
                    if self.bottom_n > 1
                    else [-0.9]
                )
                y_positions = list(top_y)
                if self.show_ellipsis:
                    y_positions.append(0.0)
                y_positions.extend(list(bottom_y))
            positions.append(y_positions)
        return positions

    def _get_layer_name(self, idx):
        if idx == 0:
            return "Input Layer"
        elif idx == len(self.layers) - 1:
            return "Output Layer"
        else:
            return f"Hidden Layer {idx}"

    def plot(self, figsize=(10, 6), save_path=None, quality="avg"):
        dpi_map = {"low": 72, "avg": 150, "high": 300}
        dpi = dpi_map.get(quality, 150)

        self.positions = self._compute_positions()
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        node_radius = 0.03
        rect_pad_x = 0.03
        rect_pad_y = 0.05

        x_positions = np.linspace(0, 1, len(self.layers))

        # Draw layer rectangles
        for idx, y_positions in enumerate(self.positions):
            x_vals = [x_positions[idx]] * len(y_positions)
            y_vals = y_positions
            x_min = min(x_vals) - node_radius - rect_pad_x
            x_max = max(x_vals) + node_radius + rect_pad_x
            y_min = min(y_vals) - node_radius - rect_pad_y
            y_max = max(y_vals) + node_radius + rect_pad_y
            rect = Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                facecolor="lightgray",
                alpha=0.2,
            )
            ax.add_patch(rect)

        # Draw neurons
        for layer_idx, y_positions in enumerate(self.positions):
            x_layer = x_positions[layer_idx]
            n_total = self.layers[layer_idx]

            # Layer label
            y_top = max(y_positions) + 0.05
            ax.text(
                x_layer,
                y_top,
                self._get_layer_name(layer_idx),
                ha="center",
                fontsize=9,
                fontweight="bold",
            )

            for i, y in enumerate(y_positions):
                circle = Circle((x_layer, y), node_radius, color="skyblue", alpha=0.5)
                ax.add_patch(circle)

                if n_total <= self.max_display:
                    ax.text(
                        x_layer,
                        y,
                        f"$a^{layer_idx}_{i + 1}$",
                        ha="center",
                        va="center",
                        fontsize=5,
                    )
                else:
                    if i < self.top_n:
                        ax.text(
                            x_layer,
                            y,
                            f"$a^{layer_idx}_{i + 1}$",
                            ha="center",
                            va="center",
                            fontsize=5,
                        )
                    elif self.show_ellipsis and i == self.top_n:
                        ax.text(
                            x_layer, y, r"$\dots$", ha="center", va="center", fontsize=6
                        )
                    else:
                        idx_num = n_total - (len(y_positions) - i - 1)
                        ax.text(
                            x_layer,
                            y,
                            f"$a^{layer_idx}_{idx_num}$",
                            ha="center",
                            va="center",
                            fontsize=5,
                        )

        # Draw arrows, weights, and biases
        for l in range(len(self.positions) - 1):
            src_y = self.positions[l]
            tgt_y = self.positions[l + 1]
            src_x = x_positions[l]
            tgt_x = x_positions[l + 1]

            for i, y_start in enumerate(src_y):
                # Choose target indices for arrows
                if len(tgt_y) <= self.max_display:
                    tgt_indices = range(len(tgt_y))
                else:
                    tgt_indices = range(len(tgt_y))
                step = max(1, len(tgt_indices) // self.max_arrows)

                for j in tgt_indices[::step]:
                    y_end = tgt_y[j]
                    # Arrow
                    ax.annotate(
                        "",
                        xy=(tgt_x - node_radius, y_end),
                        xytext=(src_x + node_radius, y_start),
                        arrowprops=dict(arrowstyle="->", lw=0.7),
                    )

                    # Weight label
                    dx = (tgt_x - src_x) * 0.5
                    dy = (y_end - y_start) * 0.5
                    ax.text(
                        src_x + dx + 0.01,
                        y_start + dy + 0.01,
                        f"$W^{{{l + 1}}}_{{{j + 1}{i + 1}}}$",
                        fontsize=6,
                        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
                    )

            # Bias labels for target layer
            for j, y_end in enumerate(tgt_y):
                ax.text(tgt_x + 0.04, y_end, f"$b^{{{l + 1}}}_{{{j + 1}}}$", fontsize=6)

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.axis("off")

        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
            print(f"Saved plot to {save_path} at {dpi} dpi ({quality} quality).")
        plt.show()


# ================= Example =================
nn = NeuralNetwork(
    layers=[4, 3, 2, 1],
    max_display=4,
    max_arrows=4,
    # top_n=2,
    # bottom_n=2,
    show_ellipsis=True,
)
nn.plot(figsize=(10, 6), save_path="./ANN/network.png", quality="low")

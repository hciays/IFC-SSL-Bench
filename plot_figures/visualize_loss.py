from matplotlib import pyplot as plt
from torch import nn

from utils.utils import get_loss_path, get_params_text


def visualize_loss(model: nn.Module, h_params: dict, data_info: dict, run_time: str) -> None:
    """
    Plots and saves training loss curves to file.
    """
    # figure size and resolution
    fig, ax = plt.subplots(figsize=(8, 4), dpi=350)
    # Plot training loss
    ax.plot(model.train_loss_epoch, label="Training Loss", linewidth=1.5)

    # Labels and title
    ax.set_xlabel("Training epochs")
    ax.set_ylabel("Loss")
    ax.set_title(
        f"{data_info['dataset']}, {h_params['model'].__name__}, {h_params['backbone'].__name__}: "
        f"{model.train_loss_epoch[-1]:.3f}"
    )

    # Add model & data params as text
    params_txt = get_params_text(h_params, data_info)
    fig.text(0.1, -0.04, params_txt, ha='left', va='bottom', fontsize=8, family='monospace')

    # Adjust layout and style
    fig.subplots_adjust(bottom=0.16)
    ax.grid(True)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=-1 if h_params['model'].__name__ == "BYOL" else 0)
    ax.legend()

    # Save image
    save_path = get_loss_path(h_params, data_info, run_time)
    fig.savefig(save_path, bbox_inches='tight', dpi=350)
    print(f"Saved loss img to {save_path}")

    plt.show(block=False)
    plt.pause(5)
    plt.close()

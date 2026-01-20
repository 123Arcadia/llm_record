
import matplotlib.pyplot as plt


def plot_losses(losses, file_name="losses.jpg"):

    ax = plt.subplot()
    plt.xlabel("train_loader (length)")
    plt.ylabel("Loss")
    plt.title(f"Losses")

    plt.plot(range(len(losses)), losses, label="Loss")
    plt.legend()
    # plt.show()
    plt.savefig(file_name, dpi=300)

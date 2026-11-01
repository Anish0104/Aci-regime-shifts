"""Save the four baseline cases: python -m data.plot_synthetic."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .synthetic import generate_ar1


def main():
    fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex=True)
    for row, shift_type in enumerate(("mean", "variance")):
        for col, speed in enumerate(("abrupt", "gradual")):
            x, shift_at = generate_ar1(shift_type=shift_type, speed=speed)
            assert int(.8 * len(x)) <= shift_at < len(x)
            ax = axes[row, col]
            ax.plot(x, linewidth=.5)
            ax.axvspan(int(.8*len(x)), len(x), color="orange", alpha=.1)
            ax.axvline(shift_at, color="red", linestyle="--")
            ax.set_title(f"{shift_type}, {speed}")
    fig.tight_layout()
    target = Path(__file__).parent / "synthetic_cases.png"
    fig.savefig(target, dpi=150)
    plt.close(fig)
    print(target)


if __name__ == "__main__":
    main()

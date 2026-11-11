import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "carousel"
GITHUB_LINK = "github.com/Vinaypaul7/loan-approval-decision-tree"
TOTAL_SLIDES = 5

NAVY, BLUE, LIGHT, RED = "#1F3864", "#2E75B6", "#F4F7FB", "#C0392B"


def new_slide(number: int):
    """Create a blank square slide with a top bar and footer."""
    fig = plt.figure(figsize=(10.8, 10.8), dpi=100)
    fig.patch.set_facecolor(LIGHT)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.add_patch(plt.Rectangle((0, 0.97), 1, 0.03, color=BLUE))
    ax.text(0.05, 0.03, "Day 3 | ML Journey", fontsize=16, color="gray")
    ax.text(0.95, 0.03, f"{number}/{TOTAL_SLIDES}", fontsize=16, color="gray", ha="right")
    return fig, ax


def title(ax, text: str, y: float = 0.93, size: int = 32) -> None:
    ax.text(0.5, y, text, fontsize=size, weight="bold", color=NAVY,
            ha="center", va="top", linespacing=1.4)


def slide_hook(results: dict):
    unlimited = results["depth_results"][-1]
    fig, ax = new_slide(1)
    ax.text(0.5, 0.68, f"{unlimited['train_accuracy']:.0%} train accuracy.",
            fontsize=46, weight="bold", color=NAVY, ha="center")
    ax.text(0.5, 0.56, f"{unlimited['test_accuracy']:.0%} on new data.",
            fontsize=46, weight="bold", color=RED, ha="center")
    ax.text(0.5, 0.40, "What my Decision Tree taught me\nabout overfitting",
            fontsize=26, color=BLUE, ha="center", linespacing=1.5)
    ax.text(0.5, 0.20, "Loan Approval Prediction  |  Day 3",
            fontsize=20, color="gray", ha="center")
    return fig


def slide_depth_table(results: dict):
    fig, ax = new_slide(2)
    title(ax, "Deeper tree = better memory,\nnot better learning", size=30)
    y = 0.68
    for x, header in [(0.15, "max_depth"), (0.50, "Train Acc"), (0.80, "Test Acc")]:
        ax.text(x, y, header, fontsize=22, weight="bold", color=BLUE, ha="center")
    y -= 0.068
    for row in results["depth_results"]:
        memorized = row["train_accuracy"] == 1.0
        if memorized:
            ax.add_patch(plt.Rectangle((0.06, y - 0.025), 0.88, 0.065, color="#FADBD8"))
        depth = "None" if row["max_depth"] is None else str(row["max_depth"])
        ax.text(0.15, y, depth, fontsize=22, ha="center", color=NAVY)
        ax.text(0.50, y, f"{row['train_accuracy']:.3f}", fontsize=22, ha="center",
                color=RED if memorized else NAVY, weight="bold" if memorized else "normal")
        ax.text(0.80, y, f"{row['test_accuracy']:.3f}", fontsize=22, ha="center", color=NAVY)
        y -= 0.068
    ax.text(0.5, 0.10, "Train accuracy keeps climbing.\nTest accuracy does not follow.",
            fontsize=22, color=NAVY, ha="center", va="center", linespacing=1.5)
    return fig


def slide_image(number: int, heading: str, image: str, caption: str, rect: list):
    fig, ax = new_slide(number)
    title(ax, heading)
    img_ax = fig.add_axes(rect)
    img_ax.imshow(mpimg.imread(BASE_DIR / image))
    img_ax.axis("off")
    ax.text(0.5, 0.15, caption, fontsize=21, color=BLUE, ha="center", linespacing=1.5)
    return fig


def slide_takeaway():
    fig, ax = new_slide(5)
    ax.text(0.5, 0.80, "Key takeaway", fontsize=26, color=BLUE, ha="center")
    ax.text(0.5, 0.56, "Always test on\nunseen data.", fontsize=48, weight="bold",
            color=NAVY, ha="center", linespacing=1.3)
    ax.text(0.5, 0.36, "A perfect training score can mean\nthe model memorized the answers.",
            fontsize=22, color="gray", ha="center", linespacing=1.5)
    ax.add_patch(plt.Rectangle((0.08, 0.14), 0.84, 0.10, color=NAVY))
    ax.text(0.5, 0.19, GITHUB_LINK, fontsize=19, color="white", ha="center", va="center")
    return fig


def main() -> None:
    results = json.loads((BASE_DIR / "results.json").read_text())
    top_feature, top_value = next(iter(results["feature_importance"].items()))
    top_label = top_feature.replace("_", " ")

    slides = [
        slide_hook(results),
        slide_depth_table(results),
        slide_image(3, f"The final tree (max_depth = {results['final_model']['max_depth']})",
                    "decision_tree.png",
                    "Every prediction can be traced\nthrough simple yes/no questions",
                    [0.03, 0.30, 0.94, 0.5]),
        slide_image(4, f"{top_label.capitalize()} drove ~{top_value:.0%}\nof the decisions",
                    "feature_importance.png",
                    "Sensible result: banks weigh credit history first",
                    [0.08, 0.25, 0.84, 0.48]),
        slide_takeaway(),
    ]

    OUT_DIR.mkdir(exist_ok=True)
    with PdfPages(BASE_DIR / "LinkedIn_carousel.pdf") as pdf:
        for i, fig in enumerate(slides, 1):
            fig.savefig(OUT_DIR / f"slide_{i}.png", dpi=100, facecolor=LIGHT)
            pdf.savefig(fig, facecolor=LIGHT)
            plt.close(fig)
    print("Saved LinkedIn_carousel.pdf and carousel/slide_1..5.png")


if __name__ == "__main__":
    main()
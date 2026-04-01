import matplotlib.pyplot as plt
from pathlib import Path

models = [
    "FP32",
    "INT8",
    "Frozen INT8",
    "Greedy Mixed",
    "SA Mixed"
]

accuracy = [
    0.9117,
    0.9117,
    0.8945,
    0.9106,
    0.9002
]

size = [
    255.41,
    63.85,
    63.85,
    228.39,
    122.67
]

plt.figure()

plt.scatter(size, accuracy)

for i, name in enumerate(models):
    plt.text(size[i]+2, accuracy[i], name)

plt.xlabel("Model Size (MB)")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Model Size Trade-off")

plt.grid(True)

FIGURE_PATH = Path(__file__).resolve().parents[1] / "figures" / "accuracy_vs_size_tradeoff.png"
FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(FIGURE_PATH, dpi=300)

plt.show()

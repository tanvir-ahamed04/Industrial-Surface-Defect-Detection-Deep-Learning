import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("model_comparison.csv")

plt.figure(figsize=(6,4))
plt.bar(df["Model"], df["Recall@0.5"])
plt.ylabel("Recall @ IoU = 0.5")
plt.title("Faster R-CNN vs RetinaNet")
plt.ylim(0, 1)

for i, v in enumerate(df["Recall@0.5"]):
    plt.text(i, v + 0.01, f"{v:.3f}", ha="center")

plt.tight_layout()
plt.savefig("recall_comparison.png", dpi=300)
plt.show()

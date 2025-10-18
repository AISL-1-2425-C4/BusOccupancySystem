import pandas as pd
import matplotlib.pyplot as plt

# Load YOLO training log
csv_path = "results.csv"  # change path to your run
df = pd.read_csv(csv_path)

# Loss types to compare
loss_types = ["box_loss", "cls_loss", "dfl_loss"]

# Plot each loss type
plt.figure(figsize=(12, 4))

for i, loss in enumerate(loss_types, 1):
    plt.subplot(1, 3, i)
    plt.plot(df['epoch'], df[f"train/{loss}"], label=f"Train {loss}")
    plt.plot(df['epoch'], df[f"val/{loss}"], label=f"Val {loss}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{loss} (Train vs Val)")
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

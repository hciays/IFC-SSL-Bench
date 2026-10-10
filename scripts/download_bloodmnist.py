from medmnist import BloodMNIST
from pathlib import Path

# persistent workspace datasets dir:
root = ""

root.mkdir(parents=True, exist_ok=True)

for split in ["train", "val", "test"]:
    print(f"Downloading BloodMNIST (size=256), split={split} to {root}")
    BloodMNIST(root=root, split=split, download=True, size=224)

print("Done.")
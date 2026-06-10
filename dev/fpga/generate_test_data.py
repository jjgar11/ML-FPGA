"""
Run on host/VM (requires sklearn).
Generates test_data.json with scaled Wine dataset for FPGA inference.
"""
import json
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

wine   = load_wine()
scaler = StandardScaler()
X      = scaler.fit_transform(wine.data).tolist()
y      = wine.target.tolist()

with open("test_data.json", "w") as f:
    json.dump({"X": X, "y": y, "names": list(wine.target_names)}, f)

print(f"Generated test_data.json: {len(y)} samples, {len(X[0])} features")
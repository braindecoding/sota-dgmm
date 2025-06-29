from scipy.io import loadmat
import numpy as np

# Check miyawaki dataset structure
print("Checking miyawaki_conditions_2to5_combined_sharp.mat:")
data = loadmat('miyawaki_conditions_2to5_combined_sharp.mat')
print("Keys:", [k for k in data.keys() if not k.startswith('__')])
for k, v in data.items():
    if not k.startswith('__'):
        if hasattr(v, 'shape'):
            print(f"{k}: {v.shape}, dtype: {v.dtype}")
            if len(v.shape) <= 2 and v.size < 20:
                print(f"   Sample values: {v.flatten()[:10]}")
            print(f"   Range: [{v.min():.3f}, {v.max():.3f}]")
        else:
            print(f"{k}: {type(v)}")
        print()

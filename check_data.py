from scipy.io import loadmat

# Check digit69_28x28.mat
print("Checking digit69_28x28.mat:")
data = loadmat('digit69_28x28.mat')
print("Keys:", [k for k in data.keys() if not k.startswith('__')])
for k, v in data.items():
    if not k.startswith('__'):
        if hasattr(v, 'shape'):
            print(f"{k}: {v.shape}")
        else:
            print(f"{k}: {type(v)}")

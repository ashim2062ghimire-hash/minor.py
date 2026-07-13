import scipy.io

# 1. Provide the path to one of your CWRU .mat files
file_path = "data/raw/cwru/H_0" 

# 2. Load the raw MATLAB file data structure
mat_data = scipy.io.loadmat(file_path)

print("--- 📁 File Keys & Metadata ---")
for key in mat_data.keys():
    # Filter out internal MATLAB system headers
    if not key.startswith('__'):
        value = mat_data[key]
        print(f"Variable Name: '{key}'")
        print(f"  └─ Data Type:  {type(value)}")
        print(f"  └─ Data Shape: {getattr(value, 'shape', 'Scalar Value')}")
        print("-" * 30)

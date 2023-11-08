import h5py

def print_h5_keys(file_path):
    with h5py.File(file_path, 'r') as file:
        def explore_group(group, indent=0):
            for key in group.keys():
                if isinstance(group[key], h5py.Group):
                    print("  " * indent + f"Group: {key}")
                    explore_group(group[key], indent + 1)
                else:
                    print("  " * indent + f"Dataset: {key}")

        print(f"Keys in {file_path}:")
        explore_group(file)

file_path = "./LUAC_1-1_A4.h5"
print_h5_keys(file_path)
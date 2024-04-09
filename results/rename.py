import pickle

# Variables for original filenames (assuming these are defined somewhere in your code)
DCII_Mod = 'DCI Lasso Mod'
DCII_Comp = 'DCI Lasso Comp'
DCII_Expl = 'DCI Lasso Expl'

# List of tuples containing the original file paths and the desired new names
files = [
    (DCII_Mod, DCII_Mod.replace("DCI Lasso", "DCI")),
    (DCII_Comp, DCII_Comp.replace("DCI Lasso", "DCI")),
    (DCII_Expl, DCII_Expl.replace("DCI Lasso", "DCI")),
]

for old_path, new_path in files:
    # Load the original pickle file
    with open(old_path, 'rb') as file:
        data = pickle.load(file)

    # Modify the keys from DCII to EDI
    # Note: This assumes that the key you're changing is a top-level key in the data dictionary
    modified_data = {key.replace("DCI Lasso", "DCI") if "DCI Lasso" in key else key: value for key, value in data.items()}

    # Save the modified data back to a new pickle file with the new name
    with open(new_path, 'wb') as file:
        pickle.dump(modified_data, file)

    print(f"Processed and saved new file: {new_path}")

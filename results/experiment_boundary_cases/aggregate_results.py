import pandas as pd
import os

# Base directory where the folders with CSV files are located
base_dir = os.getcwd()

# Placeholder for collecting DataFrames
dfs = []

# Loop through each item in base directory
for folder_name in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder_name)

    # Check if the item is a directory
    if os.path.isdir(folder_path):
        # Loop through each file in the folder
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)

            # Check if the file is a CSV
            if file_path.endswith('.csv'):
                # Read the CSV file into a DataFrame and append it to the list
                df = pd.read_csv(file_path)
                dfs.append(df)

# Instead of concatenating, we process each DataFrame to sum the values grouped by "case"
summed_dfs = []
for df in dfs:
    # Exclude the 'seed' column from the sum if it's numeric, sum others
    summed_df = df.drop(columns=['seed']).groupby('case').sum().reset_index()
    summed_dfs.append(summed_df)

# Now, concatenate the summed DataFrames for mean and std calculation
combined_summed_df = pd.concat(summed_dfs, ignore_index=True)

# Group by 'case' again in the combined summed DataFrame and calculate mean and std
final_grouped = combined_summed_df.groupby('case').agg(['mean', 'std']).reset_index()

# Flatten MultiIndex columns if needed
final_grouped.columns = [' '.join(col).strip() for col in final_grouped.columns.values]

# Optionally, save the final grouped DataFrame with mean and std to a CSV
final_grouped.to_csv('grouped_summary_statistics.csv', index=False)

print(final_grouped)

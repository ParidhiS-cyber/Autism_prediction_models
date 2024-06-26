import pandas as pd
import os

# Define the folder containing the tsv files
input_folder = "/home/ibab/output_processed"

# Define the folder to store the output csv files
output_folder = "/home/ibab/output_processed_counts"

# Iterate over each file in the folder
for file_name in os.listdir(input_folder):
    if file_name.endswith(".tsv"):
        # Construct the full path to the input file
       input_file_path = os.path.join(input_folder, file_name)

        # Read the tsv file into a dataframe
       df = pd.read_csv(input_file_path, sep="\t")

        # Group the dataframe by variant type and count the number of varaints in each group
       variant_counts = df.groupby(df.columns[0]).size().reset_index(name='Count')

        # Display the variant counts
       print(f"Variant counts for file: {file_name}")
       print(variant_counts)

        # Generate the output file name
       output_file_name = file_name.replace(".tsv", "_variant_counts.csv")
       output_file_path = os.path.join(output_folder, output_file_name)

        # Save the variant counts to a csv file in the output folder
       variant_counts.to_csv(output_file_path, index=False)





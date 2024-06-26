import os

# Define the input folder containing GVCF files
input_folder = "/home/group_shyam01/annotated_gvcfs/non-ASD"

# Define the output folder to store the extracted information
output_folder = "/home/group_shyam01/annotated_gvcfs/non-ASD/extracted_info"

# Define the chromosomes of interest
chromosomes = ["chr7", "chr11"]

# Process each GVCF file in the input folder
for root, dirs, files in os.walk(input_folder):
    for file in files:
        if file.endswith(".gvcf"):
            vcf_file = os.path.join(root, file)
            output_subfolder = os.path.join(output_folder, os.path.basename(root))
            os.makedirs(output_subfolder, exist_ok=True)
            output_file_prefix = os.path.splitext(file)[0]
            # Process each chromosome
            for chromosome in chromosomes:
                output_file = os.path.join(output_subfolder, f"{output_file_prefix}_extracted_info_{chromosome}.tsv")
                # Open the input VCF file for reading and the output file for writing
                with open(vcf_file, "r") as infile, open(output_file, "w") as outfile:
                    # Write the header to the output file
                    outfile.write("Variant_Type\tREF\tALT\tGene\n")
                    # Iterate over each line in the VCF file
                    for line in infile:
                        # Skip header lines and lines not belonging to the chromosome of interest
                        if line.startswith("#") or not line.startswith(chromosome):
                            continue
                        # Split the line into fields
                        fields = line.strip().split("\t")
                        # Extract relevant information
                        REF = fields[3]
                        ALT = fields[4].split(",")[0]  # Only consider the first ALT allele
                        annotation = fields[7]
                        annotations = annotation.split(",")
                        for ann in annotations:
                            ann_parts = ann.split("|")
                            if len(ann_parts) >= 5:
                                gene = ann_parts[3]
                                variant_type = ann_parts[1]
                                # Write the extracted information to the output file
                                outfile.write(f"{variant_type}\t{REF}\t{ALT}\t{gene}\n")
            print(f"Extracted information for GVCF file {file} saved to {output_subfolder}")
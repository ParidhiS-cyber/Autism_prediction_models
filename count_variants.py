import os
import pandas as pd

# Define the directory containing the TSV files
directory = '/home/group_shyam01/annotated_gvcfs/non-ASD/filter_tsv'

# Define the output CSV file
output_file = '/home/group_shyam01/annotated_gvcfs/non-ASD/filter_tsv/results.csv'

# Define the chromosomes and variant types of interest
chromosomes_of_interest = ['chr4', 'chr19', 'chr11', 'chr17', 'chr7', 'chr15', 'chrX']
variant_types_of_interest = [
    'intron_variant',
    'missense_variant',
    '5_prime_UTR_variant',
    '3_prime_UTR_variant',
    'intergenic_region'
]

# List to hold the results for each file
results = []

# Process each TSV file
for filename in os.listdir(directory):
    if filename.endswith(".tsv"):
        filepath = os.path.join(directory, filename)
        df = pd.read_csv(filepath, sep='\t')

        # Initialize counters for the current file
        total_variant_count = len(df)
        chromosome_total_counts = {chromosome: 0 for chromosome in chromosomes_of_interest}
        variant_counts = {
            chromosome: {variant_type: 0 for variant_type in variant_types_of_interest}
            for chromosome in chromosomes_of_interest
        }

        # Update counts for specific chromosomes and variant types
        for chromosome in chromosomes_of_interest:
            chromosome_df = df[df['CHROM'] == chromosome]
            chromosome_total_counts[chromosome] = len(chromosome_df)
            for variant_type in variant_types_of_interest:
                variant_counts[chromosome][variant_type] += len(chromosome_df[chromosome_df['TYPE'] == variant_type])

        # Append the results for the current file to the list
        for chromosome, counts in variant_counts.items():
            result = {
                'Filename': filename,
                'Total_Variants': total_variant_count,
                'Chromosome': chromosome,
                'Total_Count': chromosome_total_counts[chromosome]
            }
            result.update({f'{variant_type}_count': count for variant_type, count in counts.items()})
            results.append(result)

# Convert the results to a DataFrame and save to a CSV file
results_df = pd.DataFrame(results)
results_df.to_csv(output_file, index=False)

print(f"Results have been saved to {output_file}")

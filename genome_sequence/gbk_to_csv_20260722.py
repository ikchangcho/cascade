from Bio import SeqIO
import pandas as pd

# 1. Set the name of your GenBank file
gbk_file = "GL78RF_45_HMWF035_1_reference.gbk"
output_csv = "gene_list_HMWF035.csv"

gene_data = []

# 2. Parse the GenBank file
print(f"Reading {gbk_file}...")
for record in SeqIO.parse(gbk_file, "genbank"):
    # Loop through all the features in each contig
    for feature in record.features:
        # We only care about Coding Sequences (CDS) and RNA
        if feature.type in ["CDS", "tRNA", "rRNA"]:
            
            # Extract locus tag (the unique ID for the gene)
            locus_tag = feature.qualifiers.get("locus_tag", ["N/A"])[0]
            
            # Extract the gene name (if it has one)
            gene_name = feature.qualifiers.get("gene", ["N/A"])[0]
            
            # Extract the product description (what the gene does)
            product = feature.qualifiers.get("product", ["Unknown function"])[0]
            
            # Save this information
            gene_data.append({
                "Contig": record.id,
                "Locus Tag": locus_tag,
                "Type": feature.type,
                "Gene Symbol": gene_name,
                "Product / Function": product
            })

# 3. Convert the list to a DataFrame and save as CSV
df = pd.DataFrame(gene_data)
df.to_csv(output_csv, index=False)

print(f"Success! {len(df)} genes have been extracted and saved to {output_csv}.")
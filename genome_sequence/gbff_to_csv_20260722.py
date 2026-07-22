import pandas as pd
from Bio import SeqIO

def parse_gbff_to_dataframe(gbff_path):
    """
    Parses a GenBank file (.gbff / .gbk) and extracts detailed gene/CDS information.
    Returns a pandas DataFrame.
    """
    records = []

    # Iterate through all genomic records (e.g., chromosomes/plasmids) in the file
    for record in SeqIO.parse(gbff_path, "genbank"):
        chromosome_id = record.id
        
        for feature in record.features:
            # We focus primarily on 'CDS' features, but you can change this to 'gene' if preferred
            if feature.type == "CDS":
                
                # Extract locations and handle strand direction
                start = int(feature.location.start) + 1  # Convert 0-based index to 1-based coordinates
                end = int(feature.location.end)
                strand = "+" if feature.location.strand == 1 else "-"

                # Extract common qualifiers safely using .get()
                qualifiers = feature.qualifiers
                
                gene_symbol = qualifiers.get("gene", [""])[0]
                locus_tag = qualifiers.get("locus_tag", [""])[0]
                product = qualifiers.get("product", [""])[0]
                protein_id = qualifiers.get("protein_id", [""])[0]
                translation = qualifiers.get("translation", [""])[0]

                records.append({
                    "Contig/Chromosome": chromosome_id,
                    "Locus Tag": locus_tag,
                    "Gene Symbol": gene_symbol,
                    "Start": start,
                    "End": end,
                    "Strand": strand,
                    "Product": product,
                    "Protein ID": protein_id,
                    "Amino Acid Length": len(translation) if translation else 0,
                    "Protein Sequence": translation
                })

    df = pd.DataFrame(records)
    return df


# --- Usage Example ---
if __name__ == "__main__":
    gbff_filename = "OTU695.gbff"  # Replace with your actual .gbff file path
    csv_output = "gene_list_OTU695.csv"

    # Convert GBFF to DataFrame
    df_genes = parse_gbff_to_dataframe(gbff_filename)

    # Save to CSV
    df_genes.to_csv(csv_output, index=False)
    
    print(f"Successfully processed {len(df_genes)} genes!")
    print(df_genes.head())  # Display the first few rows
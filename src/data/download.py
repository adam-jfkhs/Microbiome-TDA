"""Dataset download scripts for HMP and AGP microbiome data."""

import os
import urllib.request

from biom import load_table


def download_hmp(data_dir="data/raw/hmp"):
    """Download HMP Phase 1 16S data.

    Downloads the OTU table (BIOM format) and metadata from the
    Human Microbiome Project DACC portal.
    """
    os.makedirs(data_dir, exist_ok=True)

    # HMP1 — 16S v3-v5 region OTU table
    biom_url = (
        "https://downloads.hmpdacc.org/dacc/hhs/genome/microbiome/"
        "wgs/analysis/hmqcp/v35/otu_table_psn_v35.biom"
    )
    biom_path = os.path.join(data_dir, "hmp1_otu_table.biom")

    if not os.path.exists(biom_path):
        print("Downloading HMP1 OTU table...")
        urllib.request.urlretrieve(biom_url, biom_path)
        print(f"Saved to {biom_path}")
    else:
        print(f"HMP1 OTU table already exists at {biom_path}")

    # HMP1 metadata
    meta_url = (
        "https://downloads.hmpdacc.org/dacc/hhs/genome/microbiome/"
        "wgs/analysis/hmqcp/v35/map_v35.txt"
    )
    meta_path = os.path.join(data_dir, "hmp1_metadata.tsv")

    if not os.path.exists(meta_path):
        print("Downloading HMP1 metadata...")
        urllib.request.urlretrieve(meta_url, meta_path)
        print(f"Saved to {meta_path}")
    else:
        print(f"HMP1 metadata already exists at {meta_path}")

    # Load and inspect
    table = load_table(biom_path)
    print(f"OTU table shape: {table.shape}")
    print(f"  Samples: {table.shape[1]}")
    print(f"  OTUs: {table.shape[0]}")

    return table


def download_agp(data_dir="data/raw/agp"):
    """Download American Gut Project data from Qiita.

    Downloads the deblurred BIOM table and sample metadata
    from Qiita study 10317.
    """
    os.makedirs(data_dir, exist_ok=True)

    study_id = 10317

    # Deblurred 150nt BIOM table
    biom_url = (
        "https://qiita.ucsd.edu/public_artifact_download/"
        "?artifact_id=77316"
    )
    biom_path = os.path.join(data_dir, "agp_otu_table.biom")

    if not os.path.exists(biom_path):
        print("Downloading AGP OTU table...")
        urllib.request.urlretrieve(biom_url, biom_path)
        print(f"Saved to {biom_path}")
    else:
        print(f"AGP OTU table already exists at {biom_path}")

    # Sample metadata
    meta_url = (
        f"https://qiita.ucsd.edu/public_download/"
        f"?data=sample_information&study_id={study_id}"
    )
    meta_path = os.path.join(data_dir, "agp_metadata.tsv")

    if not os.path.exists(meta_path):
        print("Downloading AGP metadata...")
        urllib.request.urlretrieve(meta_url, meta_path)
        print(f"Saved to {meta_path}")
    else:
        print(f"AGP metadata already exists at {meta_path}")

    return biom_path, meta_path


if __name__ == "__main__":
    download_hmp()
    download_agp()

#!/usr/bin/env python3
import math
import numpy as np
import faiss
from tqdm import tqdm
from datasets import load_dataset
import os

fair_co2_path = os.environ.get('FAIR_CO2')

# Fixed number of vectors to add per batch.
NUM_VECTORS_PER_BATCH = 100_000

def create_faiss_index_from_dataset(vectors, dim):
    """
    Create and populate a FAISS index using vectors from a Hugging Face dataset.
    
    Parameters:
      vectors (np.ndarray): Array of vectors with shape (total_vectors, dim).
      dim (int): Dimensionality of each vector.
      
    Returns:
      faiss.IndexIVFScalarQuantizer: The populated FAISS index.
    """
    total_vectors = vectors.shape[0]

    # --- FAISS Index Configuration ---
    C = 1
    nlists = C * int(np.sqrt(total_vectors))
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFScalarQuantizer(
        quantizer, dim, nlists, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT
    )

    # --- Index Training ---
    initial_train_size = int(total_vectors / 10)
    print(f"Training the FAISS index with {initial_train_size} vectors from the dataset...")
    index.train(vectors[:initial_train_size])

    # --- Adding Dataset Vectors ---
    print("Adding dataset vectors to the FAISS index in batches...")
    for i in tqdm(range(0, total_vectors, NUM_VECTORS_PER_BATCH), desc="Batches Processed", unit="batch"):
        batch = np.array(vectors[i : i + NUM_VECTORS_PER_BATCH])
        index.add(batch)
    return index

def main():
    dataset_name = "mohdumar/SPHERE_100M"
    streaming = False

    # --- Load the Hugging Face Dataset ---
    print(f"Loading Hugging Face dataset: {dataset_name} ...")
    dataset = load_dataset(dataset_name, split="train", streaming=streaming)

    if "vector" not in dataset.column_names:
        raise ValueError("The dataset does not contain a 'vector' column. Please verify the dataset fields.")

    vectors = np.array(dataset["vector"], dtype="float32")
    total_vectors = vectors.shape[0]
    print(f"Dataset loaded. Total vectors: {total_vectors}")

    if vectors.shape[1] != 768:
        raise ValueError(f"Provided dimension 768 does not match vector dimension {vectors.shape[1]} in the dataset.")

    # --- Create and Populate the FAISS Index ---
    index = create_faiss_index_from_dataset(vectors, 768)

    # Save index
    index_filename = f"100M_IVF16K_SQ8.faiss"
    out_file = os.path.join(fair_co2_path, "workloads/faiss/indices", index_filename)
    faiss.write_index(index, out_file)
    print(f"FAISS index saved to {out_file}")

if __name__ == "__main__":
    main()

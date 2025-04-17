import argparse
import pandas as pd
import faiss_lib
import datetime

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_time")
    parser.add_argument("--rounds")
    parser.add_argument("--sweep", action='store_true')
    args = parser.parse_args()
    start_time = int(args.start_time)
    rounds = int(args.rounds)
    embeddings_path = "/rag-carbon/faiss_indices/triviaqa_encodings.npy"
    ivf_index = "/rag-carbon/faiss_indices/100M_IVF16K_SQ8.faiss"
    hnsw_index = "/rag-carbon/faiss_indices/100M_HNSW.faiss"

    print(datetime.datetime.now())
    nprobe_depth = 128
    efsearch_depth = 192
    doc_cnt = 20
    num_queries = rounds
    if args.sweep:
        batch_size_list = [8, 16, 32, 64, 128, 256, 512, 1024]
        # mode_list = ["IVF", "HNSW"]
        mode_list = ["IVF"]
    else:
        batch_size_list = [64]
        mode_list = ["IVF"]

    faiss_lib.faiss_sweep(embeddings_path, ivf_index, hnsw_index, batch_size_list, nprobe_depth, efsearch_depth, doc_cnt, num_queries, mode_list, start_time)
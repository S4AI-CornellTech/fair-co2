import datetime
import numpy as np
import pandas as pd
from faiss import read_index
import time

def faiss_search(embeddings_array, index, doc_cnt, batch_size, num_queries, start_time):
    n_batch = 0
    query_start_list = []
    query_end_list = []
    query_time_list = []

    for start_idx in range(0, len(embeddings_array), batch_size):
        n_batch += 1
        # Query Arrray
        end_idx = min(start_idx + batch_size, len(embeddings_array))
        questions_array = embeddings_array[start_idx:end_idx]
        
        # Warmup Runs
        if start_idx == 0:
            for i in range(0, 50):
                distances, indices = index.search(questions_array, doc_cnt)
            print("Warmup runs completed")
            curr_time = time.time()
            sleep_time = start_time - curr_time
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        query_start = datetime.datetime.now()
        distances, indices = index.search(questions_array, doc_cnt)
        query_end = datetime.datetime.now()
        query_start_list.append(query_start)
        query_end_list.append(query_end)
        query_time = query_end - query_start
        query_time_list.append(query_time)
        print("Batch: " + str(n_batch) + ", time: " + str(query_time))
                        
        if n_batch >= num_queries:
            break

    return query_start_list, query_end_list, query_time_list

def faiss_sweep_batch_sizes(embeddings_array, index, batch_size_list, depth, doc_cnt, num_queries, mode, start_time):
    df = pd.DataFrame(columns=["mode", "batch_size", "query_start", "query_end", "query_time"])

    index = read_index(index)

    if mode == "IVF":
        index.nprobe = depth
    elif mode == "HNSW":
        index.hnsw.efSearch = depth
    else:
        print("Invalid mode")
        return False
    
    print("Index loaded")

    for batch_size in batch_size_list:
        print(f"Running {mode} with batch size {batch_size}")
        query_start_list, query_end_list, query_time_list = faiss_search(embeddings_array, index, doc_cnt, batch_size, num_queries, start_time)
        df_temp = pd.DataFrame({"mode": mode, "batch_size": batch_size, "query_start": query_start_list, "query_end": query_end_list, "query_time": query_time_list})
        df = pd.concat([df, df_temp], ignore_index=True)
        # Save df to csv
        if (mode == "IVF"):
            df.to_csv(f"/rag-carbon/data/faiss_times_ivf.csv", index=False)
        elif (mode == "HNSW"):
            df.to_csv(f"/rag-carbon/data/faiss_times_hnsw.csv", index=False)

def faiss_sweep(embeddings_path, ivf_index, hnsw_index, batch_size_list, nprobe_depth, efsearch_depth, doc_cnt, num_queries, mode_list, start_time):
    
    df = pd.DataFrame(columns=["mode", "batch_size", "query_start", "query_end", "query_time"])
    embeddings_array = np.load(embeddings_path)
    print("Embeddings loaded")

    for mode in mode_list:
        if mode == "IVF":
            depth = nprobe_depth
            index = ivf_index
        elif mode == "HNSW":
            depth = efsearch_depth
            index = hnsw_index
        else:
            print("Invalid mode")
            return False

        faiss_sweep_batch_sizes(embeddings_array, index, batch_size_list, depth, doc_cnt, num_queries, mode, start_time)
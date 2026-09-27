import time
import numpy as np
import scipy.sparse as sp
import joblib

print("Generating dummy data...")
# 1.7M S1s, 500k S23s
n_s1 = 1700000
n_s23 = 500000
vocab_size = 100000

print("Generating sparse matrices...")
s1_matrix = sp.random(n_s1, vocab_size, density=0.0001, format='csr')
s23_matrix = sp.random(n_s23, vocab_size, density=0.0001, format='csr')

def process_chunk(start_idx, chunk_size, s23_mat, s1_matT, top_k):
    end_idx = min(start_idx + chunk_size, s23_mat.shape[0])
    chunk = s23_mat[start_idx:end_idx]
    # Dot product
    sim = chunk.dot(s1_matT).toarray()
    
    # Get top K
    res = []
    for row_idx in range(sim.shape[0]):
        row = sim[row_idx]
        k = min(top_k, len(row))
        if k == 0: continue
        top_k_idx = np.argpartition(row, -k)[-k:]
        valid = top_k_idx[row[top_k_idx] > 0.10]
        for v in valid:
            res.append((start_idx + row_idx, v, row[v]))
    return res

if __name__ == '__main__':
    print("Running parallel dot product...")
    t0 = time.time()
    s1_matrixT = s1_matrix.T.tocsr()

    chunk_size = 5000
    chunks = list(range(0, n_s23, chunk_size))

    # Run on 4 jobs to be safe on memory
    results = joblib.Parallel(n_jobs=4, batch_size=1)(
        joblib.delayed(process_chunk)(start, chunk_size, s23_matrix, s1_matrixT, 5) 
        for start in chunks
    )

    print(f"Finished {n_s23} queries in {time.time()-t0:.1f}s")

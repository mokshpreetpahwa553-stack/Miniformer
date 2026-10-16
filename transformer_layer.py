from maths_attention import multihead_attention, positional_encoding
import numpy as np

d_model=64
np.random.seed(42)
# ONE big matrix per type (random ONCE)
Wq_total = np.random.rand(d_model, d_model) * 0.1  # (4,4)
Wk_total = np.random.rand(d_model, d_model) * 0.1  # (4,4)
Wv_total = np.random.rand(d_model, d_model) * 0.1  # (4,4)
Wo_total = np.random.rand(d_model, d_model) * 0.1  # (4,4)

#usage explanation in function ffnn
dff=d_model*4
wff1=np.random.rand(d_model,dff) # Expand 4→16
wff2=np.random.rand(dff,d_model) # Contract 16->4



def ffnn(Z):

    # FFNN processes each word individually to learn deeper meaning
    # Attention told us WHICH words relate, FFNN figures out WHAT that means
    # We expand 4→16 to give network more "scratch space" to find deeper patterns
    # We compress 16→4 to get back to original embedding size
    

    Z1=Z @ wff1

    Z2=np.maximum(0,Z1) #to make every -ve to 0

    return Z2 @ wff2


#function to calculate layer norm
def layer_norm(Z):

    #getting mean
    mean=np.mean(Z,axis=-1,keepdims=True) #axis=-1 will calculate mean across roeach row
    '''keepdims keeps 
    # Row 0: [4.50, 4.52, 2.82, 3.33] → avg = 3.79
    # Row 1: [3.86, 3.80, 3.36, 3.96] → avg = 3.75  
    # Row 2: [12.36,9.77,8.16,9.11] → avg = 9.85
    # Result shape: (3,1) = [[3.79], [3.75], [9.85]]
    '''
    #getting variance
    var=np.var(Z,axis=-1,keepdims=True)

    eps=1e-5
    return (Z-mean) / np.sqrt(var + eps)


#function that will calculate full transformer output
def encoding(X): 
    num_heads=4
    attn=multihead_attention(X,num_heads,Wq_total,Wv_total,Wk_total,Wo_total)

    #residual 1
    Residual1=attn + X

    L1=layer_norm(Residual1)
    F1=ffnn(L1)

    #residual 2
    Residual2 = L1  + F1

    encoding_out=layer_norm(Residual2)
    return encoding_out

#it is function where final transformer output is obtained
def transformer(X):

    #as there are total 6 encoding layer so thats why we loop 6 times
    for i in range(6):
        res=encoding(X)
        X=res

    # MEAN POOL: Make EVERY chunk same size for vector search!
    # Why? Vector search needs FIXED dimension across ALL chunks.
    # WITHOUT POOL: chunks have different rows → crash!
    # Different chunks have different lengths (3 tokens vs 100 tokens).
    # FAISS/HNSW indexes (N_chunks, fixed_dim) matrices only.
    #It makes output of every chunk of same size
    # EXAMPLE 1: Short chunk (3 tokens)
    # X after layers: [[0.12, 0.05, 0.28, 0.42],  # "attention"
    #                  [0.48, 0.12,-0.08, 0.58],  # "scales"  
    #                  [0.25, 0.08, 0.15, 0.49]]  # "quadratic"
    # Mean(axis=0): [(0.12+0.48+0.25)/3, (0.05+0.12+0.08)/3, ...]
    # Result: [0.283, 0.083, 0.117, 0.497]  # Shape: (4,)
    #
    # EXAMPLE 2: Long chunk (10 tokens) 
    # X after layers: [[0.11, 0.03, 0.31, 0.41],  # token1
    #                  [0.47, 0.11,-0.09, 0.57],  # token2
    #                  ... 8 more rows ...
    #                  [0.29, 0.09, 0.18, 0.51]]  # token10
    # Mean(axis=0): average ALL 10 rows → [0.287, 0.087, 0.122, 0.502]  # Shape: (4,) SAME!
    #
    # MAGIC: Regardless of num_tokens (3 or 1000), output ALWAYS (d_model,)
    # Enables: np.stack([chunk1_vec, chunk2_vec, ...]) → (5000_chunks, 4) for FAISS
    pool=np.mean(X,axis=0) # axis=0 = average across words (down columns)
    return pool




        


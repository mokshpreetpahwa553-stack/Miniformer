#THIS IS MATHS WE NEED TO COMPUTE ATTENTION

import numpy as np


#function to compute softmax, we can compute softmax of whole matrix i.e for every word
#but now we will calculate for each row(word) seperately
def softmax(score_row):
    #step1 calculating e^ for every point
    exp_score=np.exp(score_row)

    #step2 calculating total
    total=np.sum(exp_score)

    #step3 dividing each point by sum
    weight=exp_score/total #is is new array that contain weigth of each word

    return weight


#this function will give positional encoding
def positional_encoding(row,cols): #we give dimensions of required matrix
    PE=np.zeros((row,cols))  #we create zero matrix with same dimensions 

    for pos in range(row):
        for i in range(0,cols,2):
            PE[pos,i] = np.sin(pos / 10000**(i/cols))
            PE[pos,i+1] = np.cos(pos/10000**(i/cols))
    
    return PE


def multihead_attention(X,num_heads,Wq,Wv,Wk,Wo): #X is embedded matrix

    #getting matrix to use
    X=X+positional_encoding(X.shape[0],X.shape[1])

    d_model=X.shape[1]
    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
    d_head=X.shape[1]//num_heads # split total embedding size into equal parts, so each head gets d_head dimension

    all_head=[] #this list will store all heads output

    #now computing query,key and value matrices
    Q=X @ Wq
    V=X @ Wv
    K=X @ Wk

    Q_heads = np.split(Q, num_heads, axis=1) # split with columns 
    V_heads=np.split(V, num_heads, axis=1)
    K_heads=np.split(K, num_heads, axis=1)
    Wo_heads=np.split(Wo,num_heads,axis=0)
    for i in range(num_heads):  #computation for each head

        A= Q_heads[i] @ np.transpose(K_heads[i]) / np.sqrt(K_heads[i].shape[1]) #we did transpose of k bcs Q and K are 3x1 and 3x1
        #matrix so product was not possible so we transposed K to make it 2x3. K.shape[1] will give
        #number of columns in matrix 
        #self attention output
        Z=np.zeros_like(X)  #this will create exact array like X but with just 0 in it

        #this loop makes each token look at every other token
        for x in range(A.shape[0]): #A is X[0] x X[0] matrix

        ## Wo: converts 2‑dim attention output back to 4‑dim embeddings (same size as X)
        # regardless of the specific random values
            Z[x]=(softmax(A[x]) @ V_heads[i] ) @ Wo_heads[i]
        

        all_head.append(Z) 

    #no we have list(all_head) with all head final outputs now we need to convert it in matrix
    z_heads = np.stack(all_head, axis=0)

    # all_head: list of per-head outputs, each of shape (X.shape[0], num_heads),
    # where X.shape[0] = number of tokens in the chunk (sequence length),
    # and num_heads = number of attention heads (each column is one head’s features).
    # Stacking gives z_heads with shape (num_heads, X.shape[0], num_heads).
    # We then average over the head dimension (axis=0), so each token’s
    # final vector is the mean of its representations from all heads
    Z = z_heads.mean(axis=0)

    #np.random.rand(Z[1],4) similarly works as WO to amke matrix back in 3x4
    final_output= Z @ Wo

    return final_output


        


            

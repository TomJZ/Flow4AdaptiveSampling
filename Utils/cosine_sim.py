import numpy as np
from numpy.linalg import norm

def cosine_similarity(A, B):
    """
    Args:
        A: time series of the first flow field
        B: time series of the second flow field
    Returns:
        Averaged cosine similarity between two flow field
    """
    print("shape of input tensor is", A.shape)
    A_ = np.swapaxes(A, 0, 1).reshape(2, -1)
    B_ = np.swapaxes(B, 0, 1).reshape(2, -1)
    print("shape of reshaped tensor is", A_.shape)
    vec_num = A_.shape[1]
    cos_dist = np.zeros(vec_num)

    for i in range(vec_num):
        cos_dist[i] = np.dot(A_[:, i], B_[:, i])/(norm(A_[:, i] * norm(B_[:, i])))

    ave_cos_dist = np.mean(cos_dist)
    return ave_cos_dist


if __name__ == "__main__":
    A = np.random.rand(50, 2, 30, 30)
    B = np.random.rand(50, 2, 30, 30)
    result = cosine_similarity(A, B)
    print("Averaged Cosine Similarity is: ", result)
import numpy as np
from sklearn.preprocessing import normalize

def normalize_data(x):
    n_samples, *a = x.shape
    x = x.reshape([1, -1])
    normalized_x = normalize(x, axis=0)
    a.insert(0, n_samples)
    return normalized_x.reshape(a)

if __name__ == "__main__":
    # data = np.load("../Data/Processed/chaotic_40by40_vorticity.npy")
    # data = normalize_data(data)
    # with open("../Data/Processed/chaotic_40by40_vorticity_normed.npy", "wb") as f:
    #     np.save(f, data)

    data = np.load("../Data/Processed/noaa_flow_field.npy")
    data = normalize_data(data)
    with open("../Data/Processed/noaa_flow_field_normed.npy", "wb") as f:
        np.save(f, data)
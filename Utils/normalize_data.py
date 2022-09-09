import numpy as np
from sklearn import preprocessing

def normalize_data(X_train,scaler="minmax"):
    bs, *a = X_train.shape
    X_train = X_train.reshape([bs, -1])
    if scaler == "minmax":
        scaler = preprocessing.MinMaxScaler().fit(X_train)
    elif scaler == "standard":
        scaler = preprocessing.StandardScaler().fit(X_train)

    X_scaled = scaler.transform(X_train)
    a.insert(0, bs)
    X_scaled = X_scaled.reshape(a)
    print("normalized shape: ", X_scaled.shape)
    print("normalized shape: ", X_scaled.shape)
    print("original data mean: {0}, std: {1}".format(np.mean(X_train), np.std(X_train)))
    print("normalized data mean: {0}, std: {1}".format(np.mean(X_scaled), np.std(X_scaled)))
    return X_scaled


if __name__ == "__main__":
    data = np.load("../Data/TrainingDataProcessed/chaotic_40by40_flow_field.npy")
    data = normalize_data(data, "standard")
    with open("../Data/TrainingDataProcessed/chaotic_40by40_flow_field_standard_scaled.npy", "wb") as f:
        np.save(f, data)

    # data = np.load("../Data/TrainingDataProcessed/chaotic_40by40_vorticity.npy")
    # data = normalize_data(data, "standard")
    # with open("../Data/TrainingDataProcessed/chaotic_40by40_vorticity_standard_scaled.npy", "wb") as f:
    #     np.save(f, data)
    #
    # data = np.load("../Data/TrainingDataProcessed/noaa_flow_field.npy")
    # data = normalize_data(data, "standard")
    # with open("../Data/TrainingDataProcessed/noaa_flow_field_standard_scaled.npy", "wb") as f:
    #     np.save(f, data)
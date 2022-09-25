package data.mnist;

import data.network_train.NNData1D;

public abstract class DataLoader1D {
    public abstract NNData1D getNextTrainData(int sizeBatch);

    public abstract NNData1D getNextTestData(int sizeBatch);
}

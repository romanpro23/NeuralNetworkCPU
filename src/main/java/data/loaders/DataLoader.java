package data.loaders;

import data.network_train.NNData;
import data.network_train.NNData1D;

public abstract class DataLoader {
    public abstract NNData getNextTrainData(int sizeBatch);

    public abstract NNData getNextTestData(int sizeBatch);
}

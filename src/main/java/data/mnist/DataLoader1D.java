package data.mnist;

public abstract class DataLoader1D {
    public abstract NNData1D getNextTrainData(int sizeBatch);

    public abstract NNData1D getNextTestData(int sizeBatch);
}

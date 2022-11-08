package data.network_train;

import nnarrays.NNArray;
import nnarrays.NNVector;

public abstract class NNData {
    public abstract NNArray[] getInput();

    public abstract NNArray[] getOutput();
}

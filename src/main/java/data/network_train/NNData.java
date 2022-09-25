package data.network_train;

import nnarrays.NNArray;
import nnarrays.NNVector;

public class NNData {
    NNArray[] input;
    NNArray[] output;

    public NNArray[] getInput() {
        return input;
    }

    public NNArray[] getOutput() {
        return output;
    }
}

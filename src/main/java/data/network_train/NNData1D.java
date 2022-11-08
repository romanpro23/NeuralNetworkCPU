package data.network_train;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;
import nnarrays.NNArray;
import nnarrays.NNVector;

@AllArgsConstructor
public class NNData1D extends NNData {
    NNVector[] inputV;
    NNVector[] outputV;

    public NNArray[] getInput() {
        return inputV;
    }

    public NNArray[] getOutput() {
        return outputV;
    }
}
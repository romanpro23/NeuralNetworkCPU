package data.network_train;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;
import nnarrays.NNArray;
import nnarrays.NNMatrix;
import nnarrays.NNVector;

@AllArgsConstructor
public class NNData2D extends NNData {
    NNMatrix[] inputM;
    NNVector[] outputV;

    public NNArray[] getInput() {
        return inputM;
    }

    public NNArray[] getOutput() {
        return outputV;
    }
}
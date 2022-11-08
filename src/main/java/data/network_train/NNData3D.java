package data.network_train;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;
import nnarrays.NNArray;
import nnarrays.NNMatrix;
import nnarrays.NNTensor;
import nnarrays.NNVector;

@AllArgsConstructor
public class NNData3D extends NNData {
    NNTensor[] inputT;
    NNVector[] outputV;

    public NNArray[] getInput() {
        return inputT;
    }

    public NNArray[] getOutput() {
        return outputV;
    }
}
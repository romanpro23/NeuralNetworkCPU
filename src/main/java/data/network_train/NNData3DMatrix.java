package data.network_train;

import lombok.AllArgsConstructor;
import nnarrays.NNArray;
import nnarrays.NNMatrix;
import nnarrays.NNTensor;
import nnarrays.NNVector;

@AllArgsConstructor
public class NNData3DMatrix extends NNData {
    NNTensor[] inputT;
    NNMatrix[] outputV;

    public NNArray[] getInput() {
        return inputT;
    }

    public NNArray[] getOutput() {
        return outputV;
    }
}
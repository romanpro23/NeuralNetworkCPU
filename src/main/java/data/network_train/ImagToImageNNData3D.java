package data.network_train;

import lombok.AllArgsConstructor;
import nnarrays.NNArray;
import nnarrays.NNTensor;
import nnarrays.NNVector;

@AllArgsConstructor
public class ImagToImageNNData3D extends NNData {
    NNTensor[] inputT;
    NNTensor[] outputT;

    public NNArray[] getInput() {
        return inputT;
    }

    public NNArray[] getOutput() {
        return outputT;
    }
}
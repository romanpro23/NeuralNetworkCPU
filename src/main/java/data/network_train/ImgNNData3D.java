package data.network_train;

import lombok.AllArgsConstructor;
import nnarrays.NNArray;
import nnarrays.NNTensor;
import nnarrays.NNVector;

@AllArgsConstructor
public class ImgNNData3D extends NNData {
    NNTensor[] input;
    NNTensor[] output;

    public NNArray[] getInput() {
        return input;
    }

    public NNArray[] getOutput() {
        return output;
    }
}
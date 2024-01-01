package data.loaders;

import lombok.Getter;
import nnarrays.NNMatrix;
import nnarrays.NNTensor;
import nnarrays.NNVector;

public class ImageData3DMatrix {
    @Getter
    private NNTensor inputs;
    @Getter
    private NNMatrix outputs;

    public ImageData3DMatrix(NNTensor inputs, NNMatrix outputs) {
        this.inputs = inputs;
        this.outputs = outputs;
    }
}

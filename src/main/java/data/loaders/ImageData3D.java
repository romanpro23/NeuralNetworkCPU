package data.loaders;

import lombok.Getter;
import nnarrays.NNTensor;
import nnarrays.NNVector;

public class ImageData3D {
    @Getter
    private NNTensor inputs;
    @Getter
    private NNVector outputs;

    public ImageData3D(NNTensor inputs, NNVector outputs) {
        this.inputs = inputs;
        this.outputs = outputs;
    }
}

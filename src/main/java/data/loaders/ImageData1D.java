package data.loaders;

import lombok.Getter;
import nnarrays.NNVector;

public class ImageData1D {
    @Getter
    private NNVector inputs;
    @Getter
    private NNVector outputs;

    public ImageData1D(NNVector inputs, NNVector outputs) {
        this.inputs = inputs;
        this.outputs = outputs;
    }
}

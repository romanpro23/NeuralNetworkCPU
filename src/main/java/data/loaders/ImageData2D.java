package data.loaders;

import lombok.Getter;
import nnarrays.NNMatrix;
import nnarrays.NNVector;

public class ImageData2D {
    @Getter
    private NNMatrix inputs;
    @Getter
    private NNVector outputs;

    public ImageData2D(NNMatrix inputs, NNVector outputs) {
        this.inputs = inputs;
        this.outputs = outputs;
    }
}

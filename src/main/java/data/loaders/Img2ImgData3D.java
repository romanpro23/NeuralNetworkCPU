package data.loaders;

import lombok.Getter;
import nnarrays.NNTensor;
import nnarrays.NNVector;

public class Img2ImgData3D {
    @Getter
    private NNTensor inputs;
    @Getter
    private NNTensor outputs;

    public Img2ImgData3D(NNTensor inputs, NNTensor outputs) {
        this.inputs = inputs;
        this.outputs = outputs;
    }
}

package neural_network.optimizers;

import lombok.Getter;
import nnarrays.NNArray;

public class DataOptimize {
    @Getter
    private NNArray weight;
    @Getter
    private NNArray derWeight;
    @Getter
    private NNArray[] additionParam;

    public DataOptimize(NNArray weight, NNArray derWeight, NNArray[] additionParam) {
        this.weight = weight;
        this.derWeight = derWeight;
        this.additionParam = additionParam;
    }
}

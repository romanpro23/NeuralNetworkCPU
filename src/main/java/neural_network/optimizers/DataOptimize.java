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
    @Getter
    private String name;

    public DataOptimize(NNArray weight, NNArray derWeight, NNArray[] additionParam) {
        this.weight = weight;
        this.derWeight = derWeight;
        this.additionParam = additionParam;
    }

    public DataOptimize(NNArray weight, NNArray derWeight, NNArray[] additionParam, String name) {
        this.weight = weight;
        this.derWeight = derWeight;
        this.additionParam = additionParam;
        this.name = name;
    }
}

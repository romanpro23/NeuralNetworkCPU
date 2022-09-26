package neural_network.optimizers;

import nnarrays.NNArray;

public class DataOptimize {
    private NNArray weight;
    private NNArray derWeight;
    private NNArray[] additionParam;

    public DataOptimize(NNArray weight, NNArray derWeight, NNArray[] additionParam) {
        this.weight = weight;
        this.derWeight = derWeight;
        this.additionParam = additionParam;
    }
}

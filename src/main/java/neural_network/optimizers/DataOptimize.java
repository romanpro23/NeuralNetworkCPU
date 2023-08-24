package neural_network.optimizers;

import lombok.Getter;
import nnarrays.NNArray;
import utilities.CublasUtil;

public class DataOptimize {
    @Getter
    private NNArray weight;
    @Getter
    private NNArray derWeight;
    @Getter
    private NNArray[] additionParam;

    @Getter
    private CublasUtil.Matrix weight_gpu;
    @Getter
    private CublasUtil.Matrix derWeight_gpu;
    @Getter
    private CublasUtil.Matrix[] additionParam_gpu;

    public DataOptimize(NNArray weight, NNArray derWeight, NNArray[] additionParam) {
        this.weight = weight;
        this.derWeight = derWeight;
        this.additionParam = additionParam;
    }

    public DataOptimize(CublasUtil.Matrix weight, CublasUtil.Matrix derWeight, CublasUtil.Matrix[] additionParam) {
        this.weight_gpu = weight;
        this.derWeight_gpu = derWeight;
        this.additionParam_gpu = additionParam;
    }
}

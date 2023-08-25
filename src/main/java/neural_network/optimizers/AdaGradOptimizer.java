package neural_network.optimizers;

import nnarrays.NNArray;
import utilities.CublasUtil;

public class AdaGradOptimizer extends Optimizer {
    /**
     * AdaGrad
     * θ(t) = θ(t-1) + dw(t) * dw(t)
     * w(t) = w(t-1) - lr * dw(t) / sqrt(θ(t))
     */
    private final float learningRate;

    public AdaGradOptimizer() {
        this(0.001);
    }

    public AdaGradOptimizer(double learningRate) {
        this.learningRate = (float) learningRate;
        this.countParam = 1;
    }

    @Override
    public void updateWeight(NNArray weight, NNArray deltaWeight, NNArray[] additionParam) {
        additionParam[0].addPow2(deltaWeight);
        weight.subDivSqrt(deltaWeight,additionParam[0], learningRate);
        deltaWeight.clear();
    }

    @Override
    protected void updateWeight(CublasUtil.Matrix weight_gpu, CublasUtil.Matrix deltaWeight_gpu, CublasUtil.Matrix[] additionParam_gpu) {

    }
}

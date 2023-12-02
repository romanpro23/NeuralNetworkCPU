package neural_network.optimizers;

import nnarrays.NNArray;

import java.util.Arrays;

public class SGDOptimizer extends Optimizer {
    /**
     * SGD
     * w(t) = w(t-1) - lr * dw(t)
     */
    private float learningRate;

    public SGDOptimizer(double learningRate) {
        super();
        this.learningRate = (float) learningRate;
        this.countParam = 0;
    }

    @Override
    public void updateWeight(NNArray weight, NNArray deltaWeight, NNArray[] additionParam) {
        weight.sub(deltaWeight.mul(learningRate));
        deltaWeight.clear();
    }
}

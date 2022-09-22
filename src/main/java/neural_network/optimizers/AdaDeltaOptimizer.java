package neural_network.optimizers;

import nnarrays.NNArray;

public class AdaDeltaOptimizer extends Optimizer {
    /**
     * dr - decay rate
     * lr - learning rate
     * AdaDelta
     * θ(t) = dr * θ(t-1) + (1 - dr) * dw(t) * dw(t)
     * w(t) = w(t-1) - lr * dw(t) / sqrt(θ(t))
     */
    private final float decayRate;
    private final float learningRate;

    public AdaDeltaOptimizer() {
        this(0.001, 0.9);
    }

    public AdaDeltaOptimizer(double learningRate) {
        this(learningRate, 0.9);
    }

    public AdaDeltaOptimizer(double learningRate, double decayRate) {
        this.decayRate = (float) decayRate;
        this.learningRate = (float) learningRate;
        this.countParam = 1;
    }

    @Override
    public void updateWeight(NNArray weight, NNArray deltaWeight, NNArray[] additionParam) {
        additionParam[0].momentumPow2(deltaWeight, decayRate);
        weight.subDivSqrt(deltaWeight, additionParam[0], learningRate);
        deltaWeight.clear();
    }
}

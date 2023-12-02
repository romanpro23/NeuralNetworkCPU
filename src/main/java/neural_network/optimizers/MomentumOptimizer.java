package neural_network.optimizers;

import nnarrays.NNArray;

public class MomentumOptimizer extends Optimizer {
    /**
     * rt - retention rate
     * Momentum
     * θ(t) = rt * θ(t-1) + (1 - rt) * dw(t)
     * w(t) = w(t-1) - lr * θ(t)
     */
    private final float learningRate;
    private final float retentionRate;

    public MomentumOptimizer(double learningRate, double retentionRate) {
        super();
        this.learningRate = (float) learningRate;
        this.retentionRate = (float) retentionRate;
        this.countParam = 1;
    }

    @Override
    public void updateWeight(NNArray weight, NNArray deltaWeight, NNArray[] additionParam) {
        additionParam[0].momentum(deltaWeight, retentionRate);
        weight.subAndMul(additionParam[0], learningRate);
        deltaWeight.clear();
    }
}

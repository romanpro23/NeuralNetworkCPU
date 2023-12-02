package neural_network.optimizers;

import nnarrays.NNArray;

public class QHMomentumOptimizer extends Optimizer {
    /**
     * rt - retention rate
     * Momentum
     * θ(t) = rt * θ(t-1) + (1 - rt) * dw(t)
     * w(t) = w(t-1) - lr * (v * θ(t) + (1 - v) * dw(t))
     */
    private final float learningRate;
    private final float retentionRate;
    private final float v;

    public QHMomentumOptimizer(double learningRate) {
        this(learningRate, 0.999, 0.7);
    }

    public QHMomentumOptimizer(double learningRate, double retentionRate) {
        this(learningRate, retentionRate, 0.7);
    }

    public QHMomentumOptimizer(double learningRate, double retentionRate, double v) {
        super();
        this.learningRate = (float) learningRate;
        this.retentionRate = (float) retentionRate;
        this.v = (float) v;
        this.countParam = 1;
    }

    @Override
    public void updateWeight(NNArray weight, NNArray deltaWeight, NNArray[] additionParam) {
        additionParam[0].momentum(deltaWeight, retentionRate);
        weight.subAndMulQH(additionParam[0], deltaWeight, learningRate, v);
        deltaWeight.clear();
    }
}

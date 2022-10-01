package neural_network.optimizers;

import nnarrays.NNArray;

public class AdaBeliefOptimizer extends Optimizer {
    /**
     * AdaBelief
     * m(t) = b1 * m(t-1) + (1 - b1) * dw(t)
     * v(t) = b2 * v(t-1) + (1 - b2) * (dw(t) - m(t)) ^ 2
     * m_(t) = m(t) / (1 - b1^t)
     * v_(t) = v(t) / (1 - b2^t)
     * w(t) = w(t-1) - lr * m_(t) / sqrt(v_(t))
     */
    private final float beta1;
    private final float beta2;
    private final float learningRate;

    private float b1t;
    private float b2t;

    @Override
    public void update() {
        t++;
        b1t = (float) (1 - Math.pow(beta1, t));
        b2t = (float) (1 - Math.pow(beta2, t));
        super.update();
    }

    public AdaBeliefOptimizer() {
        this(0.9, 0.999, 0.001);
    }

    public AdaBeliefOptimizer(double beta1, double beta2, double learningRate) {
        this.beta1 = (float) beta1;
        this.beta2 = (float) beta2;
        this.learningRate = (float) learningRate;
        t = 0;
        countParam = 2;
    }

    public AdaBeliefOptimizer(double learningRate) {
        this(0.9, 0.999, learningRate);
    }

    @Override
    public void updateWeight(NNArray weight, NNArray deltaWeight, NNArray[] additionParam) {
        additionParam[0].momentum(deltaWeight, beta1);
        momentumSubPow2(additionParam[1], deltaWeight, additionParam[0]);

        weight.subDivSqrtNorm(additionParam[0], additionParam[1], learningRate, b1t, b2t);
        deltaWeight.clear();
    }

    private void momentumSubPow2(NNArray v, NNArray dw, NNArray m) {
        final float dr = 1 - beta2;
        float sub;
        for (int i = 0; i < v.size(); i++) {
            sub = dw.getData()[i] - m.getData()[i];
            v.getData()[i] = beta2 * v.getData()[i] + dr * sub * sub;
        }
    }
}

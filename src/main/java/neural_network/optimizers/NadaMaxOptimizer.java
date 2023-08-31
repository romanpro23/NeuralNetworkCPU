package neural_network.optimizers;

import nnarrays.NNArray;

public class NadaMaxOptimizer extends Optimizer {
    /**
     * NadaMax
     * m(t) = b1 * m(t-1) + (1 - b1) * dw(t)
     * m_(t) = m(t) / (1 - b1^t)
     * v(t) = max( b2 * v(t-1), abs(dw(t)))
     * w(t) = w(t-1) - lr * (m_(t) * b1 + (1 - b1) * dw(t) / (1 - b1^t)) / v_(t)
     */
    private final float beta1;
    private final float beta2;
    private final float learningRate;

    private float b1t;

    @Override
    public void update() {
        t++;
        b1t = (float) (1 - Math.pow(beta1, t));
        super.update();
    }

    public NadaMaxOptimizer() {
        this(0.9, 0.999, 0.001);
    }

    public NadaMaxOptimizer(double beta1, double beta2, double learningRate) {
        super();
        this.beta1 = (float) beta1;
        this.beta2 = (float) beta2;
        this.learningRate = (float) learningRate;
        t = 0;
        countParam = 2;
    }

    public NadaMaxOptimizer(double learningRate) {
        this(0.9, 0.999, learningRate);
    }

    @Override
    public void updateWeight(NNArray weight, NNArray deltaWeight, NNArray[] additionParam) {
        if (t == 1) {
            additionParam[1].fill(Float.MIN_VALUE);
        }

        additionParam[0].momentum(deltaWeight, beta1);
        max(additionParam[1], deltaWeight);

        weight.subDivNormNesterov(additionParam[0], additionParam[1], deltaWeight, learningRate, beta1, b1t);
        deltaWeight.clear();
    }

    private void max(NNArray v, NNArray dw) {
        for (int i = 0; i < v.size(); i++) {
            v.getData()[i] = Math.max(beta2 * v.getData()[i], Math.abs(dw.getData()[i]));
        }
    }
}

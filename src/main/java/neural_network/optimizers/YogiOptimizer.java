package neural_network.optimizers;

import nnarrays.NNArray;

public class YogiOptimizer extends Optimizer {
    /**
     * Adam
     * m(t) = b1 * m(t-1) + (1 - b1) * dw(t)
     * v(t) = v(t-1) - (1 - b2) * sign(v(t-1) - dw(t) ^ 2) * dw(t) ^ 2
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
        b1t = (float) (1.0f - Math.pow(beta1, t));
        b2t = (float) (1.0f - Math.pow(beta2, t));
        super.update();
    }

    public YogiOptimizer() {
        this(0.9, 0.999, 0.001);
    }

    public YogiOptimizer(double beta1, double beta2, double learningRate) {
        super();
        this.beta1 = (float) beta1;
        this.beta2 = (float) beta2;
        this.learningRate = (float) learningRate;
        t = 0;
        countParam = 2;
    }

    public YogiOptimizer(double learningRate) {
        this(0.9, 0.999, learningRate);
    }

    @Override
    public void updateWeight(NNArray weight, NNArray deltaWeight, NNArray[] additionParam) {
        additionParam[0].momentum(deltaWeight, beta1);
        additionParam[1].momentumPow2Sign(deltaWeight, beta2);

        weight.subDivSqrtNorm(additionParam[0], additionParam[1], learningRate, b1t, b2t);
        deltaWeight.clear();
    }
}

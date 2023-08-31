package neural_network.optimizers;

import nnarrays.NNArray;

public class AdamNormOptimizer extends Optimizer {
    /**
     * AdamNorm
     * dw_norm = abs(dw(t))
     * e(t) = y * e(t-1) + (1 - y) * dw_norm
     * s(t) = dw(t)
     * if e(t) > dw_norm:
     *      s(t) = (e(t) / dw_norm) * dw(t)
     * m(t) = b1 * m(t-1) + (1 - b1) * s(t)
     * v(t) = b2 * v(t-1) + (1 - b2) * dw(t) ^ 2
     * m_(t) = m(t) / (1 - b1^t)
     * v_(t) = v(t) / (1 - b2^t)
     * w(t) = w(t-1) - lr * m_(t) / sqrt(v_(t))
     */
    private final float beta1;
    private final float beta2;
    private final float gamma;
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

    public AdamNormOptimizer() {
        this(0.9, 0.999, 0.001, 95);
    }

    public AdamNormOptimizer(double beta1, double beta2, double learningRate, double gamma) {
        super();
        this.beta1 = (float) beta1;
        this.beta2 = (float) beta2;
        this.learningRate = (float) learningRate;
        this.gamma = (float) gamma;
        t = 0;
        countParam = 3;
    }

    public AdamNormOptimizer(double learningRate) {
        this(0.9, 0.999, learningRate, 0.95);
    }

    @Override
    public void updateWeight(NNArray weight, NNArray deltaWeight, NNArray[] additionParam) {
        additionParam[2].momentumAbs(deltaWeight, gamma);
        additionParam[0].momentumNorm(deltaWeight, additionParam[2], beta1);
        additionParam[1].momentumPow2(deltaWeight, beta2);

        weight.subDivSqrtNorm(additionParam[0], additionParam[1], learningRate, b1t, b2t);
        deltaWeight.clear();
    }
}

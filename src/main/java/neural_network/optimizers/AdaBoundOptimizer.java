package neural_network.optimizers;

import nnarrays.NNArray;
import utilities.CublasUtil;

public class AdaBoundOptimizer extends Optimizer {
    /**
     * Adam
     * m(t) = b1 * m(t-1) + (1 - b1) * dw(t)
     * v(t) = b2 * v(t-1) + (1 - b2) * dw(t) ^ 2
     * m_(t) = m(t) / (1 - b1^t)
     * lr(t) = clip(lr * sqrt(v(t) / (1 - b2^t)), 0.1 - 0.1 / (1 - b2) ^ (t + 1), 0.1 + 0.1 / (1 - b2) ^ t)
     * w(t) = w(t-1) - m_(t) * lr(t))
     */
    private final float beta1;
    private final float beta2;
    private final float learningRate;

    private float b1t;
    private float b2t;
    private float eta_l;
    private float eta_u;

    @Override
    public void update() {
        t++;
        b1t = (float) (1.0f - Math.pow(beta1, t));
        b2t = (float) (1.0f - Math.pow(beta2, t));
        eta_l = (float) (0.1f - 0.1f / (Math.pow(1 - beta2, t + 1)));
        eta_u = (float) (0.1f + 0.1f / (Math.pow(1 - beta2, t)));
        super.update();
    }

    public AdaBoundOptimizer() {
        this(0.9, 0.999, 0.001);
    }

    public AdaBoundOptimizer(double beta1, double beta2, double learningRate) {
        this.beta1 = (float) beta1;
        this.beta2 = (float) beta2;
        this.learningRate = (float) learningRate;
        t = 0;
        countParam = 2;
    }

    public AdaBoundOptimizer(double learningRate) {
        this(0.9, 0.999, learningRate);
    }

    @Override
    public void updateWeight(NNArray weight, NNArray deltaWeight, NNArray[] additionParam) {
        additionParam[0].momentum(deltaWeight, beta1);
        additionParam[1].momentumPow2(deltaWeight, beta2);

        weight.subDivSqrtNormClip(additionParam[0], additionParam[1], learningRate, b1t, b2t, eta_l, eta_u);
        deltaWeight.clear();
    }

    @Override
    protected void updateWeight(CublasUtil.Matrix weight_gpu, CublasUtil.Matrix deltaWeight_gpu, CublasUtil.Matrix[] additionParam_gpu) {

    }
}

package neural_network.optimizers;

import nnarrays.NNArray;
import utilities.CublasUtil;

public class AdamInjectOptimizer extends Optimizer {
    /**
     * AdamInject
     * m(t) = b1 * m(t-1) + (1 - b1) * (dw(t) + delta_w(t-1) * dw(t) ^ 2) / k
     * v(t) = b2 * v(t-1) + (1 - b2) * dw(t) ^ 2
     * m_(t) = m(t) / (1 - b1^t)
     * v_(t) = v(t) / (1 - b2^t)
     * w(t) = w(t-1) - lr * m_(t) / sqrt(v_(t))
     */
    private final float beta1;
    private final float beta2;
    private final float learningRate;
    private final float k;

    private float b1t;
    private float b2t;

    @Override
    public void update() {
        t++;
        b1t = (float) (1.0f - Math.pow(beta1, t));
        b2t = (float) (1.0f - Math.pow(beta2, t));
        super.update();
    }

    public AdamInjectOptimizer() {
        this(0.9, 0.999, 0.001, 2);
    }

    public AdamInjectOptimizer(double beta1, double beta2, double learningRate, double k) {
        this.beta1 = (float) beta1;
        this.beta2 = (float) beta2;
        this.learningRate = (float) learningRate;
        this.k = (float) k;
        t = 0;
        countParam = 3;
    }

    public AdamInjectOptimizer(double learningRate) {
        this(0.9, 0.999, learningRate, 2);
    }

    @Override
    public void updateWeight(NNArray weight, NNArray deltaWeight, NNArray[] additionParam) {
        if (t == 1) {
            additionParam[0].momentum(deltaWeight, beta1);
        } else {
            additionParam[0].momentumInject(deltaWeight, additionParam[2], beta1, k);
        }
        additionParam[1].momentumPow2(deltaWeight, beta2);

        additionParam[2].clear();
        additionParam[2].deltaSubDivSqrtNorm(additionParam[0], additionParam[1], learningRate, b1t, b2t);
        weight.sub(additionParam[2]);
        deltaWeight.clear();
    }

    @Override
    protected void updateWeight(CublasUtil.Matrix weight_gpu, CublasUtil.Matrix deltaWeight_gpu, CublasUtil.Matrix[] additionParam_gpu) {

    }
}

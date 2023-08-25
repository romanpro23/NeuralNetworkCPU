package neural_network.optimizers;

import nnarrays.NNArray;
import utilities.CublasUtil;

public class QHAdamOptimizer extends Optimizer {
    /**
     * QHAdam
     * m(t) = b1 * m(t-1) + (1 - b1) * dw(t)
     * v(t) = b2 * v(t-1) + (1 - b2) * dw(t) ^ 2
     * m_(t) = m(t) / (1 - b1^t)
     * v_(t) = v(t) / (1 - b2^t)
     * w(t) = w(t-1) - lr * ((1 - v1) * dw(t) + v1 * m_(t)) / sqrt((1 - v2) * dw(t)* dw(t) + v2 * v_(t))
     */
    private final float beta1;
    private final float beta2;
    private final float v1;
    private final float v2;
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

    public QHAdamOptimizer() {
        this(0.9, 0.999, 0.7, 1,  0.001);
    }

    public QHAdamOptimizer(double beta1, double beta2, double learningRate) {
        this(beta1, beta2, 0.7, 1,  learningRate);
    }

    public QHAdamOptimizer(double beta1, double beta2, double v1, double v2, double learningRate) {
        this.beta1 = (float) beta1;
        this.beta2 = (float) beta2;
        this.learningRate = (float) learningRate;
        this.v1 = (float) v1;
        this.v2 = (float) v2;
        t = 0;
        countParam = 2;
    }

    public QHAdamOptimizer(double learningRate) {
        this(0.9, 0.999, 0.7, 1, learningRate);
    }

    @Override
    public void updateWeight(NNArray weight, NNArray deltaWeight, NNArray[] additionParam) {
        additionParam[0].momentum(deltaWeight, beta1);
        additionParam[1].momentumPow2(deltaWeight, beta2);

        weight.subDivSqrtNormQH(deltaWeight,additionParam[0], additionParam[1], learningRate, b1t, b2t, v1, v2);
        deltaWeight.clear();
    }

    @Override
    protected void updateWeight(CublasUtil.Matrix weight_gpu, CublasUtil.Matrix deltaWeight_gpu, CublasUtil.Matrix[] additionParam_gpu) {

    }
}

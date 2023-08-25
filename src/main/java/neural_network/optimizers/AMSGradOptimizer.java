package neural_network.optimizers;

import nnarrays.NNArray;
import utilities.CublasUtil;

public class AMSGradOptimizer extends Optimizer {
    /**
     * AMSGrad
     * m(t) = b1 * m(t-1) + (1 - b1) * dw(t)
     * v(t) = b2 * v(t-1) + (1 - b2) * (dw(t) - m(t)) ^ 2
     * m_(t) = m(t) / (1 - b1^t)
     * v_(t) = max( v_(t-1), v(t) / (1 - b2^t))
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

    public AMSGradOptimizer() {
        this(0.9, 0.999, 0.001);
    }

    public AMSGradOptimizer(double beta1, double beta2, double learningRate) {
        this.beta1 = (float) beta1;
        this.beta2 = (float) beta2;
        this.learningRate = (float) learningRate;
        t = 0;
        countParam = 3;
    }

    public AMSGradOptimizer(double learningRate) {
        this(0.9, 0.999, learningRate);
    }

    @Override
    public void updateWeight(NNArray weight, NNArray deltaWeight, NNArray[] additionParam) {
        if (t == 1) {
            additionParam[2].fill(Float.MIN_VALUE);
        }
        additionParam[0].momentum(deltaWeight, beta1);
        additionParam[1].momentumPow2(deltaWeight, beta2);
        max(additionParam[2], additionParam[1]);

        weight.subDivSqrtNorm(additionParam[0], additionParam[2], learningRate, b1t, b2t);
        deltaWeight.clear();
    }

    @Override
    protected void updateWeight(CublasUtil.Matrix weight_gpu, CublasUtil.Matrix deltaWeight_gpu, CublasUtil.Matrix[] additionParam_gpu) {

    }

    private void max(NNArray v_, NNArray v) {
        for (int i = 0; i < v.size(); i++) {
            v_.getData()[i] = Math.max(v.getData()[i], v_.getData()[i]);
        }
    }
}

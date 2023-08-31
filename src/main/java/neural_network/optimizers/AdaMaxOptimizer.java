package neural_network.optimizers;

import nnarrays.NNArray;

public class AdaMaxOptimizer extends Optimizer {
    /**
     * AdaMax
     * m(t) = b1 * m(t-1) + (1 - b1) * dw(t)
     * v(t) = max( b2 * v(t-1), abs(dw(t)))
     * w(t) = w(t-1) - lr * m(t) / ((1 - b1^t) * v(t))
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

    public AdaMaxOptimizer() {
        this(0.9, 0.999, 0.001);
    }

    public AdaMaxOptimizer(double beta1, double beta2, double learningRate) {
        super();
        this.beta1 = (float) beta1;
        this.beta2 = (float) beta2;
        this.learningRate = (float) learningRate;
        t = 0;
        countParam = 2;
    }

    public AdaMaxOptimizer(double learningRate) {
        this(0.9, 0.999, learningRate);
    }

    @Override
    public void updateWeight(NNArray weight, NNArray deltaWeight, NNArray[] additionParam) {
        if (t == 1) {
            additionParam[1].fill(Float.MIN_VALUE);
        }

        additionParam[0].momentum(deltaWeight, beta1);
        max(additionParam[1], deltaWeight);

        subDivSqrtNorm(weight, additionParam[0], additionParam[1]);
        deltaWeight.clear();
    }

    private void subDivSqrtNorm(NNArray w, NNArray m, NNArray v) {
        float cur_lr = learningRate /  (b1t + 0.0000001f);
        for (int i = 0; i < w.size(); i++) {
            w.getData()[i] -= cur_lr * m.getData()[i] / (v.getData()[i]);
        }
    }

    private void max(NNArray v, NNArray dw) {
        for (int i = 0; i < v.size(); i++) {
            v.getData()[i] = Math.max(beta2 * v.getData()[i], Math.abs(dw.getData()[i]));
        }
    }
}

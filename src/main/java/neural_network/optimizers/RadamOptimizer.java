package neural_network.optimizers;

import nnarrays.NNArray;

public class RadamOptimizer extends Optimizer {
    /**
     * Radam
     * m(t) = b1 * m(t-1) + (1 - b1) * dw(t)
     * v(t) = b2 * v(t-1) + (1 - b2) * dw(t) ^ 2
     * m_(t) = m(t) / (1 - b1^t)
     * p(t) = p(inf) - 2 * t * b2^t / (1 - b2^t)
     * p(inf) = 2 / (1 - b2) - 1
     * if p(t) > 4:
     *      l(t) = sqrt((1 - b2^t) / v(t))
     *      r(t) = sqrt(((p(t) - 4) * (p(t) - 2) * p(inf)) / ((p(inf) - 4) * (p(inf) - 2) * p(t)))
     *      w(t) = w(t-1) - lr * m_(t) * r(t) * l(t)
     * else:
     *      w(t) = w(t-1) - lr * m_(t)
     */
    private final float beta1;
    private final float beta2;
    private final float learningRate;
    private final float pInf;

    private float b1t;
    private float b2t;
    private float pt;
    private float rt;

    @Override
    public void update() {
        t++;
        b1t = (float) (1.0f - Math.pow(beta1, t));
        b2t = (float) (1.0f - Math.pow(beta2, t));
        pt = (float) (pInf - ((2 * t * Math.pow(beta2, t)) / b2t));
        rt = (float) Math.sqrt(((pt - 4) * (pt - 2) * pInf) / ((pInf - 4) * (pInf - 2) * pt));
        super.update();
    }

    public RadamOptimizer() {
        this(0.9, 0.999, 0.001);
    }

    public RadamOptimizer(double beta1, double beta2, double learningRate) {
        this.beta1 = (float) beta1;
        this.beta2 = (float) beta2;
        this.learningRate = (float) learningRate;
        this.pInf = (float) (2.0 / (1.0 - beta2) - 1);
        t = 0;
        countParam = 2;
    }

    public RadamOptimizer(double learningRate) {
        this(0.9, 0.999, learningRate);
    }

    @Override
    public void updateWeight(NNArray weight, NNArray deltaWeight, NNArray[] additionParam) {
        additionParam[0].momentum(deltaWeight, beta1);
        additionParam[1].momentumPow2(deltaWeight, beta2);

        if(pt > 4){
            subDivSqrtNorm(weight, additionParam[0], additionParam[1]);
        } else{
            subDivSqrtNorm(weight, additionParam[0]);
        }
        deltaWeight.clear();
    }

    private void subDivSqrtNorm(NNArray w, NNArray m) {
        float cur_lr = learningRate /  (b1t + 0.0000001f);
        for (int i = 0; i < w.size(); i++) {
            w.getData()[i] -= cur_lr * m.getData()[i];
        }
    }

    private void subDivSqrtNorm(NNArray w, NNArray m, NNArray v) {
        float cur_lr = learningRate /  (b1t + 0.0000001f);
        for (int i = 0; i < w.size(); i++) {
            w.getData()[i] -= cur_lr * m.getData()[i] * rt * Math.sqrt(b2t / (v.getData()[i] + 0.00000001f));
        }
    }
}

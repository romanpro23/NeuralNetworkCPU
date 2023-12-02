package neural_network.optimizers;

import nnarrays.NNArray;

public class AngularGradOptimizer extends Optimizer {
    /**
     * AngularGrad
     * m(t) = b1 * m(t-1) + (1 - b1) * dw(t)
     * v(t) = b2 * v(t-1) + (1 - b2) * dw(t) ^ 2
     * m_(t) = m(t) / (1 - b1^t)
     * v_(t) = v(t) / (1 - b2^t)
     * A(t) = tan^(-1)((dw(t) - dw(t-1)) / (1 + dw(t) * dw(t-1)))
     * A(min) = min(A(t), A(t-1))
     * phi = tanh(cos(A(min))) * lambda1 + lambda2
     * w(t) = w(t-1) - lr * phi * m_(t) / sqrt(v_(t))
     */
    private final float beta1;
    private final float beta2;
    private final float learningRate;
    private float lambda1;
    private float lambda2;
    private boolean cos;

    private float b1t;
    private float b2t;

    @Override
    public void update() {
        t++;
        b1t = (float) (1.0f - Math.pow(beta1, t));
        b2t = (float) (1.0f - Math.pow(beta2, t));
        super.update();
    }

    public AngularGradOptimizer() {
        this(0.9, 0.999, 0.001);
    }

    public AngularGradOptimizer(double beta1, double beta2, double learningRate) {
        super();
        this.beta1 = (float) beta1;
        this.beta2 = (float) beta2;
        this.learningRate = (float) learningRate;
        this.lambda1 = 0.5f;
        this.lambda2 = 0.5f;
        this.cos = true;
        t = 0;
        countParam = 4;
    }

    public AngularGradOptimizer useTan() {
        this.cos = false;
        return this;
    }

    public AngularGradOptimizer setLambda(float lambda) {
        return setLambda(lambda, lambda);
    }

    public AngularGradOptimizer setLambda(float lambda1, float lambda2) {
        this.lambda1 = lambda1;
        this.lambda2 = lambda2;
        return this;
    }

    public AngularGradOptimizer(double learningRate) {
        this(0.9, 0.999, learningRate);
    }

    @Override
    public void updateWeight(NNArray weight, NNArray deltaWeight, NNArray[] additionParam) {
        if (t == 1) {
            additionParam[3].fill(Float.MAX_VALUE);
        }
        additionParam[0].momentum(deltaWeight, beta1);
        additionParam[1].momentumPow2(deltaWeight, beta2);
        NNArray a = deltaWeight.angularGrad(additionParam[2]);
        NNArray phi;
        if (cos) {
            phi = deltaWeight.angularCos(additionParam[3], lambda1, lambda2);
        } else {
            phi = deltaWeight.angularTan(additionParam[3], lambda1, lambda2);
        }
        weight.subDivSqrtNorm(additionParam[0], additionParam[1], phi, learningRate, b1t, b2t);
        additionParam[2].copy(deltaWeight);
        additionParam[3].copy(a);
        deltaWeight.clear();
    }
}

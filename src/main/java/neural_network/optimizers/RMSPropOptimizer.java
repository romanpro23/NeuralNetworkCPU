package neural_network.optimizers;

import nnarrays.NNArray;
import utilities.CublasUtil;

public class RMSPropOptimizer extends Optimizer {
    /**
     * dr - decay rate
     * RMSProp
     * θ(t) = dr * θ(t-1) + (1 - dr) * dw(t) * dw(t)
     * w(t) = w(t-1) - m(t) / sqrt(θ(t))
     * m(t) = dr * m(t-1) + m(t) / sqrt(θ(t)
     */
    private final float decayRate;

    public RMSPropOptimizer() {
        this(0.95);
    }

    public RMSPropOptimizer(double decayRate) {
        this.decayRate = (float) decayRate;
        this.countParam = 2;
    }

    @Override
    public void updateWeight(NNArray weight, NNArray deltaWeight, NNArray[] additionParam) {
        additionParam[1].momentumPow2(deltaWeight, decayRate);
        NNArray deltaW = deltaWeight.divSqrt(additionParam[0], additionParam[1]);
        weight.sub(deltaW);
        additionParam[0].momentumPow2(deltaW, decayRate);
        deltaWeight.clear();
    }

    @Override
    protected void updateWeight(CublasUtil.Matrix weight_gpu, CublasUtil.Matrix deltaWeight_gpu, CublasUtil.Matrix[] additionParam_gpu) {

    }
}

package neural_network.optimizers;

import lombok.Data;
import lombok.Getter;
import nnarrays.NNArray;

import java.util.ArrayList;

@Data
public abstract class Optimizer {
    protected ArrayList<DataOptimize> dataOptimize;
    protected float clipValue = 0;
    @Getter
    protected int countParam = 0;

    protected int t = 0;

    public Optimizer() {
        dataOptimize = new ArrayList<>();
    }

    public void update() {
        t++;
    }

    public abstract void updateWeight(NNArray weight, NNArray deltaWeight, NNArray[] additionParam);

    public Optimizer setClipValue(double clipValue) {
        this.clipValue = (float) clipValue;

        return this;
    }

    public float getClipValue() {
        return clipValue;
    }
}

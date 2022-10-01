package neural_network.optimizers;

import lombok.Data;
import lombok.Getter;
import nnarrays.NNArray;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

@Data
public abstract class Optimizer {
    protected ArrayList<DataOptimize> optimizeData;
    protected float clipValue = 0;
    @Getter
    protected int countParam = 0;

    protected int t = 0;

    public Optimizer() {
        optimizeData = new ArrayList<>();
    }

    public void update() {
        ExecutorService executor = Executors.newFixedThreadPool(optimizeData.size());
        for (int t = 0; t < optimizeData.size(); t++) {
            final int finalT = t;
            executor.execute(() -> {
                DataOptimize data = optimizeData.get(finalT);
                if (clipValue != 0) {
                    data.getDerWeight().clip(clipValue);
                }
                updateWeight(data.getWeight(), data.getDerWeight(), data.getAdditionParam());
            });
        }
        executor.shutdown();
        while (!executor.isTerminated()) {
        }
    }

    public void addDataOptimize(NNArray weight, NNArray derWeight) {
        NNArray[] additionParam = new NNArray[countParam];
        for (int i = 0; i < countParam; i++) {
            additionParam[i] = new NNArray(weight.size());
        }
        optimizeData.add(new DataOptimize(weight, derWeight, additionParam));
    }

    protected abstract void updateWeight(NNArray weight, NNArray deltaWeight, NNArray[] additionParam);

    public Optimizer setClipValue(double clipValue) {
        this.clipValue = (float) clipValue;

        return this;
    }

    public float getClipValue() {
        return clipValue;
    }
}

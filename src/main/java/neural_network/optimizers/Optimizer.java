package neural_network.optimizers;

import lombok.Data;
import lombok.Getter;
import lombok.SneakyThrows;
import nnarrays.NNArray;

import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;
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
    
    @SneakyThrows
    public void save(FileWriter writer){
        writer.write(t + "\n");
        writer.write(clipValue + "\n");
        writer.write(countParam + "\n");
        writer.flush();

        for (DataOptimize data : optimizeData) {
            for (int j = 0; j < countParam; j++) {
                data.getAdditionParam()[j].save(writer);
            }
        }
    }

    @SneakyThrows
    public Optimizer read(Scanner scanner){
        this.t = Integer.parseInt(scanner.nextLine());
        this.clipValue = Float.parseFloat(scanner.nextLine());
        this.countParam = Integer.parseInt(scanner.nextLine());

        for (DataOptimize data : optimizeData) {
            for (int j = 0; j < countParam; j++) {
                data.getAdditionParam()[j] = NNArray.read(scanner);
            }
        }

        return this;
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
}

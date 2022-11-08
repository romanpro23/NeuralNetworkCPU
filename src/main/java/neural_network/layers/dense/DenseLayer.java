package neural_network.layers.dense;

import lombok.Getter;
import lombok.Setter;
import lombok.SneakyThrows;
import neural_network.initialization.Initializer;
import neural_network.optimizers.Optimizer;
import neural_network.regularization.Regularization;
import nnarrays.NNArray;
import nnarrays.NNArrays;
import nnarrays.NNMatrix;
import nnarrays.NNVector;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class DenseLayer extends DenseNeuralLayer {
    //trainable parts
    private Regularization regularization;
    private Initializer initializer;
    private boolean loadWeight;

    //weight and threshold
    @Getter
    private NNMatrix weight;
    private NNMatrix derWeight;

    private NNVector threshold;
    private NNVector derThreshold;

    public DenseLayer(int countNeuron) {
        super();
        this.countNeuron = countNeuron;
        this.trainable = true;
        initializer = new Initializer.HeNormal();
    }

    @Override
    public void initialize(Optimizer optimizer) {
        optimizer.addDataOptimize(weight, derWeight);
        optimizer.addDataOptimize(threshold, derThreshold);
    }

    public DenseLayer setTrainable(boolean trainable) {
        this.trainable = trainable;

        return this;
    }

    public DenseLayer setInitializer(Initializer initializer) {
        this.initializer = initializer;

        return this;
    }

    public DenseLayer setRegularization(Regularization regularization) {
        this.regularization = regularization;

        return this;
    }

    @Override
    public int info() {
        int countParam = weight.size() + threshold.size();
        System.out.println("Dense \t\t|  " + weight.getColumn() + "\t\t\t|  " + countNeuron + "\t\t\t|\t" + countParam);
        return countParam;
    }

    @Override
    public void write(FileWriter writer) throws IOException {
        writer.write("Dense layer\n");
        writer.write(countNeuron + "\n");
        threshold.save(writer);
        weight.save(writer);
        if (regularization != null) {
            regularization.write(writer);
        } else {
            writer.write("null\n");
        }
        writer.write(trainable + "\n");
        writer.flush();
    }

    @Override
    public void initialize(int[] size) {
        if (size.length != 1) {
            throw new ExceptionInInitializerError("Error size pre layer!");
        }
        derThreshold = new NNVector(countNeuron);
        derWeight = new NNMatrix(countNeuron, size[0]);

        if (!loadWeight) {
            threshold = new NNVector(countNeuron);
            weight = new NNMatrix(countNeuron, size[0]);
            initializer.initialize(weight);
        }
    }

    @SneakyThrows
    @Override
    public void generateOutput(NNArray[] inputs) {
        this.input = NNArrays.isVector(inputs);
        this.output = new NNVector[input.length];

        int countC = getCountCores();
        ExecutorService executor = Executors.newFixedThreadPool(countC);
        for (int t = 0; t < countC; t++) {
            final int firstIndex = t * input.length / countC;
            final int lastIndex = Math.min(input.length, (t + 1) * input.length / countC);
            executor.execute(() -> {
                for (int i = firstIndex; i < lastIndex; i++) {
                    output[i] = input[i].mul(weight);
                    output[i].add(threshold);
                }
            });
        }
        executor.shutdown();
        while (!executor.isTerminated()) {
        }
    }

    @SneakyThrows
    @Override
    public void generateError(NNArray[] errors) {
        errorNL = getErrorNextLayer(errors);
        this.error = new NNVector[errors.length];

        int countC = getCountCores();
        ExecutorService executor = Executors.newFixedThreadPool(countC);
        for (int t = 0; t < countC; t++) {
            final int firstIndex = t * input.length / countC;
            final int lastIndex = Math.min(input.length, (t + 1) * input.length / countC);
            executor.execute(() -> {
                for (int i = firstIndex; i < lastIndex; i++) {
                    error[i] = errorNL[i].mulT(weight);
                    if(trainable) {
                        derivativeWeight(input[i], errorNL[i]);
                    }
                }
            });
        }
        executor.shutdown();
        while (!executor.isTerminated()) {
        }

        if (trainable && regularization != null) {
                regularization.regularization(weight);
                regularization.regularization(threshold);
        }
    }

    private void derivativeWeight(NNVector input, NNVector error){
        for (int j = 0, index = 0; j < error.size(); j++) {
            for (int k = 0; k < input.size(); k++, index++) {
                derWeight.getData()[index] += error.getData()[j] * input.getData()[k];
            }
        }
        derThreshold.add(error);
    }

    public static DenseLayer read(Scanner scanner) {
        DenseLayer denseLayer = new DenseLayer(Integer.parseInt(scanner.nextLine()));
        denseLayer.threshold = NNVector.read(scanner);
        denseLayer.weight = NNMatrix.read(scanner);
        denseLayer.setRegularization(Regularization.read(scanner));
        denseLayer.setTrainable(Boolean.parseBoolean(scanner.nextLine()));
        denseLayer.loadWeight = true;
        return denseLayer;
    }
}

package neural_network.layers.dense;

import lombok.Getter;
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
import java.util.Scanner;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class SNDenseLayer extends DenseNeuralLayer {
    //trainable parts
    private Regularization regularization;
    private Initializer initializer;
    private boolean loadWeight;
    private boolean useBias;

    //weightAttention and threshold
    @Getter
    private NNMatrix weight;
    private NNMatrix derWeight;

    private NNVector threshold;
    private NNVector derThreshold;

    private NNVector u;
    private NNVector v;
    private float sigma;

    public SNDenseLayer(int countNeuron) {
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

    public SNDenseLayer setTrainable(boolean trainable) {
        this.trainable = trainable;

        return this;
    }

    public SNDenseLayer setUseBias(boolean useBias) {
        this.useBias = useBias;

        return this;
    }

    public SNDenseLayer setInitializer(Initializer initializer) {
        this.initializer = initializer;

        return this;
    }

    public SNDenseLayer setRegularization(Regularization regularization) {
        this.regularization = regularization;

        return this;
    }

    @Override
    public int info() {
        int countParam = weight.size() + threshold.size();
        System.out.println("SN dense \t|  " + weight.getColumn() + "\t\t\t|  " + countNeuron + "\t\t\t|\t" + countParam);
        return countParam;
    }

    @Override
    public void save(FileWriter writer) throws IOException {
        writer.write("Spectral normalization dense layer\n");
        writer.write(countNeuron + "\n");
        threshold.save(writer);
        weight.save(writer);

        u.save(writer);
        v.save(writer);
        if (regularization != null) {
            regularization.write(writer);
        } else {
            writer.write("null\n");
        }
        writer.write(trainable + "\n");
        writer.write(useBias + "\n");
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

            u = new NNVector(countNeuron);
            v = new NNVector(size[0]);

            Initializer initializerSN = new Initializer.RandomNormal();
            initializerSN.initialize(u);
            u.l2norm();
            initializerSN.initialize(v);
            v.l2norm();
        }
    }

    private float spectralNorm() {
        v.clear();
        v.addMulT(u, weight);
        v.l2norm();

        u.clear();
        u.addMul(v, weight);
        u.l2norm();

        return 0.0000001f + NNArrays.sum(v.mul(u.dotT(weight)));
    }

    @SneakyThrows
    @Override
    public void generateOutput(NNArray[] inputs) {
        this.input = NNArrays.isVector(inputs);
        this.output = new NNVector[input.length];

        sigma = spectralNorm();
        weight.div(sigma);

        int countC = getCountCores();
        ExecutorService executor = Executors.newFixedThreadPool(countC);
        for (int t = 0; t < countC; t++) {
            final int firstIndex = t * input.length / countC;
            final int lastIndex = Math.min(input.length, (t + 1) * input.length / countC);
            executor.execute(() -> {
                for (int i = firstIndex; i < lastIndex; i++) {
                    output[i] = input[i].dot(weight);
                    if(useBias) {
                        output[i].add(threshold);
                    }
                }
            });
        }
        executor.shutdown();
        while (!executor.isTerminated()) {
        }
    }

    private void backSpectralNorm() {
        NNMatrix dW = u.dot(v);
        dW.oneSub();
        dW.dotT(weight);
        dW.div(sigma);

        derWeight.mul(dW);
        weight.mul(sigma);
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
                    error[i] = errorNL[i].dotT(weight);
                    if (trainable) {
                        derivativeWeight(input[i], errorNL[i]);
                        if (useBias) {
                            derThreshold.add(errorNL[i]);
                        }
                    }
                }
            });
        }
        executor.shutdown();
        while (!executor.isTerminated()) {
        }

        if (trainable) {
            backSpectralNorm();
            if (regularization != null) {
                regularization.regularization(weight);
                regularization.regularization(threshold);
            }
        }
    }

    private void derivativeWeight(NNVector input, NNVector error) {
        for (int j = 0, index = 0; j < error.size(); j++) {
            for (int k = 0; k < input.size(); k++, index++) {
                derWeight.getData()[index] += error.getData()[j] * input.getData()[k];
            }
        }
    }

    public static SNDenseLayer read(Scanner scanner) {
        SNDenseLayer layer = new SNDenseLayer(Integer.parseInt(scanner.nextLine()));
        layer.threshold = NNVector.read(scanner);
        layer.weight = NNMatrix.read(scanner);
        layer.u = NNVector.read(scanner);
        layer.v = NNVector.read(scanner);

        layer.setRegularization(Regularization.read(scanner));
        layer.setTrainable(Boolean.parseBoolean(scanner.nextLine()));
        layer.setUseBias(Boolean.parseBoolean(scanner.nextLine()));
        layer.loadWeight = true;
        return layer;
    }
}

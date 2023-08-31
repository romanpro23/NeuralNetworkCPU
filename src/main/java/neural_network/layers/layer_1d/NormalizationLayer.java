package neural_network.layers.layer_1d;

import lombok.Setter;
import neural_network.optimizers.Optimizer;
import neural_network.regularization.Regularization;
import nnarrays.NNArray;
import nnarrays.NNArrays;
import nnarrays.NNVector;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class NormalizationLayer extends DenseNeuralLayer {
    //trainable parts
    private Regularization regularization;
    @Setter
    private boolean loadWeight;

    private final float epsilon;

    //betta
    private NNVector betta;
    private NNVector derBetta;
    //gamma
    private NNVector gamma;
    private NNVector derGamma;

    private NNVector[] normOutput;

    private NNVector[] mean, var;

    public NormalizationLayer() {
        this.trainable = true;
        this.epsilon = 0.001f;
    }

    @Override
    public void initialize(int[] size) {
        if (size.length != 1) {
            throw new ExceptionInInitializerError("Error size pre layer!");
        }
        countNeuron = size[0];
        derBetta = new NNVector(countNeuron);
        derGamma = new NNVector(countNeuron);

        if (!loadWeight) {
            betta = new NNVector(countNeuron);
            gamma = new NNVector(countNeuron);

            gamma.fill(1);
        }
    }

    @Override
    public void initialize(Optimizer optimizer) {
        optimizer.addDataOptimize(betta, derBetta);
        optimizer.addDataOptimize(gamma, derGamma);
    }

    @Override
    public void generateOutput(NNArray[] input) {
        this.input = NNArrays.isVector(input);
        this.output = new NNVector[input.length];
        this.normOutput = new NNVector[input.length];

        this.mean = new NNVector[input.length];
        this.var = new NNVector[input.length];

        ExecutorService executor = Executors.newFixedThreadPool(input.length);
        for (int t = 0; t < input.length; t++) {
            final int i = t;
            executor.execute(() -> {
                mean[i] = findMean(this.input[i]);
                var[i] = findVariance(this.input[i], mean[i]);

                normOutput[i] = new NNVector(countNeuron);
                output[i] = normalization(normOutput[i], this.input[i], mean[i], var[i]);
            });
        }
        executor.shutdown();
        while (!executor.isTerminated()) {
        }
    }

    private NNVector normalization(NNVector outputNorm, NNVector input, NNVector mean, NNVector var) {
        NNVector output = new NNVector(outputNorm);
        float varSqrt = (float) (Math.sqrt(var.get(0) + epsilon));

        for (int j = 0; j < outputNorm.size(); j++) {
            outputNorm.getData()[j] = (input.get(j) - mean.get(0)) / varSqrt;
            output.getData()[j] = outputNorm.get(j) * gamma.get(j) + betta.get(j);
        }
        return output;
    }

    private NNVector findMean(NNVector input) {
        NNVector mean = new NNVector(1);
        for (int j = 0; j < input.size(); j++) {
            mean.getData()[0] += input.get(j);
        }

        mean.div(input.size());
        return mean;
    }

    private NNVector findVariance(NNVector input, NNVector mean) {
        NNVector var = new NNVector(1);
        float sub;
        for (int j = 0; j < input.size(); j++) {
            sub = input.get(j) - mean.get(0);
            var.getData()[0] += sub * sub;
        }
        var.div(input.size());
        return var;
    }

    @Override
    public void generateError(NNArray[] errors) {
        errorNL = getErrorNextLayer(errors);
        this.error = new NNVector[errors.length];

        ExecutorService executor = Executors.newFixedThreadPool(input.length);
        for (int t = 0; t < input.length; t++) {
            final int i = t;
            executor.execute(() -> {
                NNVector errorNorm = new NNVector(countNeuron);
                errorNorm.addProduct(errorNL[i], gamma);

                NNVector errorVariance = derVar(errorNorm, input[i], mean[i], var[i]);
                NNVector errorMean = derMean(errorNorm, input[i], errorVariance, mean[i], var[i]);

                error[i] = derNorm(errorNorm, input[i], errorMean, errorVariance, mean[i], var[i]);

                if (trainable) {
                    derivativeWeight(errorNL[i], normOutput[i]);
                }
            });
        }
        executor.shutdown();
        while (!executor.isTerminated()) {
        }

        if (trainable && regularization != null) {
            regularization.regularization(betta);
            regularization.regularization(gamma);
        }
    }

    private NNVector derVar(NNVector error, NNVector input, NNVector mean, NNVector var) {
        NNVector derVariance = new NNVector(var);
        float[] dVar = new float[var.size()];
        for (int i = 0; i < var.size(); i++) {
            dVar[i] = (float) (-0.5 * Math.pow(var.get(i) + epsilon, -1.5));
        }

        for (int j = 0; j < error.size(); j++) {
            derVariance.getData()[0] += error.get(j) * (input.get(j) - mean.get(0));
        }
        for (int i = 0; i < derVariance.size(); i++) {
            derVariance.getData()[i] *= dVar[i];
        }
        return derVariance;
    }

    private NNVector derMean(NNVector error, NNVector input, NNVector derVar, NNVector mean, NNVector var) {
        NNVector derMean = new NNVector(mean.size());
        float[] dMean = new float[mean.size()];
        float[] dVar = new float[var.size()];
        for (int i = 0; i < var.size(); i++) {
            dMean[i] = (float) (-1.0f / Math.sqrt(var.get(i) + epsilon));
        }

        for (int j = 0; j < error.size(); j++) {
            derMean.getData()[0] += error.get(j);
            dVar[0] += input.get(j) - mean.get(0);
        }

        for (int i = 0; i < derMean.size(); i++) {
            derMean.getData()[i] *= dMean[i];
            derMean.getData()[i] += (-2.0f * derVar.get(i) * dVar[i]) / error.size();
        }
        return derMean;
    }

    private NNVector derNorm(NNVector errors, NNVector input, NNVector errorMean, NNVector errorVar, NNVector mean, NNVector var) {
        NNVector error = new NNVector(input);
        errorMean.div(errors.size());
        errorVar.mul(2.0f / errors.size());

        float[] dVar = new float[var.size()];
        for (int i = 0; i < var.size(); i++) {
            dVar[i] = (float) (1.0 / Math.sqrt(var.getData()[i] + epsilon));
        }

        for (int i = 0; i < errors.size(); i++) {
            error.getData()[i] = errors.get(i) * dVar[0] + errorVar.get(0) * (input.get(i) - mean.get(0)) + errorMean.get(0);
        }
        return error;
    }

    private void derivativeWeight(NNVector error, NNVector normOutput) {
        for (int j = 0; j < error.size(); j++) {
            derBetta.getData()[j] += error.get(j);
            derGamma.getData()[j] += error.get(j) * normOutput.get(j);
        }
    }

    public NormalizationLayer setTrainable(boolean trainable) {
        this.trainable = trainable;

        return this;
    }

    public NormalizationLayer setRegularization(Regularization regularization) {
        this.regularization = regularization;

        return this;
    }

    @Override
    public int info() {
        int countParam = betta.size() * 2;
        System.out.println("Layer norm\t|  " + countNeuron + "\t\t\t|  " + countNeuron + "\t\t\t|\t" + countParam);

        return countParam;
    }

    @Override
    public void save(FileWriter writer) throws IOException {
        writer.write("Normalization layer\n");
        gamma.save(writer);
        betta.save(writer);

        if (regularization != null) {
            regularization.write(writer);
        } else {
            writer.write("null\n");
        }
        writer.write(trainable + "\n");
        writer.flush();
    }

    public static NormalizationLayer read(Scanner scanner) {
        NormalizationLayer layer = new NormalizationLayer();
        layer.loadWeight = false;
        layer.gamma = NNVector.read(scanner);
        layer.betta = NNVector.read(scanner);
        layer.setRegularization(Regularization.read(scanner));
        layer.setTrainable(Boolean.parseBoolean(scanner.nextLine()));
        layer.loadWeight = true;
        return layer;
    }
}

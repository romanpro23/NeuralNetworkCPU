package neural_network.layers.layer_2d;

import lombok.Setter;
import neural_network.optimizers.Optimizer;
import neural_network.regularization.Regularization;
import nnarrays.*;
import utilities.CublasUtil;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class NormalizationLayer2D extends NeuralLayer2D {
    //trainable parts
    private Regularization regularization;
    @Setter
    private boolean loadWeight;

    private final float epsilon;

    //betta
    @Setter
    private NNVector betta;
    private NNVector derBetta;
    //gamma
    @Setter
    private NNVector gamma;
    private NNVector derGamma;

    private NNMatrix[] normOutput;
    private NNVector[] mean, var;

    public NormalizationLayer2D() {
        this.trainable = true;
        this.epsilon = 0.001f;
    }

    @Override
    public void initialize(int[] size) {
        if (size.length != 2) {
            throw new ExceptionInInitializerError("Error depth pre layer!");
        }
        this.depth = size[1];
        this.width = size[0];

        outDepth = depth;
        outWidth = width;

        derBetta = new NNVector(depth);
        derGamma = new NNVector(depth);

        if (!loadWeight) {
            betta = new NNVector(depth);
            gamma = new NNVector(depth);

            gamma.fill(1);
        }
    }

    @Override
    public void initialize(Optimizer optimizer) {
        optimizer.addDataOptimize(betta, derBetta);
        optimizer.addDataOptimize(gamma, derGamma);
    }

    @Override
    public void generateError(CublasUtil.Matrix[] errors) {

    }

    @Override
    public void generateOutput(NNArray[] input) {
        this.input = NNArrays.isMatrix(input);
        this.output = new NNMatrix[input.length];
        this.normOutput = new NNMatrix[input.length];
        this.mean = new NNVector[input.length];
        this.var = new NNVector[input.length];

        ExecutorService executor = Executors.newFixedThreadPool(input.length);
        for (int t = 0; t < input.length; t++) {
            final int i = t;
            executor.execute(() -> {
                normOutput[i] = new NNMatrix(outWidth, outDepth);
                output[i] = new NNMatrix(outWidth, outDepth);
                findMean(i);
                findVariance(i);
                normalization(i);
            });
        }
        executor.shutdown();
        while (!executor.isTerminated()) {
        }
    }

    @Override
    public void generateOutput(CublasUtil.Matrix[] input_gpu) {

    }

    private void normalization(int n) {
        float[] varSqrt = new float[var[n].size()];
        for (int i = 0; i < var[n].size(); i++) {
            varSqrt[i] = (float) (Math.sqrt(var[n].getData()[i] + epsilon));
        }
        int index = 0;
        for (int j = 0; j < width; j++) {
            for (int k = 0; k < depth; k++, index++) {
                normOutput[n].getData()[index] = (input[n].getData()[index] - mean[n].get(j)) / varSqrt[j];
                output[n].getData()[index] = normOutput[n].getData()[index] * gamma.get(k) + betta.get(k);
            }
        }
    }

    private void findMean(int n) {
        mean[n] = new NNVector(width);

        int index = 0;
        for (int j = 0; j < width; j++) {
            for (int k = 0; k < depth; k++, index++) {
                mean[n].getData()[j] += input[n].getData()[index];
            }
        }
        mean[n].div(depth);
    }

    private void findVariance(int n) {
        var[n] = new NNVector(width);
        float sub;
        int index = 0;
        for (int j = 0; j < width; j++) {
            for (int k = 0; k < depth; k++, index++) {
                sub = input[n].getData()[index] - mean[n].getData()[j];
                var[n].getData()[j] += sub * sub;
            }
        }
        var[n].div(depth);
    }

    @Override
    public void generateError(NNArray[] errors) {
        errorNL = getErrorNextLayer(errors);
        this.error = new NNMatrix[errors.length];

        ExecutorService executor = Executors.newFixedThreadPool(input.length);
        for (int t = 0; t < input.length; t++) {
            final int i = t;
            executor.execute(() -> {
                error[i] = new NNMatrix(outWidth, outDepth);
                NNMatrix errorNorm = generateErrorNorm(i);
                NNVector errorVariance = derVar(errorNorm, i);
                NNVector errorMean = derMean(errorNorm, errorVariance, i);

                derNorm(errorNorm, errorMean, errorVariance, i);

                if (trainable) {
                    derivativeWeight(errorNL[i], i);
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

    private NNMatrix generateErrorNorm(int n) {
        NNMatrix errorNorm = new NNMatrix(outWidth, outDepth);
        int index = 0;
        for (int j = 0; j < width; j++) {
            for (int k = 0; k < depth; k++, index++) {
                errorNorm.getData()[index] = errorNL[n].getData()[index] * gamma.get(k);
            }
        }

        return errorNorm;
    }

    private NNVector derVar(NNMatrix error, int n) {
        NNVector derVariance = new NNVector(var[n].size());
        float[] dVar = new float[var[n].size()];
        for (int i = 0; i < var[n].size(); i++) {
            dVar[i] = (float) (-0.5 * Math.pow(var[n].get(i) + epsilon, -1.5));
        }

        int index = 0;
        for (int j = 0; j < width; j++) {
            for (int k = 0; k < depth; k++, index++) {
                derVariance.getData()[j] += error.get(index) * (input[n].get(index) - mean[n].get(j));
            }
        }

        for (int i = 0; i < derVariance.size(); i++) {
            derVariance.getData()[i] *= dVar[i];
        }
        return derVariance;
    }

    private NNVector derMean(NNMatrix error, NNVector derVar, int n) {
        NNVector derMean = new NNVector(mean[n].size());
        float[] dMean = new float[mean[n].size()];
        float[] dVar = new float[var[n].size()];
        for (int i = 0; i < var[n].size(); i++) {
            dMean[i] = (float) (-1.0f / Math.sqrt(var[n].get(i) + epsilon));
        }

        int index = 0;
        for (int j = 0; j < width; j++) {
            for (int k = 0; k < depth; k++, index++) {
                derMean.getData()[j] += error.get(index);
                dVar[j] += input[n].get(index) - mean[n].get(j);
            }
        }

        for (int i = 0; i < derMean.size(); i++) {
            derMean.getData()[i] *= dMean[i];
            derMean.getData()[i] += (-2.0f * derVar.get(i) * dVar[i]) / (depth);
        }
        return derMean;
    }

    private void derNorm(NNMatrix errors, NNVector errorMean, NNVector errorVar, int n) {
        errorMean.div(depth);
        errorVar.mul(2.0f / (depth));

        float[] dVar = new float[var[n].size()];
        for (int i = 0; i < var[n].size(); i++) {
            dVar[i] = (float) (1.0 / Math.sqrt(var[n].getData()[i] + epsilon));
        }

        int index = 0;
        for (int j = 0; j < width; j++) {
            for (int k = 0; k < depth; k++, index++) {
                error[n].getData()[index] = errors.getData()[index] * dVar[j] + errorVar.get(j) *
                        (input[n].get(index) - mean[n].get(j)) + errorMean.get(j);
            }
        }
    }

    private void derivativeWeight(NNMatrix error, int n) {
        int index = 0;
        for (int j = 0; j < width; j++) {
            for (int k = 0; k < depth; k++, index++) {
                derBetta.getData()[k] += error.getData()[index];
                derGamma.getData()[k] += error.getData()[index] * ((output[n].getData()[index] - betta.get(k)) / gamma.get(k));
            }
        }
    }

    public NormalizationLayer2D setTrainable(boolean trainable) {
        this.trainable = trainable;

        return this;
    }

    public NormalizationLayer2D setRegularization(Regularization regularization) {
        this.regularization = regularization;

        return this;
    }

    @Override
    public int info() {
        int countParam = betta.size() * 2;
        System.out.println("Layer norm\t| " + width + ",\t" + depth + "\t\t| " + outWidth + ",\t" + outDepth + "\t\t|\t" + countParam);

        return countParam;
    }

    @Override
    public void save(FileWriter writer) throws IOException {
        writer.write("Normalization layer 2D\n");
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

    public static NormalizationLayer2D read(Scanner scanner) {
        NormalizationLayer2D layer = new NormalizationLayer2D();
        layer.loadWeight = false;
        layer.gamma = NNVector.read(scanner);
        layer.betta = NNVector.read(scanner);
        layer.setRegularization(Regularization.read(scanner));
        layer.setTrainable(Boolean.parseBoolean(scanner.nextLine()));
        layer.loadWeight = true;
        return layer;
    }
}

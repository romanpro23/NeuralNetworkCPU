package neural_network.layers.convolution_3d;

import lombok.Setter;
import neural_network.optimizers.Optimizer;
import neural_network.regularization.Regularization;
import nnarrays.NNArray;
import nnarrays.NNArrays;
import nnarrays.NNTensor;
import nnarrays.NNVector;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class InstanceNormalizationLayer3D extends ConvolutionNeuralLayer {
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

    private NNTensor[] normOutput;

    private NNVector[] mean, var;

    private int size;

    public InstanceNormalizationLayer3D() {
        this.trainable = true;
        this.epsilon = 0.001f;
    }

    @Override
    public void initialize(int[] size) {
        if (size.length != 3) {
            throw new ExceptionInInitializerError("Error size pre layer!");
        }
        this.depth = size[2];
        this.height = size[0];
        this.width = size[1];
        this.size = height * width;

        outDepth = depth;
        outWidth = width;
        outHeight = height;

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
    public void generateOutput(NNArray[] input) {
        this.input = NNArrays.isTensor(input);
        this.output = new NNTensor[input.length];
        //this.normOutput = new NNTensor[input.length];
        this.mean = new NNVector[input.length];
        this.var = new NNVector[input.length];

        int countC = getCountCores();
        ExecutorService executor = Executors.newFixedThreadPool(countC);
        for (int t = 0; t < countC; t++) {
            final int firstIndex = t * input.length / countC;
            final int lastIndex = Math.min(input.length, (t + 1) * input.length / countC);
            executor.execute(() -> {
                for (int i = firstIndex; i < lastIndex; i++) {
                    output[i] = new NNTensor(outHeight, outWidth, outDepth);
                    findMean(i);
                    findVariance(i);
                    normalization(i);
                }
            });
        }
        executor.shutdown();
        while (!executor.isTerminated()) {
        }
    }

    private void normalization(int n) {
        float[] varSqrt = new float[var[n].size()];
        for (int i = 0; i < var[n].size(); i++) {
            varSqrt[i] = (float) (Math.sqrt(var[n].getData()[i] + epsilon));
        }
        int index = 0;
        for (int j = 0; j < size; j++) {
            for (int k = 0; k < depth; k++, index++) {
                output[n].getData()[index] = (input[n].getData()[index] - mean[n].get(k)) / varSqrt[k];
                output[n].getData()[index] = output[n].getData()[index] * gamma.get(k) + betta.get(k);
            }
        }
    }

    private void findMean(int n) {
        mean[n] = new NNVector(depth);

        int index = 0;
        for (int j = 0; j < size; j++) {
            for (int k = 0; k < depth; k++, index++) {
                mean[n].getData()[k] += input[n].getData()[index];
            }
        }
        mean[n].div(size);
    }

    private void findVariance(int n) {
        var[n] = new NNVector(depth);
        float sub;
        int index = 0;
        for (int j = 0; j < size; j++) {
            for (int k = 0; k < depth; k++, index++) {
                sub = input[n].getData()[index] - mean[n].getData()[k];
                var[n].getData()[k] += sub * sub;
            }
        }
        var[n].div(size);
    }

    @Override
    public void generateError(NNArray[] errors) {
        errorNL = getErrorNextLayer(errors);
        this.error = new NNTensor[errors.length];

        int countC = getCountCores();
        ExecutorService executor = Executors.newFixedThreadPool(countC);
        for (int t = 0; t < countC; t++) {
            final int firstIndex = t * input.length / countC;
            final int lastIndex = Math.min(input.length, (t + 1) * input.length / countC);
            executor.execute(() -> {
                for (int i = firstIndex; i < lastIndex; i++) {
                    error[i] = new NNTensor(outHeight, outWidth, outDepth);
                    NNTensor errorNorm = generateErrorNorm(i);
                    NNVector errorVariance = derVar(errorNorm, i);
                    NNVector errorMean = derMean(errorNorm, errorVariance, i);

                    derNorm(errorNorm, errorMean, errorVariance, i);

                    if (trainable) {
                        derivativeWeight(errorNL[i], i);
                    }
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

    private NNTensor generateErrorNorm(int n) {
        NNTensor errorNorm = new NNTensor(outHeight, outWidth, outDepth);
        int index = 0;
        for (int j = 0; j < size; j++) {
            for (int k = 0; k < depth; k++, index++) {
                errorNorm.getData()[index] = errorNL[n].getData()[index] * gamma.get(k);
            }
        }

        return errorNorm;
    }

    private NNVector derVar(NNTensor error, int n) {
        NNVector derVariance = new NNVector(var[n].size());
        float[] dVar = new float[var[n].size()];
        for (int i = 0; i < var[n].size(); i++) {
            dVar[i] = (float) (-0.5 * Math.pow(var[n].get(i) + epsilon, -1.5));
        }

        int index = 0;
        for (int j = 0; j < size; j++) {
            for (int k = 0; k < depth; k++, index++) {
                derVariance.getData()[k] += error.get(index) * (input[n].get(index) - mean[n].get(k));
            }
        }

        for (int i = 0; i < derVariance.size(); i++) {
            derVariance.getData()[i] *= dVar[i];
        }
        return derVariance;
    }

    private NNVector derMean(NNTensor error, NNVector derVar, int n) {
        NNVector derMean = new NNVector(mean[n].size());
        float[] dMean = new float[mean[n].size()];
        float[] dVar = new float[var[n].size()];
        for (int i = 0; i < var[n].size(); i++) {
            dMean[i] = (float) (-1.0f / Math.sqrt(var[n].get(i) + epsilon));
        }

        int index = 0;
        for (int j = 0; j < size; j++) {
            for (int k = 0; k < depth; k++, index++) {
                derMean.getData()[k] += error.get(index);
                dVar[k] += input[n].get(index) - mean[n].get(k);
            }
        }

        for (int i = 0; i < derMean.size(); i++) {
            derMean.getData()[i] *= dMean[i];
            derMean.getData()[i] += (-2.0f * derVar.get(i) * dVar[i]) / (size);
        }
        return derMean;
    }

    private void derNorm(NNTensor errors, NNVector errorMean, NNVector errorVar, int n) {
        errorMean.div(size);
        errorVar.mul(2.0f / (size));

        float[] dVar = new float[var[n].size()];
        for (int i = 0; i < var[n].size(); i++) {
            dVar[i] = (float) (1.0 / Math.sqrt(var[n].getData()[i] + epsilon));
        }

        int index = 0;
        for (int j = 0; j < size; j++) {
            for (int k = 0; k < depth; k++, index++) {
                error[n].getData()[index] = errors.getData()[index] * dVar[k] + errorVar.get(k) *
                        (input[n].get(index) - mean[n].get(k)) + errorMean.get(k);
            }
        }
    }

    private void derivativeWeight(NNTensor error, int n) {
        int size = error.getColumns() * error.getRows();
        int index = 0;
        for (int j = 0; j < size; j++) {
            for (int k = 0; k < depth; k++, index++) {
                derBetta.getData()[k] += error.getData()[index];
                derGamma.getData()[k] += error.getData()[index] * ((output[n].getData()[index] - betta.get(k)) / gamma.get(k));
            }
        }
    }

    public InstanceNormalizationLayer3D setTrainable(boolean trainable) {
        this.trainable = trainable;

        return this;
    }

    public InstanceNormalizationLayer3D setRegularization(Regularization regularization) {
        this.regularization = regularization;

        return this;
    }

    @Override
    public int info() {
        int countParam = betta.size() * 4;
        System.out.println("Instanc norm| " + height + ",\t" + width + ",\t" + depth + "\t| "
                + outHeight + ",\t" + outWidth + ",\t" + outDepth + "\t|\t" + countParam);

        return countParam;
    }

    @Override
    public void save(FileWriter writer) throws IOException {
        writer.write("Instance normalization layer 3D\n");
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

    public static InstanceNormalizationLayer3D read(Scanner scanner) {
        InstanceNormalizationLayer3D layer = new InstanceNormalizationLayer3D();
        layer.loadWeight = false;
        layer.gamma = NNVector.read(scanner);
        layer.betta = NNVector.read(scanner);
        layer.setRegularization(Regularization.read(scanner));
        layer.setTrainable(Boolean.parseBoolean(scanner.nextLine()));
        layer.loadWeight = true;
        return layer;
    }
}

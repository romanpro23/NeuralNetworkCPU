package neural_network.layers.dense;

import lombok.Setter;
import neural_network.optimizers.Optimizer;
import neural_network.regularization.Regularization;
import nnarrays.NNArray;
import nnarrays.NNArrays;
import nnarrays.NNVector;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;

public class BatchNormalizationLayer extends DenseNeuralLayer {
    //trainable parts
    private Regularization regularization;
    private boolean trainable;
    @Setter
    private boolean loadWeight;

    private final float momentum;

    //betta
    @Setter
    private NNVector betta;
    private NNVector derBetta;
    private NNVector[] optimizeBetta;
    //gamma
    @Setter
    private NNVector gamma;
    private NNVector derGamma;
    private NNVector[] optimizeGamma;

    private NNVector movingMean;
    private NNVector movingVar;

    private NNVector[] normOutput;

    private NNVector mean, var;

    public BatchNormalizationLayer() {
        this(0.99);
    }

    public BatchNormalizationLayer(double momentum) {
        this.momentum = (float) momentum;
        this.trainable = true;
    }

    @Override
    public void initialize(int[] size) {
        if (size.length != 1) {
            throw new ExceptionInInitializerError("Error size pre layer!");
        }
        countNeuron = size[0];
        if (!loadWeight) {
            gamma = new NNVector(countNeuron);
            derGamma = new NNVector(countNeuron);
            movingMean = new NNVector(countNeuron);
            mean = new NNVector(countNeuron);
            movingVar = new NNVector(countNeuron);
            var = new NNVector(countNeuron);
            betta = new NNVector(countNeuron);
            derBetta = new NNVector(countNeuron);
            gamma.fill(1);
            movingVar.fill(1);
        }
    }

    @Override
    public void initialize(Optimizer optimizer) {
        if (optimizer.getCountParam() > 0) {
            optimizeGamma = new NNVector[optimizer.getCountParam()];
            optimizeBetta = new NNVector[optimizer.getCountParam()];

            for (int i = 0; i < optimizer.getCountParam(); i++) {
                optimizeGamma[i] = new NNVector(gamma);
                optimizeBetta[i] = new NNVector(betta);
            }
        }
    }

    @Override
    public void update(Optimizer optimizer) {
        if (trainable) {
            if (input.length != 1) {
                derBetta.div(input.length);
                derGamma.div(input.length);
            }

            if (optimizer.getClipValue() != 0) {
                derBetta.clip(optimizer.getClipValue());
                derGamma.clip(optimizer.getClipValue());
            }

            if (regularization != null) {
                regularization.regularization(betta);
                regularization.regularization(gamma);
            }

            optimizer.updateWeight(betta, derBetta, optimizeBetta);
            optimizer.updateWeight(gamma, derGamma, optimizeGamma);
        }
    }

    @Override
    public void generateOutput(NNArray[] input) {
        this.input = NNArrays.isVector(input);
        this.output = new NNVector[input.length];
        this.normOutput = new NNVector[input.length];

        normalization(movingMean, movingVar);
    }

    private void normalization(NNVector mean, NNVector var) {
        float[] varSqrt = new float[var.size()];
        for (int i = 0; i < var.size(); i++) {
            varSqrt[i] = (float) (Math.sqrt(var.getData()[i]) + 0.0000001f);
        }
        for (int i = 0; i < input.length; i++) {
            output[i] = new NNVector(countNeuron);
            normOutput[i] = new NNVector(countNeuron);
            for (int j = 0; j < input[i].size(); j++) {
                normOutput[i].getData()[j] = (input[i].getData()[j] - mean.get(j)) / varSqrt[j];
                output[i].getData()[j] = normOutput[i].getData()[j] * gamma.get(j) + betta.get(j);
            }
        }
    }

    @Override
    public void generateTrainOutput(NNArray[] input) {
        this.input = NNArrays.isVector(input);
        this.output = new NNVector[input.length];
        this.normOutput = new NNVector[input.length];

        findMean();
        findVariance();

        movingMean.momentum(mean, momentum);
        movingVar.momentum(var, momentum);

        normalization(mean, var);
    }

    private void findMean() {
        mean.clear();
        for (NNVector vector : input) {
            for (int j = 0; j < vector.size(); j++) {
                mean.getData()[j] += vector.getData()[j];
            }
        }
        mean.div(input.length);
    }

    private void findVariance() {
        var.clear();
        float sub;
        for (NNVector vector : input) {
            for (int j = 0; j < vector.size(); j++) {
                sub = vector.getData()[j] - mean.getData()[j];
                var.getData()[j] += sub * sub;
            }
        }
        var.div(input.length);
    }

    @Override
    public void generateError(NNArray[] errors) {
        errorNL = getErrorNextLayer(errors);
        this.error = new NNVector[errors.length];

        for (int i = 0; i < errorNL.length; i++) {
            error[i] = new NNVector(countNeuron);
            for (int j = 0; j < errorNL[i].size(); j++) {
                error[i].getData()[j] = errorNL[i].getData()[j] * gamma.get(j);
            }
        }

        NNVector errorVariance = derVar(error);
        NNVector errorMean = derMean(error, errorVariance);
        derNorm(errorVariance, errorMean);

        if (trainable) {
            derivativeWeight(errorNL);
        }
    }

    public NNVector derVar(NNVector[] error) {
        NNVector derVariance = new NNVector(var);
        float[] dVar = new float[var.size()];
        for (int i = 0; i < var.size(); i++) {
            dVar[i] = (float) (-0.5 * Math.pow(var.getData()[i] + 0.00000001f, -1.5));
        }

        for (int i = 0; i < error.length; i++) {
            for (int j = 0; j < error[i].size(); j++) {
                derVariance.getData()[j] += error[i].get(j) * (input[i].get(j) - mean.get(j));
            }
        }
        for (int i = 0; i < derVariance.size(); i++) {
            derVariance.getData()[i] *= dVar[i];
        }
        return derVariance;
    }

    public NNVector derMean(NNVector[] error, NNVector derVar) {
        NNVector derMean = new NNVector(mean.size());
        float[] dMean = new float[mean.size()];
        float[] dVar = new float[var.size()];
        for (int i = 0; i < var.size(); i++) {
            dMean[i] = (float) (-1.0f / Math.sqrt(var.getData()[i] + 0.00000001f));
        }

        for (int i = 0; i < error.length; i++) {
            for (int j = 0; j < error[i].size(); j++) {
                derMean.getData()[j] += error[i].get(j);
                dVar[j] += input[i].getData()[j] - mean.get(j);
            }
        }
        for (int i = 0; i < derMean.size(); i++) {
            derMean.getData()[i] *= dMean[i];
            derMean.getData()[i] += (-2 * derVar.get(i) * dVar[i]) / error.length;
        }
        return derMean;
    }

    public void derNorm(NNVector derMean, NNVector derVar) {
        derMean.div(error.length);
        derVar.mul(2.0f / error.length);

        float[] dVar = new float[var.size()];
        for (int i = 0; i < var.size(); i++) {
            dVar[i] = (float) (1.0 / Math.sqrt(var.getData()[i] + 0.00000001f));
        }

        for (int i = 0; i < error.length; i++) {
            for (int j = 0; j < error[i].size(); j++) {
                error[i].getData()[j] = error[i].get(j) * dVar[j] + derVar.get(j) *
                        (input[i].get(j) - mean.get(j)) + derMean.get(j);
            }
        }
    }

    private void derivativeWeight(NNVector[] error) {
        for (int i = 0; i < error.length; i++) {
            for (int j = 0; j < error[i].size(); j++) {
                derBetta.getData()[j] += error[i].getData()[j];
                derGamma.getData()[j] += error[i].getData()[j] * normOutput[i].getData()[j];
            }
        }
    }

    public BatchNormalizationLayer setTrainable(boolean trainable) {
        this.trainable = trainable;

        return this;
    }

    public BatchNormalizationLayer setRegularization(Regularization regularization) {
        this.regularization = regularization;

        return this;
    }

    @Override
    public int info() {
        int countParam = betta.size() * 4;
        System.out.println("Batch norm\t|  " + countNeuron + "\t\t\t|  " + countNeuron + "\t\t\t|\t" + countParam);

        return countParam;
    }

    @Override
    public void write(FileWriter writer) throws IOException {
        writer.write("Batch normalization layer\n");
        writer.write(momentum + "\n");
        gamma.save(writer);
        betta.save(writer);
        movingMean.save(writer);
        movingVar.save(writer);

        if (regularization != null) {
            regularization.write(writer);
        } else {
            writer.write("null\n");
        }
        writer.write(trainable + "\n");
        writer.flush();
    }

    public static BatchNormalizationLayer read(Scanner scanner) {
        BatchNormalizationLayer layer = new BatchNormalizationLayer(Float.parseFloat(scanner.nextLine()));
        layer.loadWeight = false;
        layer.gamma = NNVector.read(scanner);
        layer.betta = NNVector.read(scanner);
        layer.movingMean = NNVector.read(scanner);
        layer.movingVar = NNVector.read(scanner);
        layer.setRegularization(Regularization.read(scanner));
        layer.setTrainable(Boolean.parseBoolean(scanner.nextLine()));
        layer.loadWeight = true;
        return layer;
    }
}

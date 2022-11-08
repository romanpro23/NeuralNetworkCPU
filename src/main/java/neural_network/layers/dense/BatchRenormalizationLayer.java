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

public class BatchRenormalizationLayer extends DenseNeuralLayer {
    //trainable parts
    private Regularization regularization;
    @Setter
    private boolean loadWeight;

    private final float momentum;
    private float rMax, dMax;

    //betta
    @Setter
    private NNVector betta;
    private NNVector derBetta;
    //gamma
    @Setter
    private NNVector gamma;
    private NNVector derGamma;

    private NNVector movingMean;
    private NNVector movingVar;

    private NNVector mean, var, r;
    private NNVector[] renormOutput;

    public BatchRenormalizationLayer() {
        this(0.99);
    }

    public BatchRenormalizationLayer(double momentum) {
        this(momentum, 1, 0);
    }

    public BatchRenormalizationLayer(double momentum, double rMax, double dMax) {
        this.momentum = (float) momentum;
        this.rMax = (float) rMax;
        this.dMax = (float) dMax;
        this.trainable = true;
    }

    public BatchRenormalizationLayer setRMax(double rMax) {
        this.rMax = (float) rMax;

        return this;
    }

    public BatchRenormalizationLayer setDMax(double dMax) {
        this.dMax = (float) dMax;

        return this;
    }

    public BatchRenormalizationLayer setMaxParamValue(double rMax, double dMax) {
        this.rMax = (float) rMax;
        this.dMax = (float) dMax;

        return this;
    }

    @Override
    public void initialize(int[] size) {
        if (size.length != 1) {
            throw new ExceptionInInitializerError("Error size pre layer!");
        }
        countNeuron = size[0];
        r = new NNVector(countNeuron);
        derBetta = new NNVector(countNeuron);
        derGamma = new NNVector(countNeuron);

        mean = new NNVector(countNeuron);
        var = new NNVector(countNeuron);

        if (!loadWeight) {
            movingMean = new NNVector(countNeuron);
            movingVar = new NNVector(countNeuron);

            gamma = new NNVector(countNeuron);
            betta = new NNVector(countNeuron);

            gamma.fill(1);
            movingVar.fill(1);
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
        this.renormOutput = new NNVector[input.length];

        findMean();
        findVariance();
        var.sqrt();

        r = NNArrays.div(var, movingVar);
        r.clip(1.0f / rMax, rMax);
        NNVector d = movingAverages();
        d.clip(dMax);

        renormalization(mean, var, r, d);
    }

    private NNVector movingAverages() {
        NNVector result = new NNVector(countNeuron);
        for (int i = 0; i < countNeuron; i++) {
            result.getData()[i] = (mean.get(i) - movingMean.get(i)) / movingVar.get(i);
        }

        return result;
    }

    private void renormalization(NNVector mean, NNVector var, NNVector r, NNVector d) {
        for (int i = 0; i < input.length; i++) {
            output[i] = new NNVector(countNeuron);
            renormOutput[i] = new NNVector(countNeuron);
            for (int j = 0; j < input[i].size(); j++) {
                renormOutput[i].getData()[j] = ((input[i].getData()[j] - mean.get(j)) / var.get(j)) * r.get(j) + d.get(j);
                output[i].getData()[j] = renormOutput[i].getData()[j] * gamma.get(j) + betta.get(j);
            }
        }
    }

    @Override
    public void generateTrainOutput(NNArray[] input) {
        this.input = NNArrays.isVector(input);
        this.output = new NNVector[input.length];
        this.renormOutput = new NNVector[input.length];

        findMean();
        findVariance();
        var.sqrt();

        r = NNArrays.div(var, movingVar);
        r.clip(1.0f / rMax, rMax);
        NNVector d = movingAverages();
        d.clip(dMax);

        renormalization(mean, var, r, d);
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
        var.sqrt();
    }

    @Override
    public void generateError(NNArray[] errors) {
        errorNL = getErrorNextLayer(errors);
        this.error = new NNVector[errors.length];
        NNVector[] errorRenorm = new NNVector[errors.length];

        for (int i = 0; i < errorNL.length; i++) {
            error[i] = new NNVector(countNeuron);
            errorRenorm[i] = new NNVector(countNeuron);
            for (int j = 0; j < errorNL[i].size(); j++) {
                errorRenorm[i].getData()[j] = errorNL[i].getData()[j] * gamma.get(j);
            }
        }

        NNVector errorVariance = derVar(errorRenorm, mean, var);
        NNVector errorMean = derMean(errorRenorm, mean, var);

        derRenorm(errorRenorm, errorMean, errorVariance, mean, var);

        if (trainable) {
            movingMean.momentumAverage(mean, momentum);
            movingVar.momentumAverage(var, momentum);
            derivativeWeight(errorNL);
        }
    }

    private NNVector derVar(NNVector[] error, NNVector mean, NNVector var) {
        NNVector derVariance = new NNVector(var);
        float[] dVar = new float[var.size()];
        for (int i = 0; i < var.size(); i++) {
            dVar[i] = -r.get(i) / (var.get(i) * var.get(i));
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

    private NNVector derMean(NNVector[] error, NNVector mean, NNVector var) {
        NNVector derMean = new NNVector(mean.size());
        float[] dMean = new float[mean.size()];
        for (int i = 0; i < var.size(); i++) {
            dMean[i] = -r.get(i) / var.get(i);
        }

        for (NNVector vector : error) {
            for (int j = 0; j < vector.size(); j++) {
                derMean.getData()[j] += vector.get(j);
            }
        }
        for (int i = 0; i < derMean.size(); i++) {
            derMean.getData()[i] *= dMean[i];
        }
        return derMean;
    }

    private void derRenorm(NNVector[] errors, NNVector errorMean, NNVector errorVar, NNVector mean, NNVector var) {
        errorMean.div(errors.length);

        float[] dVar = new float[var.size()];
        float[] dVarM = new float[var.size()];
        for (int i = 0; i < var.size(); i++) {
            dVar[i] = r.get(i) / var.get(i);
            dVarM[i] = errors.length * var.get(i);
        }

        for (int i = 0; i < error.length; i++) {
            for (int j = 0; j < error[i].size(); j++) {
                error[i].getData()[j] = errors[i].getData()[j] * dVar[j] + errorVar.get(j) *
                        (input[i].get(j) - mean.get(j)) / dVarM[j] + errorMean.get(j);
            }
        }
    }

    private void derivativeWeight(NNVector[] error) {
        for (int i = 0; i < error.length; i++) {
            for (int j = 0; j < error[i].size(); j++) {
                derBetta.getData()[j] += error[i].getData()[j];
                derGamma.getData()[j] += error[i].getData()[j] * renormOutput[i].getData()[j];
            }
        }

        if (regularization != null) {
            regularization.regularization(betta);
            regularization.regularization(gamma);
        }
    }

    public BatchRenormalizationLayer setTrainable(boolean trainable) {
        this.trainable = trainable;

        return this;
    }

    public BatchRenormalizationLayer setRegularization(Regularization regularization) {
        this.regularization = regularization;

        return this;
    }

    @Override
    public int info() {
        int countParam = betta.size() * 4;
        System.out.println("Batch renorm|  " + countNeuron + "\t\t\t|  " + countNeuron + "\t\t\t|\t" + countParam);
        return countParam;
    }

    @Override
    public void write(FileWriter writer) throws IOException {
        writer.write("Batch renormalization layer\n");
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

    public static BatchRenormalizationLayer read(Scanner scanner) {
        BatchRenormalizationLayer layer = new BatchRenormalizationLayer(Float.parseFloat(scanner.nextLine()));
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

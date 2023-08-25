package neural_network.layers.layer_3d;

import lombok.Setter;
import neural_network.optimizers.Optimizer;
import neural_network.regularization.Regularization;
import nnarrays.NNArray;
import nnarrays.NNArrays;
import nnarrays.NNTensor;
import nnarrays.NNVector;
import utilities.CublasUtil;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;

public class BatchRenormalizationLayer3D extends NeuralLayer3D {
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
    private NNTensor[] renormOutput;
    private int size;

    public BatchRenormalizationLayer3D() {
        this(0.99);
    }

    public BatchRenormalizationLayer3D(double momentum) {
        this(momentum, 1, 0);
    }

    public BatchRenormalizationLayer3D(double momentum, double rMax, double dMax) {
        this.momentum = (float) momentum;
        this.rMax = (float) rMax;
        this.dMax = (float) dMax;
        this.trainable = true;
    }

    public BatchRenormalizationLayer3D setRMax(double rMax) {
        this.rMax = (float) rMax;

        return this;
    }

    public BatchRenormalizationLayer3D setDMax(double dMax) {
        this.dMax = (float) dMax;

        return this;
    }

    public BatchRenormalizationLayer3D setMaxParamValue(double rMax, double dMax) {
        this.rMax = (float) rMax;
        this.dMax = (float) dMax;

        return this;
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
        r = new NNVector(depth);
        derBetta = new NNVector(depth);
        derGamma = new NNVector(depth);

        mean = new NNVector(depth);
        var = new NNVector(depth);

        if (!loadWeight) {
            movingMean = new NNVector(depth);
            movingVar = new NNVector(depth);

            gamma = new NNVector(depth);
            betta = new NNVector(depth);

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
        this.input = NNArrays.isTensor(input);
        this.output = new NNTensor[input.length];
        this.renormOutput = new NNTensor[input.length];

        findMean();
        findVariance();
        var.sqrt();

        r = NNArrays.div(var, movingVar);
        r.clip(1.0f / rMax, rMax);
        NNVector d = movingAverages();
        d.clip(dMax);

        renormalization(mean, var, r, d);
    }

    @Override
    public void generateOutput(CublasUtil.Matrix[] input_gpu) {

    }

    private NNVector movingAverages() {
        NNVector result = new NNVector(depth);
        for (int i = 0; i < depth; i++) {
            result.getData()[i] = (mean.get(i) - movingMean.get(i)) / movingVar.get(i);
        }

        return result;
    }

    private void renormalization(NNVector mean, NNVector var, NNVector r, NNVector d) {
        for (int i = 0; i < input.length; i++) {
            output[i] = new NNTensor(outHeight, outWidth, outDepth);
            renormOutput[i] = new NNTensor(outHeight, outWidth, outDepth);
            int index = 0;
            for (int j = 0; j < size; j++) {
                for (int k = 0; k < depth; k++, index++) {
                    renormOutput[i].getData()[index] = ((input[i].getData()[index] - mean.get(index)) / var.get(index))
                            * r.get(index) + d.get(index);
                    output[i].getData()[index] = renormOutput[i].getData()[index] * gamma.get(k) + betta.get(k);
                }
            }
        }
    }

    @Override
    public void generateTrainOutput(NNArray[] input) {
        this.input = NNArrays.isTensor(input);
        this.output = new NNTensor[input.length];
        this.renormOutput = new NNTensor[input.length];

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
        for (NNTensor tensor : input) {
            int index = 0;
            for (int j = 0; j < size; j++) {
                for (int k = 0; k < depth; k++, index++) {
                    mean.getData()[k] += tensor.getData()[index];
                }
            }
        }
        mean.div(input.length * size);
    }

    private void findVariance() {
        var.clear();
        float sub;
        for (NNTensor tensor : input) {
            int index = 0;
            for (int j = 0; j < size; j++) {
                for (int k = 0; k < depth; k++, index++) {
                    sub = tensor.getData()[index] - mean.getData()[k];
                    var.getData()[k] += sub * sub;
                }
            }
        }
        var.div(input.length * size);
        var.sqrt();
    }

    @Override
    public void generateError(NNArray[] errors) {
        errorNL = getErrorNextLayer(errors);
        this.error = new NNTensor[errors.length];
        NNTensor[] errorRenorm = new NNTensor[errors.length];

        for (int i = 0; i < errorNL.length; i++) {
            error[i] = new NNTensor(outHeight, outWidth, outDepth);
            errorRenorm[i] = new NNTensor(outHeight, outWidth, outDepth);
            int index = 0;
            for (int j = 0; j < size; j++) {
                for (int k = 0; k < depth; k++, index++) {
                    errorRenorm[i].getData()[index] = errorNL[i].getData()[index] * gamma.get(k);
                }
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

    @Override
    public CublasUtil.Matrix[] getOutput_gpu() {
        return new CublasUtil.Matrix[0];
    }

    @Override
    public CublasUtil.Matrix[] getError_gpu() {
        return new CublasUtil.Matrix[0];
    }

    private NNVector derVar(NNTensor[] error, NNVector mean, NNVector var) {
        NNVector derVariance = new NNVector(var);
        float[] dVar = new float[var.size()];
        for (int i = 0; i < var.size(); i++) {
            dVar[i] = -r.get(i) / (var.get(i) * var.get(i));
        }

        for (int i = 0; i < error.length; i++) {
            int index = 0;
            for (int j = 0; j < size; j++) {
                for (int k = 0; k < depth; k++, index++) {
                    derVariance.getData()[k] += error[i].get(index) * (input[i].get(index) - mean.get(k));
                }
            }
        }
        for (int i = 0; i < derVariance.size(); i++) {
            derVariance.getData()[i] *= dVar[i];
        }
        return derVariance;
    }

    private NNVector derMean(NNTensor[] error, NNVector mean, NNVector var) {
        NNVector derMean = new NNVector(mean.size());
        float[] dMean = new float[mean.size()];
        for (int i = 0; i < var.size(); i++) {
            dMean[i] = -r.get(i) / var.get(i);
        }

        for (NNTensor tensor : error) {
            int index = 0;
            for (int j = 0; j < size; j++) {
                for (int k = 0; k < depth; k++, index++) {
                    derMean.getData()[k] += tensor.get(index);
                }
            }
        }
        for (int i = 0; i < derMean.size(); i++) {
            derMean.getData()[i] *= dMean[i];
        }
        return derMean;
    }

    private void derRenorm(NNTensor[] errors, NNVector errorMean, NNVector errorVar, NNVector mean, NNVector var) {
        errorMean.div(errors.length * size);

        float[] dVar = new float[var.size()];
        float[] dVarM = new float[var.size()];
        for (int i = 0; i < var.size(); i++) {
            dVar[i] = r.get(i) / var.get(i);
            dVarM[i] = errors.length * size * var.get(i);
        }

        for (int i = 0; i < error.length; i++) {
            int index = 0;
            for (int j = 0; j < size; j++) {
                for (int k = 0; k < depth; k++, index++) {
                    error[i].getData()[index] = errors[i].getData()[index] * dVar[k] + errorVar.get(k) *
                            (input[i].get(index) - mean.get(k)) / dVarM[j] + errorMean.get(k);
                }
            }
        }
    }

    private void derivativeWeight(NNTensor[] error) {
        for (int i = 0; i < error.length; i++) {
            int index = 0;
            for (int j = 0; j < size; j++) {
                for (int k = 0; k < depth; k++, index++) {
                    derBetta.getData()[k] += error[i].getData()[index];
                    derGamma.getData()[k] += error[i].getData()[index] * renormOutput[i].getData()[index];
                }
            }
        }

        if (regularization != null) {
            regularization.regularization(betta);
            regularization.regularization(gamma);
        }
    }

    public BatchRenormalizationLayer3D setTrainable(boolean trainable) {
        this.trainable = trainable;

        return this;
    }

    public BatchRenormalizationLayer3D setRegularization(Regularization regularization) {
        this.regularization = regularization;

        return this;
    }

    @Override
    public int info() {
        int countParam = betta.size() * 4;
        System.out.println("Batch renorm|  "  + height + ",\t" + width + ",\t" + depth + "\t|  "
                + outHeight + ",\t" + outWidth + ",\t" + outDepth + "\t|\t" + countParam);
        return countParam;
    }

    @Override
    public void save(FileWriter writer) throws IOException {
        writer.write("Batch renormalization layer 3D\n");
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

    public static BatchRenormalizationLayer3D read(Scanner scanner) {
        BatchRenormalizationLayer3D layer = new BatchRenormalizationLayer3D(Float.parseFloat(scanner.nextLine()));
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

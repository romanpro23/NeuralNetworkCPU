package neural_network.layers.convolution_3d;

import lombok.Setter;
import neural_network.layers.dense.DenseNeuralLayer;
import neural_network.optimizers.Optimizer;
import neural_network.regularization.Regularization;
import nnarrays.NNArray;
import nnarrays.NNArrays;
import nnarrays.NNTensor;
import nnarrays.NNVector;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;

public class BatchNormalizationLayer3D extends ConvolutionNeuralLayer {
    //trainable parts
    private Regularization regularization;
    private boolean trainable;
    @Setter
    private boolean loadWeight;

    private final float momentum;
    private final float epsilon;

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

    private NNTensor[] normOutput;

    private NNVector mean, var;

    private int size;

    public BatchNormalizationLayer3D() {
        this(0.99);
    }

    public BatchNormalizationLayer3D(double momentum) {
        this.momentum = (float) momentum;
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

        mean = new NNVector(depth);
        var = new NNVector(depth);
        derBetta = new NNVector(depth);
        derGamma = new NNVector(depth);

        if (!loadWeight) {
            movingMean = new NNVector(depth);
            movingVar = new NNVector(depth);

            betta = new NNVector(depth);
            gamma = new NNVector(depth);

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
        this.normOutput = new NNTensor[input.length];

        normalization(movingMean, movingVar);
    }

    private void normalization(NNVector mean, NNVector var) {
        float[] varSqrt = new float[var.size()];
        for (int i = 0; i < var.size(); i++) {
            varSqrt[i] = (float) (Math.sqrt(var.getData()[i] + epsilon));
        }
        for (int i = 0; i < input.length; i++) {
            output[i] = new NNTensor(outHeight, outWidth, outDepth);
            normOutput[i] = new NNTensor(outHeight, outWidth, outDepth);
            int index = 0;
            for (int j = 0; j < size; j++) {
                for (int k = 0; k < depth; k++, index++) {
                    normOutput[i].getData()[index] = (input[i].getData()[index] - mean.get(k)) / varSqrt[k];
                    output[i].getData()[index] = normOutput[i].getData()[index] * gamma.get(k) + betta.get(k);
                }
            }
        }
    }

    @Override
    public void generateTrainOutput(NNArray[] input) {
        this.input = NNArrays.isTensor(input);
        this.output = new NNTensor[input.length];
        this.normOutput = new NNTensor[input.length];

        findMean();
        findVariance();

        movingMean.momentum(mean, momentum);
        movingVar.momentum(var, momentum);

        normalization(mean, var);
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
    }

    @Override
    public void generateError(NNArray[] errors) {
        errorNL = getErrorNextLayer(errors);
        this.error = new NNTensor[errors.length];
        NNTensor[] errorNorm = new NNTensor[errors.length];

        for (int i = 0; i < errorNL.length; i++) {
            error[i] = new NNTensor(outHeight, outWidth, outDepth);
            errorNorm[i] = new NNTensor(outHeight, outWidth, outDepth);
            int index = 0;
            for (int j = 0; j < size; j++) {
                for (int k = 0; k < depth; k++, index++) {
                    errorNorm[i].getData()[index] = errorNL[i].getData()[index] * gamma.get(k);
                }
            }
        }

        NNVector errorVariance = derVar(errorNorm);
        NNVector errorMean = derMean(errorNorm, errorVariance);

        derNorm(errorNorm, errorMean, errorVariance);

        if (trainable) {
            derivativeWeight(errorNL);
        }
    }

    private NNVector derVar(NNTensor[] error) {
        NNVector derVariance = new NNVector(var);
        float[] dVar = new float[var.size()];
        for (int i = 0; i < var.size(); i++) {
            dVar[i] = (float) (-0.5 * Math.pow(var.get(i) + epsilon, -1.5));
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

    private NNVector derMean(NNTensor[] error, NNVector derVar) {
        NNVector derMean = new NNVector(mean.size());
        float[] dMean = new float[mean.size()];
        float[] dVar = new float[var.size()];
        for (int i = 0; i < var.size(); i++) {
            dMean[i] = (float) (-1.0f / Math.sqrt(var.getData()[i] + epsilon));
        }

        for (int i = 0; i < error.length; i++) {
            int size = error[i].getRow() * error[i].getDepth();
            int index = 0;
            for (int j = 0; j < size; j++) {
                for (int k = 0; k < depth; k++, index++) {
                    derMean.getData()[k] += error[i].get(index);
                    dVar[k] += input[i].get(index) - mean.get(k);
                }
            }
        }
        for (int i = 0; i < derMean.size(); i++) {
            derMean.getData()[i] *= dMean[i];
            derMean.getData()[i] += (-2.0f * derVar.get(i) * dVar[i]) / (error.length * size);
        }
        return derMean;
    }

    private void derNorm(NNTensor[] errors, NNVector errorMean, NNVector errorVar) {
        errorMean.div(errors.length * size);
        errorVar.mul(2.0f / (errors.length * size));

        float[] dVar = new float[var.size()];
        for (int i = 0; i < var.size(); i++) {
            dVar[i] = (float) (1.0 / Math.sqrt(var.getData()[i] + epsilon));
        }

        for (int i = 0; i < error.length; i++) {
            int size = error[i].getRow() * error[i].getDepth();
            int index = 0;
            for (int j = 0; j < size; j++) {
                for (int k = 0; k < depth; k++, index++) {
                    error[i].getData()[index] = errors[i].getData()[index] * dVar[k] + errorVar.get(k) *
                            (input[i].get(index) - mean.get(k)) + errorMean.get(k);
                }
            }
        }
    }

    private void derivativeWeight(NNTensor[] error) {
        for (int i = 0; i < error.length; i++) {
            int size = error[i].getRow() * error[i].getDepth();
            int index = 0;
            for (int j = 0; j < size; j++) {
                for (int k = 0; k < depth; k++, index++) {
                    derBetta.getData()[k] += error[i].getData()[index];
                    derGamma.getData()[k] += error[i].getData()[index] * normOutput[i].getData()[index];
                }
            }
        }

        if (input.length != 1) {
            derBetta.div(input.length * size);
            derGamma.div(input.length * size);
        }

        if (regularization != null) {
            regularization.regularization(betta);
            regularization.regularization(gamma);
        }
    }

    public BatchNormalizationLayer3D setTrainable(boolean trainable) {
        this.trainable = trainable;

        return this;
    }

    public BatchNormalizationLayer3D setRegularization(Regularization regularization) {
        this.regularization = regularization;

        return this;
    }

    @Override
    public int info() {
        int countParam = betta.size() * 4;
        System.out.println("Batch norm\t|  " + height + ",\t" + width + ",\t" + depth + "\t|  "
                + outHeight + ",\t" + outWidth + ",\t" + outDepth + "\t|\t" + countParam);

        return countParam;
    }

    @Override
    public void write(FileWriter writer) throws IOException {
        writer.write("Batch normalization layer 3D\n");
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

    public static BatchNormalizationLayer3D read(Scanner scanner) {
        BatchNormalizationLayer3D layer = new BatchNormalizationLayer3D(Float.parseFloat(scanner.nextLine()));
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

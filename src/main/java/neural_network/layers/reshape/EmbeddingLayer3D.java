package neural_network.layers.reshape;

import lombok.Setter;
import neural_network.initialization.Initializer;
import neural_network.layers.NeuralLayer;
import neural_network.optimizers.Optimizer;
import neural_network.regularization.Regularization;
import nnarrays.*;
import utilities.CublasUtil;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Scanner;

public class EmbeddingLayer3D extends NeuralLayer {
    //trainable parts
    private Regularization regularization;
    private Initializer initializer;
    @Setter
    private boolean loadWeight;

    protected NNVector[] input;
    protected NNTensor[] output;
    protected NNTensor[] errorNL;

    private final int width;
    private final int height;
    private final int depth;
    private int countLabel;

    private NNMatrix weight;
    private NNMatrix derWeight;

    public EmbeddingLayer3D(int width, int height, int depth) {
        this.width = width;
        this.height = height;
        this.depth = depth;

        initializer = new Initializer.RandomUniform();
        trainable = true;
    }

    @Override
    public void initialize(int[] size) {
        if (size.length != 1) {
            throw new ExceptionInInitializerError("Error size pre layer!");
        }
        this.countLabel = size[0];
        derWeight = new NNMatrix(countLabel, height * width * depth);

        if (!loadWeight) {
            weight = new NNMatrix(countLabel, height * width * depth);
            initializer.initialize(weight);
        }
    }

    @Override
    public void generateOutput(NNArray[] input) {
        this.input = NNArrays.isVector(input);
        this.output = new NNTensor[input.length];

        int index, indexOut;
        for (int i = 0; i < output.length; i++) {
            this.output[i] = new NNTensor(height, width, depth);
            index = this.input[i].indexMaxElement();
            indexOut = weight.getRowIndex()[index];
            System.arraycopy(weight.getData(), indexOut, output[i].getData(), 0, this.output[i].size());
        }
    }

    @Override
    public void generateOutput(CublasUtil.Matrix[] input_gpu) {

    }

    @Override
    public void generateTrainOutput(NNArray[] input) {
        generateOutput(input);
    }

    @Override
    public void generateError(NNArray[] error) {
        if (trainable) {
            errorNL = NNArrays.isTensor(error);
            int index, indexOut;

            for (int i = 0; i < output.length; i++) {
                index = this.input[i].indexMaxElement();
                indexOut = weight.getRowIndex()[index];
                System.arraycopy(errorNL[i].getData(), 0, derWeight.getData(), indexOut, errorNL[i].size());
            }

            if (regularization != null) {
                regularization.regularization(weight);
            }
        }
    }

    @Override
    public int[] size() {
        return new int[]{height, width, depth};
    }

    @Override
    public void initialize(Optimizer optimizer) {
        optimizer.addDataOptimize(weight, derWeight);
    }

    @Override
    public int info() {
        int countParam = weight.size();
        System.out.println("Embedding\t|  " + countLabel + "\t\t\t|  " + height + ",\t" + width + ",\t" + depth + "\t|\t" + countParam);
        return countParam;
    }

    @Override
    public void save(FileWriter writer) throws IOException {
        writer.write("Embedding layer 3D\n");
        writer.write(height + " " + width + " " + depth + "\n");
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
    public NNArray[] getOutput() {
        return output;
    }

    @Override
    public CublasUtil.Matrix[] getOutput_gpu() {
        return new CublasUtil.Matrix[0];
    }

    @Override
    public NNArray[] getError() {
        return errorNL;
    }

    @Override
    public CublasUtil.Matrix[] getError_gpu() {
        return new CublasUtil.Matrix[0];
    }

    public static EmbeddingLayer3D read(Scanner scanner) {
        int[] arr = Arrays.stream(scanner.nextLine().split(" ")).mapToInt(Integer::parseInt).toArray();
        EmbeddingLayer3D layer = new EmbeddingLayer3D(arr[0], arr[1], arr[2]);
        layer.weight = NNMatrix.read(scanner);
        layer.setRegularization(Regularization.read(scanner));
        layer.setTrainable(Boolean.parseBoolean(scanner.nextLine()));
        layer.loadWeight = true;
        return layer;
    }

    public EmbeddingLayer3D setRegularization(Regularization regularization) {
        this.regularization = regularization;
        return this;
    }

    public EmbeddingLayer3D setTrainable(boolean trainable) {
        this.trainable = trainable;
        return this;
    }

    public EmbeddingLayer3D setInitializer(Initializer initializer) {
        this.initializer = initializer;
        return this;
    }
}

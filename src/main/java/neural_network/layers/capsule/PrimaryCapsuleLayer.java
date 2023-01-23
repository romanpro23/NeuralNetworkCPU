package neural_network.layers.capsule;

import lombok.Getter;
import neural_network.initialization.Initializer;
import neural_network.layers.NeuralLayer;
import neural_network.optimizers.Optimizer;
import neural_network.regularization.Regularization;
import nnarrays.*;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Scanner;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class PrimaryCapsuleLayer extends NeuralLayer {
    //trainable parts
    private Regularization regularization;
    private Initializer initializer;

    private boolean loadWeight;

    //weightAttention
    @Getter
    private NNTensor4D weight;
    private NNTensor4D derWeight;

    private final int paddingY;
    private final int paddingX;
    private final int step;
    private final int heightKernel;
    private final int widthKernel;
    private final int sizeVector;
    private final int countCapsule;

    private int depth, width, height;
    private int outDepth, outWidth;
    private int outDepthConv, outWidthConv, outHeightConv;

    private NNTensor[] input;
    private NNMatrix[] output;
    private NNTensor[] error;
    private NNMatrix[] errorNL;

    public PrimaryCapsuleLayer(int countKernel, int countCapsule, int sizeKernel) {
        this(countKernel, countCapsule, sizeKernel, sizeKernel, 1, 0, 0);
    }

    public PrimaryCapsuleLayer(int countKernel, int countCapsule, int sizeKernel, int step) {
        this(countKernel, countCapsule, sizeKernel, sizeKernel, step, 0, 0);
    }

    public PrimaryCapsuleLayer(int countKernel, int countCapsule, int sizeKernel, int step, int padding) {
        this(countKernel, countCapsule, sizeKernel, sizeKernel, step, padding, padding);
    }

    public PrimaryCapsuleLayer(int countKernel, int countCapsule, int heightKernel, int widthKernel, int step, int paddingY, int paddingX) {
        this.sizeVector = countKernel;
        this.countCapsule = countCapsule;
        this.paddingX = paddingX;
        this.paddingY = paddingY;
        this.step = step;
        this.heightKernel = heightKernel;
        this.widthKernel = widthKernel;
        trainable = true;

        initializer = new Initializer.HeNormal();
    }

    @Override
    public void initialize(int[] size) {
        if (size.length != 3) {
            throw new ExceptionInInitializerError("Error size pre layer!");
        }
        this.depth = size[2];
        this.height = size[0];
        this.width = size[1];

        outWidthConv = (width - widthKernel + 2 * paddingX) / step + 1;
        outHeightConv = (height - heightKernel + 2 * paddingY) / step + 1;
        outDepthConv = sizeVector * countCapsule;

        outWidth = outWidthConv * outHeightConv * countCapsule;
        outDepth = sizeVector;

        derWeight = new NNTensor4D(sizeVector * countCapsule, heightKernel, widthKernel, depth);
        if (!loadWeight) {
            weight = new NNTensor4D(sizeVector * countCapsule, heightKernel, widthKernel, depth);
            initializer.initialize(weight);
        }
    }

    @Override
    public int[] size() {
        return new int[]{outWidth, outDepth};
    }

    @Override
    public void initialize(Optimizer optimizer) {
        optimizer.addDataOptimize(weight, derWeight);
    }

    @Override
    public void generateOutput(NNArray[] inputs) {
        this.input = NNArrays.isTensor(inputs);
        output = new NNMatrix[inputs.length];

        ExecutorService executor = Executors.newFixedThreadPool(inputs.length);
        for (int t = 0; t < inputs.length; t++) {
            final int i = t;
            executor.execute(() -> {
                NNTensor outputConv = new NNTensor(outHeightConv, outWidthConv, outDepthConv);
                outputConv.convolution(input[i], weight, step, paddingY, paddingX);
                output[i] = new NNMatrix(outWidth, outDepth, outputConv.getData());
            });
        }
        executor.shutdown();
        while (!executor.isTerminated()) {
        }
    }

    @Override
    public void generateTrainOutput(NNArray[] input) {
        generateOutput(input);
    }

    @Override
    public void generateError(NNArray[] errors) {
        errorNL = getErrorNextLayer(errors);
        error = new NNTensor[errors.length];

        ExecutorService executor = Executors.newFixedThreadPool(input.length);
        for (int t = 0; t < input.length; t++) {
            final int i = t;
            executor.execute(() -> {
                error[i] = new NNTensor(height, width, depth);
                NNTensor errNL = new NNTensor(outHeightConv, outWidthConv, outDepthConv, errorNL[i].getData());
                error[i].transposeConvolution(errNL.stride(step), weight, step, paddingY, paddingX);

                if (trainable) {
                    derWeight.convolution(input[i], errNL, step, paddingY, paddingX);
                }
            });
        }
        executor.shutdown();
        while (!executor.isTerminated()) {
        }

        if (regularization != null && trainable) {
            regularization.regularization(weight);
        }
    }

    @Override
    public NNArray[] getOutput() {
        return output;
    }

    @Override
    public NNArray[] getError() {
        return error;
    }

    @Override
    public int info() {
        int countParam = weight.size();
        System.out.println("Primary caps| " + height + ",\t" + width + ",\t" + depth + "\t| "
                + outWidth + ",\t" + outDepth + "\t\t|\t" + countParam);
        return countParam;
    }

    @Override
    public void save(FileWriter writer) throws IOException {
        writer.write("Primary capsule layer\n");
        writer.write(sizeVector + " " + countCapsule + " " + heightKernel + " " + widthKernel
                + " " + step + " " + paddingY + " " + paddingX + "\n");
        weight.save(writer);
        if (regularization != null) {
            regularization.write(writer);
        } else {
            writer.write("null\n");
        }
        writer.write(trainable + "\n");
        writer.flush();
    }

    public static PrimaryCapsuleLayer read(Scanner scanner) {
        int[] param = Arrays.stream(scanner.nextLine().split(" ")).mapToInt(Integer::parseInt).toArray();

        PrimaryCapsuleLayer layer = new PrimaryCapsuleLayer(param[0], param[1], param[2], param[3], param[4], param[5], param[6]);
        layer.loadWeight = false;
        layer.weight = NNTensor4D.read(scanner);
        layer.setRegularization(Regularization.read(scanner));
        layer.setTrainable(Boolean.parseBoolean(scanner.nextLine()));
        layer.loadWeight = true;
        return layer;
    }

    public PrimaryCapsuleLayer setRegularization(Regularization regularization) {
        this.regularization = regularization;
        return this;
    }

    public PrimaryCapsuleLayer setTrainable(boolean trainable) {
        this.trainable = trainable;
        return this;
    }

    public PrimaryCapsuleLayer setInitializer(Initializer initializer) {
        this.initializer = initializer;
        return this;
    }

    public void setLoadWeight(boolean loadWeight) {
        this.loadWeight = loadWeight;
    }

    public NNMatrix[] getErrorNextLayer(NNArray[] error) {
        NNMatrix[] errorNL = NNArrays.isMatrix(error);

        if (!nextLayers.isEmpty()) {
            for (int i = 0; i < errorNL.length; i++) {
                for (NeuralLayer nextLayer : nextLayers) {
                    errorNL[i].add(nextLayer.getErrorNL()[i]);
                }
            }
        }
        return errorNL;
    }
}

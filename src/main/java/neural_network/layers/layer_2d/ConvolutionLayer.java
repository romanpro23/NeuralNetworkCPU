package neural_network.layers.layer_2d;

import lombok.Setter;
import neural_network.initialization.Initializer;
import neural_network.optimizers.Optimizer;
import neural_network.regularization.Regularization;
import nnarrays.*;
import utilities.CublasUtil;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Scanner;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class ConvolutionLayer extends NeuralLayer2D {
    //trainable parts
    private Regularization regularization;
    private Initializer initializer;

    private boolean loadWeight;

    //weightAttention
    @Setter
    private NNTensor weight;
    private NNTensor derWeight;
    @Setter
    private NNVector threshold;
    private NNVector derThreshold;

    private final int padding;
    private final int step;
    private final int sizeKernel;
    private final int countKernel;

    public ConvolutionLayer(int countKernel, int sizeKernel) {
        this(countKernel, sizeKernel, 1, 0);
    }

    public ConvolutionLayer(int countKernel, int sizeKernel, int step) {
        this(countKernel, sizeKernel, step, 0);
    }

    public ConvolutionLayer(int countKernel, int sizeKernel, int step, int padding) {
        this.countKernel = countKernel;
        this.padding = padding;
        this.step = step;
        this.sizeKernel = sizeKernel;
        trainable = true;

        initializer = new Initializer.XavierUniform();
    }

    @Override
    public void initialize(int[] size) {
        if (size.length != 2) {
            throw new ExceptionInInitializerError("Error size pre layer!");
        }
        this.depth = size[1];
        this.width = size[0];

        outWidth = (width - sizeKernel + 2 * padding) / step + 1;
        outDepth = countKernel;

        derThreshold = new NNVector(countKernel);
        derWeight = new NNTensor(countKernel, sizeKernel, depth);
        if (!loadWeight) {
            weight = new NNTensor(countKernel, sizeKernel, depth);
            threshold = new NNVector(countKernel);
            initializer.initialize(weight);
        }
    }

    @Override
    public void initialize(Optimizer optimizer) {
        optimizer.addDataOptimize(weight, derWeight);
        optimizer.addDataOptimize(threshold, derThreshold);
    }

    @Override
    public void generateError(CublasUtil.Matrix[] errors) {

    }

    @Override
    public void generateOutput(NNArray[] inputs) {
        this.input = NNArrays.isMatrix(inputs);
        output = new NNMatrix[inputs.length];

        ExecutorService executor = Executors.newFixedThreadPool(input.length);
        for (int t = 0; t < input.length; t++) {
            final int i = t;
            executor.execute(() -> {
                output[i] = new NNMatrix(outWidth, outDepth);
                output[i].convolution(input[i], weight, step, padding);
                output[i].add(threshold);
            });
        }
        executor.shutdown();
        while (!executor.isTerminated()) {
        }
    }

    @Override
    public void generateOutput(CublasUtil.Matrix[] input_gpu) {

    }

    @Override
    public void generateError(NNArray[] errors) {
        errorNL = getErrorNextLayer(errors);
        error = new NNMatrix[errors.length];

        ExecutorService executor = Executors.newFixedThreadPool(input.length);
        for (int t = 0; t < input.length; t++) {
            final int i = t;
            executor.execute(() -> {
                error[i] = new NNMatrix(width, depth);
                error[i].transposeConvolution(errorNL[i].stride(step), weight, padding);

                if (trainable) {
                    derWeight.convolution(input[i], errorNL[i], step, padding);
                    derThreshold.add(errorNL[i]);
                }
            });
        }
        executor.shutdown();
        while (!executor.isTerminated()) {
        }

        if (trainable && regularization != null) {
            regularization.regularization(weight);
            regularization.regularization(threshold);
        }
    }

    @Override
    public int info() {
        int countParam = weight.size() + threshold.size();
        System.out.println("Convolution\t| " + width + ",\t" + depth + "\t\t|  " + outWidth + ",\t" + outDepth + "\t\t|\t" + countParam);
        return countParam;
    }

    @Override
    public void save(FileWriter writer) throws IOException {
        writer.write("Convolution layer 2D\n");
        writer.write(countKernel + " " + sizeKernel + " " + step + " " + padding + "\n");
        threshold.save(writer);
        weight.save(writer);
        if (regularization != null) {
            regularization.write(writer);
        } else {
            writer.write("null\n");
        }
        writer.write(trainable + "\n");
        writer.flush();
    }

    public static ConvolutionLayer read(Scanner scanner) {
        int[] param = Arrays.stream(scanner.nextLine().split(" ")).mapToInt(Integer::parseInt).toArray();

        ConvolutionLayer layer = new ConvolutionLayer(param[0], param[1], param[2], param[3]);
        layer.loadWeight = false;
        layer.threshold = NNVector.read(scanner);
        layer.weight = NNTensor.read(scanner);
        layer.setRegularization(Regularization.read(scanner));
        layer.setTrainable(Boolean.parseBoolean(scanner.nextLine()));
        layer.loadWeight = true;
        return layer;
    }

    public ConvolutionLayer setRegularization(Regularization regularization) {
        this.regularization = regularization;
        return this;
    }

    public ConvolutionLayer setTrainable(boolean trainable) {
        this.trainable = trainable;
        return this;
    }

    public ConvolutionLayer setInitializer(Initializer initializer) {
        this.initializer = initializer;
        return this;
    }

    public void setLoadWeight(boolean loadWeight) {
        this.loadWeight = loadWeight;
    }
}

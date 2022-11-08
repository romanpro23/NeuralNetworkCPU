package neural_network.layers.convolution_3d;

import lombok.Getter;
import lombok.Setter;
import neural_network.initialization.Initializer;
import neural_network.optimizers.Optimizer;
import neural_network.regularization.Regularization;
import nnarrays.*;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Scanner;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class ConvolutionTransposeLayer extends ConvolutionNeuralLayer {
    //trainable parts
    private Regularization regularization;
    private Initializer initializer;

    private boolean loadWeight;

    //weight
    @Setter
    private NNTensor4D weight;
    @Getter
    private NNTensor4D derWeight;
    @Setter
    private NNVector threshold;
    private NNVector derThreshold;

    private final int paddingY;
    private final int paddingX;
    private final int stride;
    private final int heightKernel;
    private final int widthKernel;
    private final int countKernel;

    public ConvolutionTransposeLayer(int countKernel, int sizeKernel) {
        this(countKernel, sizeKernel, sizeKernel, 1, 0, 0);
    }

    public ConvolutionTransposeLayer(int countKernel, int sizeKernel, int stride) {
        this(countKernel, sizeKernel, sizeKernel, stride, 0, 0);
    }

    public ConvolutionTransposeLayer(int countKernel, int sizeKernel, int stride, int padding) {
        this(countKernel, sizeKernel, sizeKernel, stride, padding, padding);
    }

    public ConvolutionTransposeLayer(int countKernel, int heightKernel, int widthKernel, int stride, int paddingY, int paddingX) {
        this.countKernel = countKernel;
        this.paddingX = paddingX;
        this.paddingY = paddingY;
        this.stride = stride;
        this.heightKernel = heightKernel;
        this.widthKernel = widthKernel;
        trainable = true;

        initializer = new Initializer.XavierUniform();
    }

    @Override
    public void initialize(int[] size) {
        if (size.length != 3) {
            throw new ExceptionInInitializerError("Error size pre layer!");
        }
        this.depth = size[2];
        this.height = size[0];
        this.width = size[1];

        int h = height, w = width;
        if (stride > 1) {
            h = (height - 1) * stride + heightKernel % stride;
            w = (width - 1) * stride + widthKernel % stride;
        }

        outWidth = widthKernel - 2 - 2 * paddingX + w + stride;
        outHeight = heightKernel - 2 - 2 * paddingY + h + stride;
        outDepth = countKernel;

        derThreshold = new NNVector(countKernel);
        derWeight = new NNTensor4D(depth, heightKernel, widthKernel, countKernel);
        if (!loadWeight) {
            weight = new NNTensor4D(depth, heightKernel, widthKernel, countKernel);
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
    public void generateOutput(NNArray[] inputs) {
        NNTensor[] inputData = NNArrays.isTensor(inputs);
        input = new NNTensor[inputs.length];
        output = new NNTensor[inputs.length];

        int countC = getCountCores();
        ExecutorService executor = Executors.newFixedThreadPool(countC);
        for (int t = 0; t < countC; t++) {
            final int firstIndex = t * inputData.length / countC;
            final int lastIndex = Math.min(inputData.length, (t + 1) * inputData.length / countC);
            executor.execute(() -> {
                for (int i = firstIndex; i < lastIndex; i++) {
                    input[i] = inputData[i].stride(stride);
                    output[i] = new NNTensor(outHeight, outWidth, outDepth);
                    output[i].transposeConvolution(input[i], weight, paddingY, paddingX);
                    output[i].add(threshold);
                }
            });
        }
        executor.shutdown();
        while (!executor.isTerminated()) {
        }
    }

    @Override
    public void generateError(NNArray[] errors) {
        errorNL = getErrorNextLayer(errors);
        error = new NNTensor[errors.length];

        int countC = getCountCores();
        ExecutorService executor = Executors.newFixedThreadPool(countC);
        for (int t = 0; t < countC; t++) {
            final int firstIndex = t * errors.length / countC;
            final int lastIndex = Math.min(errors.length, (t + 1) * errors.length / countC);
            executor.execute(() -> {
                for (int i = firstIndex; i < lastIndex; i++) {
                    error[i] = new NNTensor(height, width, depth);
                    error[i].convolution(errorNL[i], weight, stride, paddingY, paddingX);

                    if (trainable) {
                        derWeight.convolutionTranspose(input[i], errorNL[i], paddingY, paddingX);
                        derThreshold.add(errorNL[i]);
                    }
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
        System.out.println("Transp conv | " + height + ",\t" + width + ",\t" + depth + "\t|  "
                + outHeight + ",\t" + outWidth + ",\t" + outDepth + "\t|\t" + countParam);
        return countParam;
    }

    @Override
    public void write(FileWriter writer) throws IOException {
        writer.write("Convolution transpose layer 3D\n");
        writer.write(countKernel + " " + heightKernel + " " + widthKernel + " " + stride + " " + paddingY + " " + paddingX + "\n");
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

    public static ConvolutionTransposeLayer read(Scanner scanner) {
        int[] param = Arrays.stream(scanner.nextLine().split(" ")).mapToInt(Integer::parseInt).toArray();

        ConvolutionTransposeLayer layer = new ConvolutionTransposeLayer(param[0], param[1], param[2], param[3], param[4], param[5]);
        layer.loadWeight = false;
        layer.threshold = NNVector.read(scanner);
        layer.weight = NNTensor4D.read(scanner);
        layer.setRegularization(Regularization.read(scanner));
        layer.setTrainable(Boolean.parseBoolean(scanner.nextLine()));
        layer.loadWeight = true;
        return layer;
    }

    public ConvolutionTransposeLayer setRegularization(Regularization regularization) {
        this.regularization = regularization;
        return this;
    }

    public ConvolutionTransposeLayer setTrainable(boolean trainable) {
        this.trainable = trainable;
        return this;
    }

    public ConvolutionTransposeLayer setInitializer(Initializer initializer) {
        this.initializer = initializer;
        return this;
    }

    public void setLoadWeight(boolean loadWeight) {
        this.loadWeight = loadWeight;
    }
}

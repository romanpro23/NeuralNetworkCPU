package neural_network.layers.layer_3d;

import lombok.Getter;
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

public class ConvolutionLayer extends NeuralLayer3D {
    //trainable parts
    private Regularization regularization;
    private Initializer initializer;

    private boolean loadWeight;

    //weightAttention
    @Getter
    private NNTensor4D weight;
    private NNTensor4D derWeight;

    private NNVector threshold;
    private NNVector derThreshold;

    private final int paddingY;
    private final int paddingX;
    private final int step;
    private final int heightKernel;
    private final int widthKernel;
    private final int countKernel;

    public ConvolutionLayer(int countKernel, int sizeKernel) {
        this(countKernel, sizeKernel, sizeKernel, 1, 0, 0);
    }

    public ConvolutionLayer(int countKernel, int sizeKernel, int step) {
        this(countKernel, sizeKernel, sizeKernel, step, 0, 0);
    }

    public ConvolutionLayer(int countKernel, int sizeKernel, int step, int padding) {
        this(countKernel, sizeKernel, sizeKernel, step, padding, padding);
    }

    public ConvolutionLayer(int countKernel, int heightKernel, int widthKernel, int step, int paddingY, int paddingX) {
        this.countKernel = countKernel;
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

        outWidth = (width - widthKernel + 2 * paddingX) / step + 1;
        outHeight = (height - heightKernel + 2 * paddingY) / step + 1;
        outDepth = countKernel;

        derThreshold = new NNVector(countKernel);
        derWeight = new NNTensor4D(countKernel, heightKernel, widthKernel, depth);
        if (!loadWeight) {
            weight = new NNTensor4D(countKernel, heightKernel, widthKernel, depth);
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
        this.input = NNArrays.isTensor(inputs);
        output = new NNTensor[inputs.length];

        ExecutorService executor = Executors.newFixedThreadPool(inputs.length);
        for (int t = 0; t < inputs.length; t++) {
            final int i = t;
            executor.execute(() -> {
                output[i] = new NNTensor(outHeight, outWidth, outDepth);
                output[i].convolution(input[i], weight, step, paddingY, paddingX);
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
        error = new NNTensor[errors.length];

        ExecutorService executor = Executors.newFixedThreadPool(input.length);
        for (int t = 0; t < input.length; t++) {
            final int i = t;
            executor.execute(() -> {
                error[i] = new NNTensor(height, width, depth);
                error[i].transposeConvolution(errorNL[i].stride(step), weight, step, paddingY, paddingX);

                if (trainable) {
                    derWeight.convolution(input[i], errorNL[i], step, paddingY, paddingX);
                        derThreshold.add(errorNL[i]);
                }
//                NNMatrix inputImg = output[i].img2col(input[i], weight, step, paddingY, paddingX);
//                NNMatrix weightM = new NNMatrix(weight.depth(), weight.size() / weight.depth(), weight.getData());
//                NNMatrix errorM = new NNMatrix(outHeight*outWidth ,outDepth, errorNL[i].getData());
//                NNMatrix errorInput = errorM.dot(weightM);
//                error[i] = output[i].deImg2col(input[i], errorInput, weight, step, paddingY, paddingX);
//
//                if (trainable) {
//                    derWeight.add(inputImg.transpose().dot(errorM).transpose());
//                    derThreshold.add(errorNL[i]);
//                }
            });
        }
        executor.shutdown();
        while (!executor.isTerminated()) {
        }

        if (regularization != null && trainable) {
            regularization.regularization(weight);
            regularization.regularization(threshold);
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

    @Override
    public int info() {
        int countParam = weight.size() + threshold.size();
        System.out.println("Convolution\t| " + height + ",\t" + width + ",\t" + depth + "\t| "
                + outHeight + ",\t" + outWidth + ",\t" + outDepth + "\t|\t" + countParam);
        return countParam;
    }

    @Override
    public void save(FileWriter writer) throws IOException {
        writer.write("Convolution layer 3D\n");
        writer.write(countKernel + " " + heightKernel + " " + widthKernel + " " + step + " " + paddingY + " " + paddingX + "\n");
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

        ConvolutionLayer layer = new ConvolutionLayer(param[0], param[1], param[2], param[3], param[4], param[5]);
        layer.loadWeight = false;
        layer.threshold = NNVector.read(scanner);
        layer.weight = NNTensor4D.read(scanner);
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

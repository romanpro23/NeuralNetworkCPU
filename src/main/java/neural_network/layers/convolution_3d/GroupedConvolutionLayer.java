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

public class GroupedConvolutionLayer extends ConvolutionNeuralLayer {
    //trainable parts
    private Regularization regularization;
    private Initializer initializer;

    private boolean loadWeight;

    //weight
    @Setter
    @Getter
    private NNTensor4D weight;
    @Getter
    private NNTensor4D derWeight;
    @Setter
    private NNVector threshold;
    private NNVector derThreshold;

    private final int paddingY;
    private final int paddingX;
    private final int step;
    private final int heightKernel;
    private final int widthKernel;
    private final int countKernel;
    private final int countGroups;

    public GroupedConvolutionLayer(int countKernel, int sizeKernel, int countGroups) {
        this(countKernel, sizeKernel, sizeKernel, 1, 0, 0, countGroups);
    }

    public GroupedConvolutionLayer(int countKernel, int sizeKernel, int step, int countGroups) {
        this(countKernel, sizeKernel, sizeKernel, step, 0, 0, countGroups);
    }

    public GroupedConvolutionLayer(int countKernel, int sizeKernel, int step, int padding, int countGroups) {
        this(countKernel, sizeKernel, sizeKernel, step, padding, padding, countGroups);
    }

    public GroupedConvolutionLayer(int countKernel, int heightKernel, int widthKernel, int step, int paddingY, int paddingX, int countGroups) {
        this.countKernel = countKernel;
        this.paddingX = paddingX;
        this.paddingY = paddingY;
        this.step = step;
        this.heightKernel = heightKernel;
        this.widthKernel = widthKernel;
        this.countGroups = countGroups;
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

        outWidth = (width - widthKernel + 2 * paddingX) / step + 1;
        outHeight = (height - heightKernel + 2 * paddingY) / step + 1;
        outDepth = countKernel;

        derThreshold = new NNVector(countKernel);
        derWeight = new NNTensor4D(countKernel, heightKernel, widthKernel, depth / countGroups);
        if (!loadWeight) {
            weight = new NNTensor4D(countKernel, heightKernel, widthKernel, depth / countGroups);
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

        int countC = getCountCores();
        ExecutorService executor = Executors.newFixedThreadPool(countC);
        for (int t = 0; t < countC; t++) {
            final int firstIndex = t * input.length / countC;
            final int lastIndex = Math.min(input.length, (t + 1) * input.length / countC);
            executor.execute(() -> {
                for (int i = firstIndex; i < lastIndex; i++) {
                    output[i] = new NNTensor(outHeight, outWidth, outDepth);
                    output[i].groupConvolution(input[i], weight, step, paddingY, paddingX, countGroups);
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
                    error[i].transposeGroupConvolution(errorNL[i].stride(step), weight, paddingY, paddingX, countGroups);

                    if(trainable){
                        derWeight.groupConvolution(input[i], errorNL[i], step, paddingY, paddingX, countGroups);
                        derThreshold.add(errorNL[i]);
                    }
                }
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
    public int info() {
        int countParam = weight.size() + threshold.size();
        System.out.println("Group conv\t| " + height + ",\t" + width + ",\t" + depth + "\t| "
                + outHeight + ",\t" + outWidth + ",\t" + outDepth + "\t|\t" + countParam);
        return countParam;
    }

    @Override
    public void write(FileWriter writer) throws IOException {
        writer.write("Grouped convolution layer 3D\n");
        writer.write(countKernel + " " + heightKernel + " " + widthKernel + " " + step + " "
                + paddingY + " " + paddingX + " " + countGroups + "\n");
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

    public static GroupedConvolutionLayer read(Scanner scanner) {
        int[] param = Arrays.stream(scanner.nextLine().split(" ")).mapToInt(Integer::parseInt).toArray();

        GroupedConvolutionLayer layer = new GroupedConvolutionLayer(param[0], param[1], param[2], param[3], param[4], param[5], param[6]);
        layer.loadWeight = false;
        layer.threshold = NNVector.read(scanner);
        layer.weight = NNTensor4D.read(scanner);
        layer.setRegularization(Regularization.read(scanner));
        layer.setTrainable(Boolean.parseBoolean(scanner.nextLine()));
        layer.loadWeight = true;
        return layer;
    }

    public GroupedConvolutionLayer setRegularization(Regularization regularization) {
        this.regularization = regularization;
        return this;
    }

    public GroupedConvolutionLayer setTrainable(boolean trainable) {
        this.trainable = trainable;
        return this;
    }

    public GroupedConvolutionLayer setInitializer(Initializer initializer) {
        this.initializer = initializer;
        return this;
    }

    public void setLoadWeight(boolean loadWeight) {
        this.loadWeight = loadWeight;
    }
}

package neural_network.layers.layer_3d.attention;

import lombok.Getter;
import neural_network.initialization.Initializer;
import neural_network.layers.layer_3d.NeuralLayer3D;
import neural_network.optimizers.Optimizer;
import neural_network.regularization.Regularization;
import nnarrays.*;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class SpatialAttentionModule extends NeuralLayer3D {
    //trainable parts
    private Regularization regularization;
    private Initializer initializer;

    private boolean loadWeight;

    //weightAttention
    @Getter
    private NNTensor4D weight;
    private NNTensor4D derWeight;

    private final int sizeKernel;

    private NNTensor[] max, average, maxAvg;
    private NNTensor[] attInput, attOutput;

    public SpatialAttentionModule() {
        this(7);
    }

    public SpatialAttentionModule(int sizeKernel) {
        this.sizeKernel = sizeKernel;
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

        outWidth = width;
        outHeight = height;
        outDepth = depth;

        derWeight = new NNTensor4D(1, sizeKernel, sizeKernel, 2);
        if (!loadWeight) {
            weight = new NNTensor4D(1, sizeKernel, sizeKernel, 2);
            initializer.initialize(weight);
        }
    }

    @Override
    public void initialize(Optimizer optimizer) {
        optimizer.addDataOptimize(weight, derWeight);
    }

    @Override
    public void generateOutput(NNArray[] inputs) {
        this.input = NNArrays.isTensor(inputs);
        output = new NNTensor[inputs.length];
        max = new NNTensor[inputs.length];
        maxAvg = new NNTensor[inputs.length];
        average = new NNTensor[inputs.length];
        attOutput = new NNTensor[inputs.length];
        attInput = new NNTensor[inputs.length];

        ExecutorService executor = Executors.newFixedThreadPool(inputs.length);
        for (int t = 0; t < inputs.length; t++) {
            final int i = t;
            executor.execute(() -> {
                max[i] = input[i].spatialMaxPool();
                average[i] = input[i].spatialAveragePool();
                maxAvg[i] = max[i].concat(average[i]);

                attInput[i] = new NNTensor(outHeight, outWidth, 1);
                attOutput[i] = new NNTensor(outHeight, outWidth, 1);
                attInput[i].convolution(maxAvg[i], weight, 1, sizeKernel/2, sizeKernel/2);
                attOutput[i].sigmoid(attInput[i]);

                output[i] = input[i].spatialMul(attOutput[i]);
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
        int padding = sizeKernel / 2;

        ExecutorService executor = Executors.newFixedThreadPool(input.length);
        for (int t = 0; t < input.length; t++) {
            final int i = t;
            executor.execute(() -> {
                error[i] = errorNL[i].spatialMul(attOutput[i]);
                NNTensor deltaAtt = errorNL[i].backSpatialMul();
                NNTensor errorAtt = new NNTensor(outHeight, outWidth, 1);
                errorAtt.derSigmoid(attOutput[i], deltaAtt);

                NNTensor errorMaxAvg = new NNTensor(outHeight, outWidth, 2);
                errorMaxAvg.transposeConvolution(errorAtt, weight, 1, padding, padding);

                NNTensor deltaMax = errorMaxAvg.subFlatTensor(0);
                NNTensor deltaAvg = errorMaxAvg.subFlatTensor(1);

                error[i].add(input[i].backSpatialMaxPool(max[i], deltaMax));
                error[i].add(input[i].backSpatialAveragePool(deltaAvg));

                if (trainable) {
                    derWeight.convolution(maxAvg[i], errorAtt, 1, padding, padding);
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
    public int info() {
        int countParam = weight.size();
        System.out.println("Spatial att\t| " + height + ",\t" + width + ",\t" + depth + "\t| "
                + outHeight + ",\t" + outWidth + ",\t" + outDepth + "\t|\t" + countParam);
        return countParam;
    }

    @Override
    public void save(FileWriter writer) throws IOException {
        writer.write("Spatial attention module\n");
        writer.write(sizeKernel + "\n");
        weight.save(writer);
        if (regularization != null) {
            regularization.write(writer);
        } else {
            writer.write("null\n");
        }
        writer.write(trainable + "\n");
        writer.flush();
    }

    public static SpatialAttentionModule read(Scanner scanner) {
        SpatialAttentionModule layer = new SpatialAttentionModule(Integer.parseInt(scanner.nextLine()));
        layer.loadWeight = false;
        layer.weight = NNTensor4D.read(scanner);
        layer.setRegularization(Regularization.read(scanner));
        layer.setTrainable(Boolean.parseBoolean(scanner.nextLine()));
        layer.loadWeight = true;
        return layer;
    }

    public SpatialAttentionModule setRegularization(Regularization regularization) {
        this.regularization = regularization;
        return this;
    }

    public SpatialAttentionModule setTrainable(boolean trainable) {
        this.trainable = trainable;
        return this;
    }

    public SpatialAttentionModule setInitializer(Initializer initializer) {
        this.initializer = initializer;
        return this;
    }

    public void setLoadWeight(boolean loadWeight) {
        this.loadWeight = loadWeight;
    }
}

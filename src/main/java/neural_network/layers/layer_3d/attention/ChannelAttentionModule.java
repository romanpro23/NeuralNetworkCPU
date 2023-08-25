package neural_network.layers.layer_3d.attention;

import lombok.Getter;
import neural_network.initialization.Initializer;
import neural_network.layers.layer_3d.NeuralLayer3D;
import neural_network.optimizers.Optimizer;
import neural_network.regularization.Regularization;
import nnarrays.*;
import utilities.CublasUtil;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class ChannelAttentionModule extends NeuralLayer3D {
    //trainable parts
    private Regularization regularization;
    private Initializer initializer;

    private boolean loadWeight;

    //weightAttention
    @Getter
    private NNMatrix weightRatio;
    private NNMatrix derWeightRatio;

    private NNMatrix weight;
    private NNMatrix derWeight;

    private final int hidden;

    private NNVector[] average, averageH, averageHO, averageOut;
    private NNVector[] max, maxH, maxHO, maxOut;
    private NNVector[] attentionI, attentionOut;

    public ChannelAttentionModule(int hidden) {
        this.hidden = hidden;
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

        derWeightRatio = new NNMatrix(depth, hidden);
        derWeight = new NNMatrix(hidden, depth);
        if (!loadWeight) {
            weightRatio = new NNMatrix(depth, hidden);
            weight = new NNMatrix(hidden, depth);

            initializer.initialize(weight);
            initializer.initialize(weightRatio);
        }
    }

    @Override
    public void initialize(Optimizer optimizer) {
        optimizer.addDataOptimize(weight, derWeight);
        optimizer.addDataOptimize(weightRatio, derWeightRatio);
    }

    @Override
    public void generateOutput(NNArray[] inputs) {
        this.input = NNArrays.isTensor(inputs);
        output = new NNTensor[inputs.length];

        average = new NNVector[inputs.length];
        averageH = new NNVector[inputs.length];
        averageHO = new NNVector[inputs.length];
        averageOut = new NNVector[inputs.length];
        max = new NNVector[inputs.length];
        maxH = new NNVector[inputs.length];
        maxHO = new NNVector[inputs.length];
        maxOut = new NNVector[inputs.length];
        attentionI = new NNVector[inputs.length];
        attentionOut = new NNVector[inputs.length];

        ExecutorService executor = Executors.newFixedThreadPool(inputs.length);
        for (int t = 0; t < inputs.length; t++) {
            final int i = t;
            executor.execute(() -> {
                average[i] = new NNVector(depth);
                max[i] = new NNVector(depth);
                average[i].globalAveragePool(input[i]);
                max[i].globalAveragePool(input[i]);

                averageH[i] = average[i].dot(weight);
                averageHO[i] = new NNVector(hidden);
                averageHO[i].relu(averageH[i]);
                averageOut[i] = averageHO[i].dot(weightRatio);

                maxH[i] = max[i].dot(weight);
                maxHO[i] = new NNVector(hidden);
                maxHO[i].relu(maxH[i]);
                maxOut[i] = maxHO[i].dot(weightRatio);

                attentionI[i] = new NNVector(depth);
                attentionI[i].add(maxOut[i]);
                attentionI[i].add(averageOut[i]);
                attentionOut[i] = new NNVector(depth);
                attentionOut[i].sigmoid(attentionI[i]);

                output[i] = input[i].mul(attentionOut[i]);
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
                error[i] = errorNL[i].mul(attentionOut[i]);
                ;
                NNVector deltaAtt = errorNL[i].mul(input[i]);
                NNVector errorAtt = new NNVector(depth);
                errorAtt.derSigmoid(attentionOut[i], deltaAtt);
                NNVector deltaH = errorAtt.dotT(weightRatio);

                NNVector errorHMax = new NNVector(hidden);
                NNVector errorHAvg = new NNVector(hidden);
                errorHMax.derRelu(maxH[i], deltaH);
                errorHAvg.derRelu(maxH[i], deltaH);

                NNVector deltaHMax = errorHMax.dotT(weight);
                NNVector deltaHAvg = errorHAvg.dotT(weight);

                error[i].backGlobalMaxPool(input[i], max[i], deltaHMax);
                error[i].backGlobalAveragePool(deltaHAvg);

                if (trainable) {
                    derWeightRatio.add(maxHO[i].dot(errorAtt));
                    derWeight.add(max[i].dot(errorHMax));
                    derWeight.add(average[i].dot(errorHAvg));
                }
            });
        }
        executor.shutdown();
        while (!executor.isTerminated()) {
        }

        if (regularization != null && trainable) {
            regularization.regularization(weight);
            regularization.regularization(weightRatio);
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
        int countParam = weight.size() + weightRatio.size();
        System.out.println("Channel att\t| " + height + ",\t" + width + ",\t" + depth + "\t| "
                + outHeight + ",\t" + outWidth + ",\t" + outDepth + "\t|\t" + countParam);
        return countParam;
    }

    @Override
    public void save(FileWriter writer) throws IOException {
        writer.write("Channel attention module\n");
        writer.write(hidden + "\n");
        weightRatio.save(writer);
        weight.save(writer);
        if (regularization != null) {
            regularization.write(writer);
        } else {
            writer.write("null\n");
        }
        writer.write(trainable + "\n");
        writer.flush();
    }

    public static ChannelAttentionModule read(Scanner scanner) {
        ChannelAttentionModule layer = new ChannelAttentionModule(Integer.parseInt(scanner.nextLine()));
        layer.loadWeight = false;
        layer.weightRatio = NNMatrix.read(scanner);
        layer.weight = NNMatrix.read(scanner);
        layer.setRegularization(Regularization.read(scanner));
        layer.setTrainable(Boolean.parseBoolean(scanner.nextLine()));
        layer.loadWeight = true;
        return layer;
    }

    public ChannelAttentionModule setRegularization(Regularization regularization) {
        this.regularization = regularization;
        return this;
    }

    public ChannelAttentionModule setTrainable(boolean trainable) {
        this.trainable = trainable;
        return this;
    }

    public ChannelAttentionModule setInitializer(Initializer initializer) {
        this.initializer = initializer;
        return this;
    }

    public void setLoadWeight(boolean loadWeight) {
        this.loadWeight = loadWeight;
    }
}

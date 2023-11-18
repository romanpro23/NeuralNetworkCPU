package neural_network.layers.layer_3d;

import lombok.Getter;
import neural_network.initialization.Initializer;
import neural_network.optimizers.Optimizer;
import neural_network.regularization.Regularization;
import nnarrays.*;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class SelfAttentionLayer extends NeuralLayer3D {
    private ConvolutionLayer queryLayer;
    private ConvolutionLayer keyLayer;
    private ConvolutionLayer valueLayer;

    private NNVector gamma;
    private NNVector derGamma;

    private boolean loadWeight;

    private final int depthAttention;

    @Getter
    private NNMatrix[] attention;
    @Getter
    private NNMatrix[] attOutput;

    private Regularization regularization;

    public SelfAttentionLayer() {
        this(8);
    }

    public SelfAttentionLayer(int depthAttention) {
        trainable = true;
        loadWeight = false;
        this.depthAttention = depthAttention;
    }

    @Override
    public void initialize(int[] size) {
        super.initialize(size);

        derGamma = new NNVector(1);

        if (!loadWeight) {
            valueLayer = new ConvolutionLayer(depth, 1, 1, 0);
            queryLayer = new ConvolutionLayer(depth / depthAttention, 1, 1, 0);
            keyLayer = new ConvolutionLayer(depth / depthAttention, 1, 1, 0);

            gamma = new NNVector(1);
        }

        valueLayer.initialize(size);
        queryLayer.initialize(size);
        keyLayer.initialize(size);
    }

    @Override
    public void initialize(Optimizer optimizer) {
        valueLayer.initialize(optimizer);
        queryLayer.initialize(optimizer);
        keyLayer.initialize(optimizer);

        optimizer.addDataOptimize(gamma, derGamma);
    }

    @Override
    public void generateOutput(NNArray[] inputs) {
        this.input = NNArrays.isTensor(inputs);
        attention = new NNMatrix[inputs.length];
        attOutput = new NNMatrix[inputs.length];
        output = new NNTensor[inputs.length];

        valueLayer.generateOutput(inputs);
        queryLayer.generateOutput(inputs);
        keyLayer.generateOutput(inputs);

        ExecutorService executor = Executors.newFixedThreadPool(input.length);
        for (int t = 0; t < input.length; t++) {
            final int i = t;
            executor.execute(() -> {
                NNMatrix query = new NNMatrix(height * width, depth / depthAttention, queryLayer.getOutput()[i].getData(), queryLayer.getOutput()[i].getSdata());
                NNMatrix key = new NNMatrix(height * width, depth / depthAttention, keyLayer.getOutput()[i].getData(), keyLayer.getOutput()[i].getSdata()).transpose();
                NNMatrix value = new NNMatrix(height * width, depth, valueLayer.getOutput()[i].getData(), valueLayer.getOutput()[i].getSdata()).transpose();

                attention[i] = new NNMatrix(height * width, height * width);
                attention[i].softmax(query.dot(key));

                attOutput[i] = (value).dot(attention[i]).transpose();
                output[i] = new NNTensor(height, width, depth, attOutput[i].getData(), attOutput[i].getSdata());
                output[i].mul(gamma.get(0));
                output[i].add(inputs[i]);
            });
        }
        executor.shutdown();
        while (!executor.isTerminated()) {
        }
    }

    @Override
    public void generateError(NNArray[] errors) {
        errorNL = getErrorNextLayer(errors);

        NNTensor[] errValue = new NNTensor[errors.length];
        NNTensor[] errQuery = new NNTensor[errors.length];
        NNTensor[] errKey = new NNTensor[errors.length];

        ExecutorService executor = Executors.newFixedThreadPool(input.length);
        for (int t = 0; t < input.length; t++) {
            final int i = t;
            executor.execute(() -> {
                NNMatrix query = new NNMatrix(height * width, depth / depthAttention, queryLayer.getOutput()[i].getData(), queryLayer.getOutput()[i].getSdata()).transpose();
                NNMatrix key = new NNMatrix(height * width, depth / depthAttention, keyLayer.getOutput()[i].getData(), keyLayer.getOutput()[i].getSdata());
                NNMatrix value = new NNMatrix(height * width, depth, valueLayer.getOutput()[i].getData(), valueLayer.getOutput()[i].getSdata());

                NNMatrix errorOut = new NNMatrix(height * width, depth);
                errorOut.add(errorNL[i]);
                errorOut.mul(gamma.get(0));
                errorOut = errorOut.transpose();

                NNMatrix errorValue = errorOut.dot(attention[i].transpose());
                errValue[i] = new NNTensor(height, width, depth, errorValue.transpose().getData(), errorValue.transpose().getSdata());

                NNMatrix errorAttention = value.dot(errorOut);
                NNMatrix errorEnergy = new NNMatrix(attention[i]);
                errorEnergy.derSoftmax(attention[i], errorAttention);

                NNMatrix errorKey = query.dot(errorEnergy);
                errKey[i] = new NNTensor(height, width, depth / depthAttention, errorKey.transpose().getData(), errorKey.transpose().getSdata());

                NNMatrix errorQuery = errorEnergy.dot(key);
                errQuery[i] = new NNTensor(height, width, depth / depthAttention, errorQuery.getData(), errorQuery.getSdata());

                if (trainable) {
                    findDerivative(attOutput[i], errorNL[i]);
                }
            });
        }
        executor.shutdown();
        while (!executor.isTerminated()) {
        }

        valueLayer.generateError(errValue);
        keyLayer.generateError(errKey);
        queryLayer.generateError(errQuery);

        this.error = new NNTensor[errors.length];
        for (int i = 0; i < errors.length; i++) {
            this.error[i] = new NNTensor(height, width, depth);
            this.error[i].add(valueLayer.getError()[i]);
            this.error[i].add(queryLayer.getError()[i]);
            this.error[i].add(keyLayer.getError()[i]);
            this.error[i].add(errorNL[i]);
        }

        if(trainable && regularization != null){
            regularization.regularization(gamma);
        }
    }

    private void findDerivative(NNMatrix out, NNTensor error) {
        for (int i = 0; i < out.size(); i++) {
            derGamma.getData()[0] += out.get(i) * error.get(i);
        }
    }

    @Override
    public int info() {
        int countParam = queryLayer.getWeight().size() + keyLayer.getWeight().size() + valueLayer.getWeight().size() + 1;
        System.out.println("S-Attention\t| " + height + ",\t" + width + ",\t" + depth + "\t| "
                + outHeight + ",\t" + outWidth + ",\t" + outDepth + "\t|\t" + countParam);
        return countParam;
    }

    @Override
    public void save(FileWriter writer) throws IOException {
        writer.write("SelfAttention layer 3D\n");
        writer.write(depthAttention + "\n");
        queryLayer.save(writer);
        keyLayer.save(writer);
        valueLayer.save(writer);
        gamma.save(writer);

        writer.write(trainable + "\n");
        writer.flush();
    }

    @Override
    public void trainable(boolean trainable) {
        this.trainable = trainable;
        queryLayer.setTrainable(trainable);
        valueLayer.setTrainable(trainable);
        keyLayer.setTrainable(trainable);
    }

    public static SelfAttentionLayer read(Scanner scanner) {
        SelfAttentionLayer layer = new SelfAttentionLayer(Integer.parseInt(scanner.nextLine()));
        layer.loadWeight = false;
        scanner.nextLine();
        layer.queryLayer = ConvolutionLayer.read(scanner);
        scanner.nextLine();
        layer.keyLayer = ConvolutionLayer.read(scanner);
        scanner.nextLine();
        layer.valueLayer = ConvolutionLayer.read(scanner);
        layer.gamma = NNVector.read(scanner);

        layer.setTrainable(Boolean.parseBoolean(scanner.nextLine()));
        layer.loadWeight = true;
        return layer;
    }

    public SelfAttentionLayer setRegularization(Regularization regularization) {
        valueLayer.setRegularization(regularization);
        queryLayer.setRegularization(regularization);
        keyLayer.setRegularization(regularization);
        this.regularization = regularization;

        return this;
    }

    public SelfAttentionLayer setTrainable(boolean trainable) {
        this.trainable(trainable);
        return this;
    }

    public SelfAttentionLayer setInitializer(Initializer initializer) {
        valueLayer.setInitializer(initializer);
        queryLayer.setInitializer(initializer);
        keyLayer.setInitializer(initializer);
        return this;
    }
}
package neural_network.layers.layer_3d;

import neural_network.initialization.Initializer;
import neural_network.optimizers.Optimizer;
import neural_network.regularization.Regularization;
import nnarrays.*;
import utilities.CublasUtil;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class ParametricReLULayer3D extends NeuralLayer3D {
    private Regularization regularization;
    private Initializer initializer;

    private boolean loadWeight;

    private NNVector alpha;
    private NNVector derAlpha;

    public ParametricReLULayer3D() {
        trainable = true;

        initializer = null;
    }

    @Override
    public void initialize(int[] size) {
        super.initialize(size);

        derAlpha = new NNVector(size[2]);
        if (!loadWeight) {
            alpha = new NNVector(size[2]);
            if (initializer != null) {
                initializer.initialize(alpha);
            }
        }
    }

    @Override
    public void initialize(Optimizer optimizer) {
        optimizer.addDataOptimize(alpha, derAlpha);
    }

    @Override
    public void generateOutput(NNArray[] input) {
        this.input = NNArrays.isTensor(input);
        this.output = new NNTensor[input.length];


        ExecutorService executor = Executors.newFixedThreadPool(input.length);
        for (int t = 0; t < input.length; t++) {
            final int i = t;
            executor.execute(() -> {
                this.output[i] = new NNTensor(height, width, depth);
                output[i].prelu(this.input[i], alpha);
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
    public void generateError(NNArray[] error) {
        errorNL = getErrorNextLayer(error);
        this.error = new NNTensor[errorNL.length];

        ExecutorService executor = Executors.newFixedThreadPool(input.length);
        for (int t = 0; t < input.length; t++) {
            final int i = t;
            executor.execute(() -> {
                this.error[i] = new NNTensor(height, width, depth);
                this.error[i].derPrelu(input[i], errorNL[i], alpha);
                if (trainable) {
                    derivativeWeight(input[i], errorNL[i]);
                }
            });
        }
        executor.shutdown();
        while (!executor.isTerminated()) {
        }

        if (trainable && regularization != null) {
            regularization.regularization(alpha);
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

    private void derivativeWeight(NNTensor input, NNTensor error) {
        for (int i = 0, index = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                for (int k = 0; k < depth; k++, index++) {
                    if (input.getData()[index] < 0) {
                        derAlpha.getData()[k] += input.get(index) * error.get(index);
                    }
                }
            }
        }
    }

    @Override
    public int info() {
        System.out.println("Activation\t| " + height + ",\t" + width + ",\t" + depth + "\t| "
                + height + ",\t" + width + ",\t" + depth + "\t|");
        return 0;
    }

    @Override
    public void save(FileWriter writer) throws IOException {
        writer.write("Parametric ReLU activation layer 3D\n");
        alpha.save(writer);
        if (regularization != null) {
            regularization.write(writer);
        } else {
            writer.write("null\n");
        }
        writer.write(trainable + "\n");
        writer.flush();
    }

    public ParametricReLULayer3D setRegularization(Regularization regularization) {
        this.regularization = regularization;
        return this;
    }

    public ParametricReLULayer3D setTrainable(boolean trainable) {
        this.trainable = trainable;
        return this;
    }

    public static ParametricReLULayer3D read(Scanner scanner) {
        ParametricReLULayer3D layer = new ParametricReLULayer3D();
        layer.loadWeight = false;
        layer.alpha = NNVector.read(scanner);
        layer.setRegularization(Regularization.read(scanner));
        layer.setTrainable(Boolean.parseBoolean(scanner.nextLine()));
        layer.loadWeight = true;
        return layer;
    }
}
package neural_network.layers.layer_3d;

import nnarrays.NNArray;
import nnarrays.NNArrays;
import nnarrays.NNTensor;
import utilities.CublasUtil;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class RandomReLULayer3D extends NeuralLayer3D {
    private final float min;
    private final float max;
    private NNTensor[] alpha;

    public RandomReLULayer3D() {
        this(0.1, 0.3);
    }

    public RandomReLULayer3D(double min, double max) {
        this.min = (float) min;
        this.max = (float) max;
    }

    @Override
    public void generateOutput(NNArray[] input) {
        this.input = NNArrays.isTensor(input);
        this.output = new NNTensor[input.length];
        this.alpha = new NNTensor[input.length];

        ExecutorService executor = Executors.newFixedThreadPool(input.length);
        for (int t = 0; t < input.length; t++) {
            final int i = t;
            executor.execute(() -> {
                this.output[i] = new NNTensor(height, width, depth);
                this.alpha[i] = new NNTensor(height, width, depth);
                output[i].leakyRelu(this.input[i], (max - min) / 2.0f + min);
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
    public void generateTrainOutput(NNArray[] input) {
        this.input = NNArrays.isTensor(input);
        this.output = new NNTensor[input.length];
        this.alpha = new NNTensor[input.length];

        ExecutorService executor = Executors.newFixedThreadPool(input.length);
        for (int t = 0; t < input.length; t++) {
            final int i = t;
            executor.execute(() -> {
                this.output[i] = new NNTensor(height, width, depth);
                this.alpha[i] = new NNTensor(height, width, depth);
                alpha[i].fillRandom(min, max);
                output[i].randomrelu(this.input[i], alpha[i]);
            });
        }
        executor.shutdown();
        while (!executor.isTerminated()) {
        }
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
                this.error[i].derRandomRelu(input[i], errorNL[i], alpha[i]);
            });
        }
        executor.shutdown();
        while (!executor.isTerminated()) {
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
        System.out.println("Activation\t| " + height + ",\t"+ width + ",\t" + depth + "\t| "
                + height + ",\t" + width + ",\t" + depth + "\t|");
        return 0;
    }

    @Override
    public void save(FileWriter writer) throws IOException {
        writer.write("Random ReLU activation layer 3D\n");
        writer.flush();
    }

    public static RandomReLULayer3D read(Scanner scanner) {
        return new RandomReLULayer3D();
    }
}
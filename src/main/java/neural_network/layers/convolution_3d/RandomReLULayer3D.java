package neural_network.layers.convolution_3d;

import neural_network.initialization.Initializer;
import neural_network.optimizers.Optimizer;
import neural_network.regularization.Regularization;
import nnarrays.NNArray;
import nnarrays.NNArrays;
import nnarrays.NNTensor;
import nnarrays.NNVector;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class RandomReLULayer3D extends ConvolutionNeuralLayer {
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

        int countC = getCountCores();
        ExecutorService executor = Executors.newFixedThreadPool(countC);
        for (int t = 0; t < countC; t++) {
            final int firstIndex = t * input.length / countC;
            final int lastIndex = Math.min(input.length, (t + 1) * input.length / countC);
            executor.execute(() -> {
                for (int i = firstIndex; i < lastIndex; i++) {
                    this.output[i] = new NNTensor(height, width, depth);
                    this.alpha[i] = new NNTensor(height, width, depth);
                    output[i].leakyRelu(this.input[i], (max - min) / 2.0f + min);
                }
            });
        }
        executor.shutdown();
        while (!executor.isTerminated()) {
        }
    }

    @Override
    public void generateTrainOutput(NNArray[] input) {
        this.input = NNArrays.isTensor(input);
        this.output = new NNTensor[input.length];
        this.alpha = new NNTensor[input.length];

        int countC = getCountCores();
        ExecutorService executor = Executors.newFixedThreadPool(countC);
        for (int t = 0; t < countC; t++) {
            final int firstIndex = t * input.length / countC;
            final int lastIndex = Math.min(input.length, (t + 1) * input.length / countC);
            executor.execute(() -> {
                for (int i = firstIndex; i < lastIndex; i++) {
                    this.output[i] = new NNTensor(height, width, depth);
                    this.alpha[i] = new NNTensor(height, width, depth);
                    alpha[i].fillRandom(min, max);
                    output[i].randomrelu(this.input[i], alpha[i]);
                }
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

        int countC = getCountCores();
        ExecutorService executor = Executors.newFixedThreadPool(countC);
        for (int t = 0; t < countC; t++) {
            final int firstIndex = t * input.length / countC;
            final int lastIndex = Math.min(input.length, (t + 1) * input.length / countC);
            executor.execute(() -> {
                for (int i = firstIndex; i < lastIndex; i++) {
                    this.error[i] = new NNTensor(height, width, depth);
                    this.error[i].derRandomRelu(input[i], errorNL[i], alpha[i]);
                }
            });
        }
        executor.shutdown();
        while (!executor.isTerminated()) {
        }
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
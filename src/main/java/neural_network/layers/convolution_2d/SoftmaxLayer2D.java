package neural_network.layers.convolution_2d;

import nnarrays.NNArray;
import nnarrays.NNArrays;
import nnarrays.NNMatrix;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class SoftmaxLayer2D extends ConvolutionNeuralLayer {

    @Override
    public void generateOutput(NNArray[] input) {
        this.input = NNArrays.isMatrix(input);
        this.output = new NNMatrix[input.length];

        int countC = getCountCores();
        ExecutorService executor = Executors.newFixedThreadPool(countC);
        for (int t = 0; t < countC; t++) {
            final int firstIndex = t * input.length / countC;
            final int lastIndex = Math.min(input.length, (t + 1) * input.length / countC);
            executor.execute(() -> {
                for (int i = firstIndex; i < lastIndex; i++) {
                    this.output[i] = new NNMatrix(this.input[i]);
                    this.output[i].softmax(this.input[i]);
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
        this.error = new NNMatrix[errorNL.length];

        int countC = getCountCores();
        ExecutorService executor = Executors.newFixedThreadPool(countC);
        for (int t = 0; t < countC; t++) {
            final int firstIndex = t * input.length / countC;
            final int lastIndex = Math.min(input.length, (t + 1) * input.length / countC);
            executor.execute(() -> {
                for (int i = firstIndex; i < lastIndex; i++) {
                    this.error[i] = new NNMatrix(this.input[i]);
                    this.error[i].derSoftmax(output[i], errorNL[i]);
                }
            });
        }
        executor.shutdown();
        while (!executor.isTerminated()) {
        }
    }

    @Override
    public int info() {
        System.out.println("Activation\t|  " + width + ",\t" + depth + "\t\t| " + width + ",\t" + depth + "\t\t|");
        return 0;
    }

    @Override
    public void save(FileWriter writer) throws IOException {
        writer.write("Softmax layer 2D\n");
        writer.flush();
    }

    public static SoftmaxLayer2D read(Scanner scanner) {
        return new SoftmaxLayer2D();
    }
}
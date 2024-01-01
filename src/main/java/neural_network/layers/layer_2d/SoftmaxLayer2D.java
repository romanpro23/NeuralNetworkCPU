package neural_network.layers.layer_2d;

import nnarrays.NNArray;
import nnarrays.NNArrays;
import nnarrays.NNMatrix;
import utilities.Use;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import static utilities.Use.GPU_Sleep;
import static utilities.Use.GPU_WakeUp;

public class SoftmaxLayer2D extends NeuralLayer2D {

    @Override
    public void generateOutput(NNArray[] input) {
        this.input = NNArrays.isMatrix(input);
        this.output = new NNMatrix[input.length];

        if ((Use.CPU) && (!Use.GPU)) {
            GPU_Sleep();
            ExecutorService executor = Executors.newFixedThreadPool(input.length);
            for (int t = 0; t < input.length; t++) {
                final int i = t;
                executor.execute(() -> {
                    this.output[i] = new NNMatrix(this.input[i]);
                    this.output[i].softmax(this.input[i]);
                });
            }
            executor.shutdown();
            while (!executor.isTerminated()) {
            }
            GPU_WakeUp();
        }

        if (Use.GPU) {
            for (int i = 0; i < input.length; i++) {
                this.output[i] = new NNMatrix(this.input[i]);
                this.output[i].softmax(this.input[i]);
            }
        }

    }

    @Override
    public void generateError(NNArray[] error) {
        errorNL = getErrorNextLayer(error);
        this.error = new NNMatrix[errorNL.length];

        if ((Use.CPU) && (!Use.GPU)) {
            GPU_Sleep();
            ExecutorService executor = Executors.newFixedThreadPool(input.length);
            for (int t = 0; t < input.length; t++) {
                final int i = t;
                executor.execute(() -> {
                    this.error[i] = new NNMatrix(this.input[i]);
                    this.error[i].derSoftmax(output[i], errorNL[i]);
                });
            }
            executor.shutdown();
            while (!executor.isTerminated()) {
            }
            GPU_WakeUp();
        }
        if (Use.GPU) {
            for (int i = 0; i < input.length; i++) {
                this.error[i] = new NNMatrix(this.input[i]);
                this.error[i].derSoftmax(output[i], errorNL[i]);
            }
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
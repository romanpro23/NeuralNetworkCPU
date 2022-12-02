package neural_network.layers.convolution_3d;

import lombok.Setter;
import neural_network.activation.FunctionActivation;
import nnarrays.*;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class ActivationLayer3D extends ConvolutionNeuralLayer {
    @Setter
    private FunctionActivation functionActivation;

    public ActivationLayer3D(FunctionActivation functionActivation) {
        this.functionActivation = functionActivation;
    }

    @Override
    public void generateOutput(NNArray[] input) {
        this.input = NNArrays.isTensor(input);
        this.output = new NNTensor[input.length];

        int countC = getCountCores();
        ExecutorService executor = Executors.newFixedThreadPool(countC);
        for (int t = 0; t < countC; t++) {
            final int firstIndex = t * input.length / countC;
            final int lastIndex = Math.min(input.length, (t + 1) * input.length / countC);
            executor.execute(() -> {
                for (int i = firstIndex; i < lastIndex; i++) {
                    this.output[i] = new NNTensor(height, width, depth);
                    functionActivation.activation(input[i], output[i]);
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
                    functionActivation.derivativeActivation(input[i], output[i], errorNL[i], this.error[i]);
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
        writer.write("Activation layer 3D\n");
        functionActivation.save(writer);
        writer.flush();
    }

    public static ActivationLayer3D read(Scanner scanner) {
        return new ActivationLayer3D(FunctionActivation.read(scanner));
    }
}
package neural_network.layers.dense;

import neural_network.activation.FunctionActivation;
import neural_network.optimizers.Optimizer;
import nnarrays.NNArray;
import nnarrays.NNArrays;
import nnarrays.NNMatrix;
import nnarrays.NNVector;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;

public class ActivationLayer extends DenseNeuralLayer {
    private final FunctionActivation functionActivation;

    public ActivationLayer(FunctionActivation functionActivation) {
        this.functionActivation = functionActivation;
    }

    @Override
    public void initialize(int[] size) {
        if (size.length != 1) {
            throw new ExceptionInInitializerError("Error size pre layer!");
        }
        countNeuron = size[0];
    }

    @Override
    public void generateOutput(NNArray[] input) {
        this.input = NNArrays.isVector(input);
        this.output = new NNVector[input.length];

        for (int i = 0; i < output.length; i++) {
            this.output[i] = new NNVector(countNeuron);
            functionActivation.activation(input[i], output[i]);
        }
    }

    @Override
    public void generateTrainOutput(NNArray[] input) {
        generateOutput(input);
    }

    @Override
    public void generateError(NNArray[] error) {
        errorNL = getErrorNextLayer(error);
        this.error = new NNVector[errorNL.length];

        for (int i = 0; i < input.length; i++) {
            this.error[i] = new NNVector(countNeuron);
            functionActivation.derivativeActivation(input[i], output[i], errorNL[i], this.error[i]);
        }
    }

    @Override
    public int info() {
        System.out.println("Activation\t|  " + countNeuron + "\t\t\t|  " + countNeuron + "\t\t\t|");
        return 0;
    }

    @Override
    public void write(FileWriter writer) throws IOException {
        writer.write("Activation layer\n");
        functionActivation.save(writer);
        writer.flush();
    }

    public static ActivationLayer read(Scanner scanner) {
        return new ActivationLayer(FunctionActivation.read(scanner));
    }
}
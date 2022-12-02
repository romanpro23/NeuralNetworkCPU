package neural_network.layers.dense;

import neural_network.optimizers.Optimizer;
import nnarrays.NNArray;
import nnarrays.NNArrays;
import nnarrays.NNMatrix;
import nnarrays.NNVector;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;

public class DropoutLayer extends DenseNeuralLayer {
    private final double dropout;

    public DropoutLayer(double dropout) {
        this.dropout = dropout;
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
        this.output = this.input;
    }

    @Override
    public void generateTrainOutput(NNArray[] input) {
        this.input = NNArrays.isVector(input);
        this.output = new NNVector[input.length];

        for (int i = 0; i < output.length; i++) {
            this.output[i] = new NNVector(countNeuron);
            output[i].dropout(this.input[i], dropout);
        }
    }

    @Override
    public void generateError(NNArray[] error) {
        errorNL = getErrorNextLayer(error);
        this.error = new NNVector[errorNL.length];

        for (int i = 0; i < input.length; i++) {
            this.error[i] = new NNVector(countNeuron);
            this.error[i].dropoutBack(output[i], errorNL[i], dropout);
        }
    }

    @Override
    public int info() {
        System.out.println("Dropout \t|  " + countNeuron + "\t\t\t|  " + countNeuron + "\t\t\t|");
        return 0;
    }

    @Override
    public void save(FileWriter writer) throws IOException {
        writer.write("Dropout layer\n");
        writer.write(dropout + "\n");
        writer.flush();
    }

    public static DropoutLayer read(Scanner scanner) {
        return new DropoutLayer(Double.parseDouble(scanner.nextLine()));
    }
}

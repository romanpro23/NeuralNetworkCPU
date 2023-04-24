package neural_network.layers.capsule;

import neural_network.layers.reshape.FlattenLayer2D;
import nnarrays.NNArray;
import nnarrays.NNArrays;
import nnarrays.NNMatrix;
import nnarrays.NNVector;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;

public class DigitCapsuleLayer extends FlattenLayer2D {

    public void initialize(int[] size) {
        if (size.length != 2) {
            throw new ExceptionInInitializerError("Error size pre layer!");
        }
        depth = size[1];
        width = size[0];
        countNeuron = width;
    }

    @Override
    public void generateOutput(NNArray[] input) {
        this.input = NNArrays.isMatrix(input);
        this.output = new NNVector[input.length];
        for (int t = 0; t < input.length; t++) {
            output[t] = this.input[t].mod();
        }
    }

    @Override
    public void generateError(NNArray[] error) {
        errorNL = getErrorNextLayer(error);
        this.error = new NNMatrix[errorNL.length];

        for (int i = 0; i < input.length; i++) {
            this.error[i] = this.input[i].backMod(output[i], errorNL[i]);
        }
    }

    @Override
    public int info() {
        System.out.println("Digit caps\t|  " + width + ",\t" + depth + "\t\t|  " + countNeuron + "\t\t\t|");
        return 0;
    }

    @Override
    public void save(FileWriter writer) throws IOException {
        writer.write("Digit capsule layer\n");
        writer.flush();
    }

    public static DigitCapsuleLayer read(Scanner scanner) {
        return new DigitCapsuleLayer();
    }
}

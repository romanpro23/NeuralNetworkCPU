package neural_network.layers.convolution_2d;

import nnarrays.NNArray;
import nnarrays.NNArrays;
import nnarrays.NNMatrix;
import nnarrays.NNTensor;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;

public class DropoutLayer2D extends ConvolutionNeuralLayer {
    private final double dropout;

    public DropoutLayer2D(double dropout) {
        this.dropout = dropout;
    }

    @Override
    public void generateOutput(NNArray[] input) {
        this.input = NNArrays.isMatrix(input);
        this.output = this.input;
    }

    @Override
    public void generateTrainOutput(NNArray[] input) {
        this.input = NNArrays.isMatrix(input);
        this.output = new NNMatrix[input.length];

        for (int i = 0; i < output.length; i++) {
            this.output[i] = new NNMatrix(outWidth, outDepth);
            output[i].dropout(this.input[i], dropout);
        }
    }

    @Override
    public void generateError(NNArray[] error) {
        errorNL = getErrorNextLayer(error);
        this.error = new NNMatrix[errorNL.length];

        for (int i = 0; i < input.length; i++) {
            this.error[i] = new NNMatrix(width, depth);
            this.error[i].dropoutBack(output[i], errorNL[i], dropout);
        }
    }

    @Override
    public int info() {
        System.out.println("Dropout \t|  " + width + ",\t" + depth + "\t\t| " + width + ",\t" + depth + "\t\t|");
        return 0;
    }

    @Override
    public void save(FileWriter writer) throws IOException {
        writer.write("Dropout layer 2D\n");
        writer.write(dropout + "\n");
        writer.flush();
    }

    public static DropoutLayer2D read(Scanner scanner) {
        return new DropoutLayer2D(Double.parseDouble(scanner.nextLine()));
    }
}

package neural_network.layers.convolution_3d;

import neural_network.layers.dense.DenseNeuralLayer;
import nnarrays.NNArray;
import nnarrays.NNArrays;
import nnarrays.NNTensor;
import nnarrays.NNVector;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;

public class DropoutLayer3D extends ConvolutionNeuralLayer {
    private final double dropout;

    public DropoutLayer3D(double dropout) {
        this.dropout = dropout;
    }

    @Override
    public void generateOutput(NNArray[] input) {
        this.input = NNArrays.isTensor(input);
        this.output = this.input;
    }

    @Override
    public void generateTrainOutput(NNArray[] input) {
        this.input = NNArrays.isTensor(input);
        this.output = new NNTensor[input.length];

        for (int i = 0; i < output.length; i++) {
            this.output[i] = new NNTensor(outHeight, outWidth, outDepth);
            output[i].dropout(this.input[i], dropout);
        }
    }

    @Override
    public void generateError(NNArray[] error) {
        errorNL = getErrorNextLayer(error);
        this.error = new NNTensor[errorNL.length];

        for (int i = 0; i < input.length; i++) {
            this.error[i] = new NNTensor(height, width, depth);
            this.error[i].dropoutBack(output[i], errorNL[i], dropout);
        }
    }

    @Override
    public int info() {
        System.out.println("Dropout \t|  " + height + ",\t"+ width + ",\t" + depth + "\t| "
                + height + ",\t" + width + ",\t" + depth + "\t|");
        return 0;
    }

    @Override
    public void write(FileWriter writer) throws IOException {
        writer.write("Dropout layer 3D\n");
        writer.write(dropout + "\n");
        writer.flush();
    }

    public static DropoutLayer3D read(Scanner scanner) {
        return new DropoutLayer3D(Double.parseDouble(scanner.nextLine()));
    }
}

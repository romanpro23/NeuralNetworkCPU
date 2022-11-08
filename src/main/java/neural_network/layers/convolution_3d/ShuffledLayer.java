package neural_network.layers.convolution_3d;

import nnarrays.NNArray;
import nnarrays.NNArrays;
import nnarrays.NNTensor;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;

public class ShuffledLayer extends ConvolutionNeuralLayer {
    private final int countGroup;

    public ShuffledLayer(int countGroup) {
        this.countGroup = countGroup;
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
            output[i].shuffle(this.input[i], countGroup);
        }
    }

    @Override
    public void generateError(NNArray[] error) {
        errorNL = getErrorNextLayer(error);
        this.error = new NNTensor[errorNL.length];

        for (int i = 0; i < input.length; i++) {
            this.error[i] = new NNTensor(height, width, depth);
            this.error[i].backShuffle(errorNL[i], countGroup);
        }
    }

    @Override
    public int info() {
        System.out.println("Shuffled\t|  " + height + ",\t"+ width + ",\t" + depth + "\t| "
                + height + ",\t" + width + ",\t" + depth + "\t|");
        return 0;
    }

    @Override
    public void write(FileWriter writer) throws IOException {
        writer.write("Shuffled layer\n");
        writer.write(countGroup + "\n");
        writer.flush();
    }

    public static ShuffledLayer read(Scanner scanner) {
        return new ShuffledLayer(Integer.parseInt(scanner.nextLine()));
    }
}

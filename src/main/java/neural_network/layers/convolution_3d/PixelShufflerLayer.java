package neural_network.layers.convolution_3d;

import nnarrays.NNArray;
import nnarrays.NNArrays;
import nnarrays.NNTensor;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Scanner;

public class PixelShufflerLayer extends ConvolutionNeuralLayer {
    private final int kernelSize;

    public PixelShufflerLayer(int kernelSize) {
        this.kernelSize = kernelSize;
    }

    @Override
    public void initialize(int[] size) {
        height = size[0];
        width = size[1];
        depth = size[2];

        outWidth = width * kernelSize;
        outHeight = height * kernelSize;
        outDepth = depth / (kernelSize * kernelSize);
    }

    @Override
    public int info() {
        System.out.println("PixelShuff\t| " + height + ",\t" + width + ",\t" + depth + "\t| "
                + outHeight + ",\t" + outWidth + ",\t" + outDepth + "\t|");
        return 0;
    }

    @Override
    public void save(FileWriter writer) throws IOException {
        writer.write("Pixel shuffler layer 3D\n");
        writer.write(kernelSize + "\n");
        writer.flush();
    }

    @Override
    public void generateOutput(NNArray[] inputs) {
        input = NNArrays.isTensor(inputs);
        output = new NNTensor[inputs.length];

        for (int i = 0; i < output.length; i++) {
            output[i] = new NNTensor(outHeight, outWidth, outDepth);
            output[i].pixelShuffle(input[i], kernelSize);
        }
    }

    @Override
    public void generateError(NNArray[] errors) {
        errorNL = getErrorNextLayer(errors);
        error = new NNTensor[errors.length];

        for (int i = 0; i < errors.length; i++) {
            error[i] = new NNTensor(height, width, depth);
            error[i].backPixelShuffle(errorNL[i], kernelSize);
        }
    }

    public static PixelShufflerLayer read(Scanner scanner) {
        int[] param = Arrays.stream(scanner.nextLine().split(" ")).mapToInt(Integer::parseInt).toArray();
        return new PixelShufflerLayer(param[0]);
    }
}

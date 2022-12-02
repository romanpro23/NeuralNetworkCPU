package neural_network.layers.convolution_3d;

import nnarrays.NNArray;
import nnarrays.NNArrays;
import nnarrays.NNTensor;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Scanner;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class UpSamplingLayer extends ConvolutionNeuralLayer {
    private final int heightKernel;
    private final int widthKernel;

    public UpSamplingLayer() {
        this(2);
    }

    public UpSamplingLayer(int sizeKernel) {
        this(sizeKernel, sizeKernel);
    }

    public UpSamplingLayer(int heightKernel, int widthKernel) {
        this.heightKernel = heightKernel;
        this.widthKernel = widthKernel;
    }

    @Override
    public void initialize(int[] size) {
        if (size.length != 3) {
            throw new ExceptionInInitializerError("Error size pre layer!");
        }
        this.depth = size[2];
        this.height = size[0];
        this.width = size[1];

        outDepth = depth;
        outWidth = width * widthKernel;
        outHeight = height * heightKernel;
    }

    @Override
    public void generateOutput(NNArray[] input) {
        this.input = NNArrays.isTensor(input);
        this.output = new NNTensor[input.length];

        for (int i = 0; i < input.length; i++) {
            output[i] = new NNTensor(outHeight, outWidth, outDepth);
            output[i].upSampling(this.input[i], heightKernel, widthKernel);
        }
    }

    @Override
    public void generateError(NNArray[] errors) {
        errorNL = getErrorNextLayer(errors);
        this.error = new NNTensor[errors.length];

        for (int i = 0; i < input.length; i++) {
            this.error[i] = new NNTensor(height, width, depth);
            this.error[i].backUpSampling(errorNL[i], heightKernel, widthKernel);
        }
    }

    @Override
    public int info() {
        System.out.println("Up sampling\t|  " + height + ",\t" + width + ",\t" + depth + "\t|  "
                + outHeight + ",\t" + outWidth + ",\t" + outDepth + "\t|");
        return 0;
    }

    @Override
    public void save(FileWriter writer) throws IOException {
        writer.write("Up sampling layer\n");
        writer.write(heightKernel + " " + widthKernel + "\n");
        writer.flush();
    }

    public static UpSamplingLayer read(Scanner scanner) {
        int[] param = Arrays.stream(scanner.nextLine().split(" ")).mapToInt(Integer::parseInt).toArray();
        UpSamplingLayer layer = new UpSamplingLayer(param[0], param[1]);
        return layer;
    }
}

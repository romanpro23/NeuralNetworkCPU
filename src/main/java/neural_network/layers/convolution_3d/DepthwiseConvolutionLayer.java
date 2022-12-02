package neural_network.layers.convolution_3d;

import neural_network.initialization.Initializer;
import neural_network.optimizers.Optimizer;
import neural_network.regularization.Regularization;
import nnarrays.*;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Scanner;

public class DepthwiseConvolutionLayer extends ConvolutionNeuralLayer {
    private ConvolutionLayer convolutionLayer;
    private GroupedConvolutionLayer groupedConvolutionLayer;

    private final int paddingY;
    private final int paddingX;
    private final int step;
    private final int heightKernel;
    private final int widthKernel;
    private final int countKernel;

    private boolean loadWeight;

    public DepthwiseConvolutionLayer(int countKernel, int sizeKernel) {
        this(countKernel, sizeKernel, sizeKernel, 1, 0, 0);
    }

    public DepthwiseConvolutionLayer(int countKernel, int sizeKernel, int step) {
        this(countKernel, sizeKernel, sizeKernel, step, 0, 0);
    }

    public DepthwiseConvolutionLayer(int countKernel, int sizeKernel, int step, int padding) {
        this(countKernel, sizeKernel, sizeKernel, step, padding, padding);
    }

    public DepthwiseConvolutionLayer(int countKernel, int heightKernel, int widthKernel, int step, int paddingY, int paddingX) {
        this.countKernel = countKernel;
        this.paddingX = paddingX;
        this.paddingY = paddingY;
        this.step = step;
        this.heightKernel = heightKernel;
        this.widthKernel = widthKernel;
        trainable = true;
        loadWeight = false;

        convolutionLayer = new ConvolutionLayer(countKernel, 1, 1, 0);
    }

    @Override
    public void initialize(int[] size) {
        if (size.length != 3) {
            throw new ExceptionInInitializerError("Error size pre layer!");
        }
        this.depth = size[2];
        this.height = size[0];
        this.width = size[1];

        outWidth = (width - widthKernel + 2 * paddingX) / step + 1;
        outHeight = (height - heightKernel + 2 * paddingY) / step + 1;
        outDepth = countKernel;

        if (!loadWeight) {
            groupedConvolutionLayer = new GroupedConvolutionLayer(depth, heightKernel, widthKernel, step, paddingY, paddingX, depth);
        }
        groupedConvolutionLayer.initialize(size);
        convolutionLayer.initialize(groupedConvolutionLayer.size());
    }

    @Override
    public void initialize(Optimizer optimizer) {
        groupedConvolutionLayer.initialize(optimizer);
        convolutionLayer.initialize(optimizer);
    }

    @Override
    public void generateOutput(NNArray[] inputs) {
        groupedConvolutionLayer.generateOutput(inputs);
        convolutionLayer.generateOutput(groupedConvolutionLayer.getOutput());

        this.output = convolutionLayer.output;
    }

    @Override
    public void generateError(NNArray[] errors) {
        convolutionLayer.generateError(errors);
        groupedConvolutionLayer.generateError(convolutionLayer.error);

        this.error = groupedConvolutionLayer.error;
    }

    @Override
    public int info() {
        int countParam = convolutionLayer.getWeight().size() + depth + groupedConvolutionLayer.getWeight().size() + countKernel;
        System.out.println("DepthW conv\t| " + height + ",\t" + width + ",\t" + depth + "\t| "
                + outHeight + ",\t" + outWidth + ",\t" + outDepth + "\t|\t" + countParam);
        return countParam;
    }

    @Override
    public void save(FileWriter writer) throws IOException {
        writer.write("Depthwise convolution layer 3D\n");
        writer.write(countKernel + " " + heightKernel + " " + widthKernel + " " + step + " "
                + paddingY + " " + paddingX + "\n");
        groupedConvolutionLayer.save(writer);
        convolutionLayer.save(writer);
        writer.write(trainable + "\n");
        writer.flush();
    }

    @Override
    public void trainable(boolean trainable){
        this.trainable = trainable;
        convolutionLayer.setTrainable(trainable);
        groupedConvolutionLayer.setTrainable(trainable);
    }

    public static DepthwiseConvolutionLayer read(Scanner scanner) {
        int[] param = Arrays.stream(scanner.nextLine().split(" ")).mapToInt(Integer::parseInt).toArray();

        DepthwiseConvolutionLayer layer = new DepthwiseConvolutionLayer(param[0], param[1], param[2], param[3], param[4], param[5]);
        layer.loadWeight = false;
        scanner.nextLine();
        layer.groupedConvolutionLayer = GroupedConvolutionLayer.read(scanner);
        layer.convolutionLayer = ConvolutionLayer.read(scanner);
        layer.setTrainable(Boolean.parseBoolean(scanner.nextLine()));
        layer.loadWeight = true;
        return layer;
    }

    public DepthwiseConvolutionLayer setRegularization(Regularization regularization) {
        groupedConvolutionLayer.setRegularization(regularization);
        convolutionLayer.setRegularization(regularization);
        return this;
    }

    public DepthwiseConvolutionLayer setTrainable(boolean trainable) {
        this.trainable(trainable);
        return this;
    }

    public DepthwiseConvolutionLayer setInitializer(Initializer initializer) {
        groupedConvolutionLayer.setInitializer(initializer);
        convolutionLayer.setInitializer(initializer);
        return this;
    }

    public void setLoadWeight(boolean loadWeight) {
        this.loadWeight = loadWeight;
    }
}

package neural_network.layers.layer_3d;

import lombok.Getter;
import neural_network.initialization.Initializer;
import neural_network.optimizers.Optimizer;
import neural_network.regularization.Regularization;
import nnarrays.*;
import utilities.CublasUtil;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Scanner;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class DeformableV2ConvolutionLayer extends NeuralLayer3D {
    //trainable parts
    private Regularization regularization;
    private Initializer initializer;

    private boolean loadWeight;

    //weight
    @Getter
    private NNTensor4D weight;
    private NNTensor4D derWeight;

    private NNTensor4D weightOffset;
    private NNTensor4D derWeightOffset;

    private NNTensor4D weightModulation;
    private NNTensor4D derWeightModulation;

    private NNVector threshold;
    private NNVector derThreshold;

    @Getter
    private NNTensor[] offset;
    private NNTensor[] modulation_input;
    private NNTensor[] modulation_output;
    private NNTensor[] offset_input;
    private NNTensor[] offset_output;

    private final int paddingY;
    private final int paddingX;
    private final int step;
    private final int heightKernel;
    private final int widthKernel;
    private final int countKernel;

    public DeformableV2ConvolutionLayer(int countKernel, int sizeKernel) {
        this(countKernel, sizeKernel, sizeKernel, 1, 0, 0);
    }

    public DeformableV2ConvolutionLayer(int countKernel, int sizeKernel, int step) {
        this(countKernel, sizeKernel, sizeKernel, step, 0, 0);
    }

    public DeformableV2ConvolutionLayer(int countKernel, int sizeKernel, int step, int padding) {
        this(countKernel, sizeKernel, sizeKernel, step, padding, padding);
    }

    public DeformableV2ConvolutionLayer(int countKernel, int heightKernel, int widthKernel, int step, int paddingY, int paddingX) {
        this.countKernel = countKernel;
        this.paddingX = paddingX;
        this.paddingY = paddingY;
        this.step = step;
        this.heightKernel = heightKernel;
        this.widthKernel = widthKernel;
        trainable = true;

        initializer = new Initializer.XavierUniform();
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

        derThreshold = new NNVector(countKernel);
        derWeight = new NNTensor4D(countKernel, heightKernel, widthKernel, depth);
        derWeightOffset = new NNTensor4D(2 * widthKernel * heightKernel, 3, 3, depth);
        derWeightModulation = new NNTensor4D(widthKernel * heightKernel, 3, 3, depth);

        if (!loadWeight) {
            weight = new NNTensor4D(countKernel, heightKernel, widthKernel, depth);
            weightOffset = new NNTensor4D(2 * widthKernel * heightKernel, 3, 3, depth);
            weightModulation = new NNTensor4D(widthKernel * heightKernel, 3, 3, depth);
            threshold = new NNVector(countKernel);

            initializer.initialize(weight);
            initializer.initialize(weightOffset);
            initializer.initialize(weightModulation);
        }
    }

    @Override
    public void initialize(Optimizer optimizer) {
        optimizer.addDataOptimize(weight, derWeight);
        optimizer.addDataOptimize(weightOffset, derWeightOffset);
        optimizer.addDataOptimize(weightModulation, derWeightModulation);
        optimizer.addDataOptimize(threshold, derThreshold);
    }

    @Override
    public void generateOutput(NNArray[] inputs) {
        this.input = NNArrays.isTensor(inputs);
        output = new NNTensor[inputs.length];
        offset = new NNTensor[inputs.length];
        modulation_input = new NNTensor[inputs.length];
        modulation_output = new NNTensor[inputs.length];
        offset_input = new NNTensor[inputs.length];
        offset_output = new NNTensor[inputs.length];

        ExecutorService executor = Executors.newFixedThreadPool(inputs.length);
        for (int t = 0; t < inputs.length; t++) {
            final int i = t;
            executor.execute(() -> {
                output[i] = new NNTensor(outHeight, outWidth, outDepth);
                offset[i] = new NNTensor(outHeight, outWidth, weightOffset.depth());
                modulation_input[i] = new NNTensor(outHeight, outWidth, weightModulation.depth());
                modulation_output[i] = new NNTensor(outHeight, outWidth, weightModulation.depth());
                offset_input[i] = new NNTensor(outHeight * heightKernel, outWidth * widthKernel, depth);
                offset_output[i] = new NNTensor(outHeight * heightKernel, outWidth * widthKernel, depth);

                offset[i].convolution(input[i], weightOffset, step, 1, 1);
                modulation_input[i].convolution(input[i], weightModulation, step, 1, 1);
                modulation_output[i].sigmoid(modulation_input[i]);

                offset_input[i].deformableConvolution(input[i], offset[i], heightKernel, widthKernel, step, paddingY, paddingX);
                offset_output[i].modulatedConvolution(offset_input[i],modulation_output[i], heightKernel, widthKernel);
                output[i].convolution(offset_output[i], weight, heightKernel, 0, 0);
                output[i].add(threshold);
            });
        }
        executor.shutdown();
        while (!executor.isTerminated()) {
        }
    }

    @Override
    public void generateOutput(CublasUtil.Matrix[] input_gpu) {

    }

    @Override
    public void generateError(NNArray[] errors) {
        errorNL = getErrorNextLayer(errors);
        error = new NNTensor[errors.length];

        ExecutorService executor = Executors.newFixedThreadPool(input.length);
        for (int t = 0; t < input.length; t++) {
            final int i = t;
            executor.execute(() -> {
                error[i] = new NNTensor(height, width, depth);
                NNTensor error_offset = new NNTensor(outHeight * heightKernel, outWidth * widthKernel, depth);
                error_offset.transposeConvolution(errorNL[i].stride(heightKernel), weight, heightKernel, 0, 0);
                NNTensor delta_offset = new NNTensor(outHeight, outWidth, weightOffset.depth());
                NNTensor modulation_error = new NNTensor(outHeight, outWidth, weightModulation.depth());
                NNTensor modulation_delta = new NNTensor(outHeight, outWidth, weightModulation.depth());

                modulation_error.backModulatedConvolution(offset_input[i], error_offset, heightKernel, widthKernel);
                modulation_delta.derSigmoid(modulation_output[i], modulation_error);

                error_offset.modulatedConvolution(error_offset, modulation_output[i], heightKernel, widthKernel);
                error[i].backDeformableConvolution(input[i], offset[i], error_offset, delta_offset,
                        heightKernel, widthKernel, step, paddingY, paddingX);

                error[i].transposeConvolution(delta_offset.stride(step), weightOffset, step, 1, 1);
                error[i].transposeConvolution(modulation_delta.stride(step), weightModulation, step, 1, 1);

                if (trainable) {
                    derWeight.convolution(offset_input[i], errorNL[i], heightKernel, 0, 0);
                    derWeightOffset.convolution(input[i], delta_offset, step, 1, 1);
                    derWeightModulation.convolution(input[i], modulation_delta, step, 1, 1);
                    derThreshold.add(errorNL[i]);
                }
            });
        }
        executor.shutdown();
        while (!executor.isTerminated()) {
        }

        if (regularization != null && trainable) {
            regularization.regularization(weight);
            regularization.regularization(weightOffset);
            regularization.regularization(weightModulation);
            regularization.regularization(threshold);
        }
    }

    @Override
    public CublasUtil.Matrix[] getOutput_gpu() {
        return new CublasUtil.Matrix[0];
    }

    @Override
    public CublasUtil.Matrix[] getError_gpu() {
        return new CublasUtil.Matrix[0];
    }

    @Override
    public int info() {
        int countParam = weight.size() + threshold.size() + weightOffset.size() + weightModulation.size();
        System.out.println("MDeform conv| " + height + ",\t" + width + ",\t" + depth + "\t| "
                + outHeight + ",\t" + outWidth + ",\t" + outDepth + "\t|\t" + countParam);
        return countParam;
    }

    @Override
    public void save(FileWriter writer) throws IOException {
        writer.write("Modulated deformable convolution layer 3D\n");
        writer.write(countKernel + " " + heightKernel + " " + widthKernel + " " + step + " " + paddingY + " " + paddingX + "\n");
        threshold.save(writer);
        weight.save(writer);
        weightOffset.save(writer);
        weightModulation.save(writer);
        if (regularization != null) {
            regularization.write(writer);
        } else {
            writer.write("null\n");
        }
        writer.write(trainable + "\n");
        writer.flush();
    }

    public static DeformableV2ConvolutionLayer read(Scanner scanner) {
        int[] param = Arrays.stream(scanner.nextLine().split(" ")).mapToInt(Integer::parseInt).toArray();

        DeformableV2ConvolutionLayer layer = new DeformableV2ConvolutionLayer(param[0], param[1], param[2], param[3], param[4], param[5]);
        layer.loadWeight = false;
        layer.threshold = NNVector.read(scanner);
        layer.weight = NNTensor4D.read(scanner);
        layer.weightOffset = NNTensor4D.read(scanner);
        layer.weightModulation = NNTensor4D.read(scanner);
        layer.setRegularization(Regularization.read(scanner));
        layer.setTrainable(Boolean.parseBoolean(scanner.nextLine()));
        layer.loadWeight = true;
        return layer;
    }

    public DeformableV2ConvolutionLayer setRegularization(Regularization regularization) {
        this.regularization = regularization;
        return this;
    }

    public DeformableV2ConvolutionLayer setTrainable(boolean trainable) {
        this.trainable = trainable;
        return this;
    }

    public DeformableV2ConvolutionLayer setInitializer(Initializer initializer) {
        this.initializer = initializer;
        return this;
    }

    public void setLoadWeight(boolean loadWeight) {
        this.loadWeight = loadWeight;
    }
}

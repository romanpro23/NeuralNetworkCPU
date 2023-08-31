package neural_network.layers.layer_3d;

import lombok.Setter;
import neural_network.initialization.Initializer;
import neural_network.optimizers.Optimizer;
import neural_network.regularization.Regularization;
import nnarrays.*;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Scanner;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class SNConvolutionLayer extends NeuralLayer3D {
    //trainable parts
    private Regularization regularization;
    private Initializer initializer;

    private boolean loadWeight;
    //weightAttention
    @Setter
    private NNTensor4D weight;
    private NNTensor4D derWeight;

    private NNVector threshold;
    private NNVector derThreshold;

    private NNVector u;
    private NNVector v;
    private float sigma;

    private final int paddingY;
    private final int paddingX;
    private final int step;
    private final int heightKernel;
    private final int widthKernel;
    private final int countKernel;
    private boolean useBias;

    public SNConvolutionLayer(int countKernel, int sizeKernel) {
        this(countKernel, sizeKernel, sizeKernel, 1, 0, 0);
    }

    public SNConvolutionLayer(int countKernel, int sizeKernel, int step) {
        this(countKernel, sizeKernel, sizeKernel, step, 0, 0);
    }

    public SNConvolutionLayer(int countKernel, int sizeKernel, int step, int padding) {
        this(countKernel, sizeKernel, sizeKernel, step, padding, padding);
    }

    public SNConvolutionLayer(int countKernel, int heightKernel, int widthKernel, int step, int paddingY, int paddingX) {
        this.countKernel = countKernel;
        this.paddingX = paddingX;
        this.paddingY = paddingY;
        this.step = step;
        this.heightKernel = heightKernel;
        this.widthKernel = widthKernel;
        trainable = true;
        useBias = true;

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

        if (!loadWeight) {
            weight = new NNTensor4D(countKernel, heightKernel, widthKernel, depth);
            threshold = new NNVector(countKernel);
            initializer.initialize(weight);

            u = new NNVector(countKernel);
            v = new NNVector(heightKernel * widthKernel * depth);

            Initializer initializerSN = new Initializer.RandomNormal();
            initializerSN.initialize(u);
            u.l2norm();
            initializerSN.initialize(v);
            v.l2norm();
        }
    }

    @Override
    public void initialize(Optimizer optimizer) {
        optimizer.addDataOptimize(weight, derWeight);
        optimizer.addDataOptimize(threshold, derThreshold);
    }

    private float spectralNorm() {
        NNMatrix weightM = new NNMatrix(countKernel, v.size(), weight.getData());

        v.clear();
        v.addMulT(u, weightM);
        v.l2norm();

        u.clear();
        u.addMul(v, weightM);
        u.l2norm();

        return 0.0000001f + NNArrays.sum(v.mul(u.dotT(weightM)));
    }

    @Override
    public void generateOutput(NNArray[] inputs) {
        this.input = NNArrays.isTensor(inputs);
        output = new NNTensor[inputs.length];

        sigma = spectralNorm();
        weight.div(sigma);

        ExecutorService executor = Executors.newFixedThreadPool(inputs.length);
        for (int t = 0; t < inputs.length; t++) {
            final int i = t;
            executor.execute(() -> {
                output[i] = new NNTensor(outHeight, outWidth, outDepth);
                output[i].convolution(input[i], weight, step, paddingY, paddingX);
                if (useBias) {
                    output[i].add(threshold);
                }
            });
        }
        executor.shutdown();
        while (!executor.isTerminated()) {
        }
    }

    private void backSpectralNorm() {
        NNMatrix dW = u.dot(v);
        dW.oneSub();
        dW.mul(weight);
        dW.div(sigma);

        derWeight.mul(dW);
        weight.mul(sigma);
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
                error[i].transposeConvolution(errorNL[i].stride(step), weight, step, paddingY, paddingX);

                if (trainable) {
                    derWeight.convolution(input[i], errorNL[i], step, paddingY, paddingX);
                    if (useBias) {
                        derThreshold.add(errorNL[i]);
                    }
                }
            });
        }
        executor.shutdown();
        while (!executor.isTerminated()) {
        }

        if (trainable) {
            backSpectralNorm();
            if (regularization != null) {
                regularization.regularization(weight);
                regularization.regularization(threshold);
            }
        }
    }

    @Override
    public int info() {
        int countParam = weight.size() + threshold.size();
        System.out.println("SN conv\t\t| " + height + ",\t" + width + ",\t" + depth + "\t| "
                + outHeight + ",\t" + outWidth + ",\t" + outDepth + "\t|\t" + countParam);
        return countParam;
    }

    @Override
    public void save(FileWriter writer) throws IOException {
        writer.write("Spectral normalization convolution layer 3D\n");
        writer.write(countKernel + " " + heightKernel + " " + widthKernel + " " + step + " " + paddingY + " " + paddingX + "\n");
        threshold.save(writer);
        weight.save(writer);

        u.save(writer);
        v.save(writer);
        if (regularization != null) {
            regularization.write(writer);
        } else {
            writer.write("null\n");
        }
        writer.write(trainable + "\n");
        writer.write(useBias + "\n");
        writer.flush();
    }

    public static SNConvolutionLayer read(Scanner scanner) {
        int[] param = Arrays.stream(scanner.nextLine().split(" ")).mapToInt(Integer::parseInt).toArray();

        SNConvolutionLayer layer = new SNConvolutionLayer(param[0], param[1], param[2], param[3], param[4], param[5]);
        layer.loadWeight = false;
        layer.threshold = NNVector.read(scanner);
        layer.weight = NNTensor4D.read(scanner);

        layer.u = NNVector.read(scanner);
        layer.v = NNVector.read(scanner);

        layer.setRegularization(Regularization.read(scanner));
        layer.setTrainable(Boolean.parseBoolean(scanner.nextLine()));
        layer.setUseBias(Boolean.parseBoolean(scanner.nextLine()));
        layer.loadWeight = true;
        return layer;
    }

    public SNConvolutionLayer setRegularization(Regularization regularization) {
        this.regularization = regularization;
        return this;
    }

    public SNConvolutionLayer setTrainable(boolean trainable) {
        this.trainable = trainable;
        return this;
    }

    public SNConvolutionLayer setUseBias(boolean useBias) {
        this.useBias = useBias;
        return this;
    }

    public SNConvolutionLayer setInitializer(Initializer initializer) {
        this.initializer = initializer;
        return this;
    }
}

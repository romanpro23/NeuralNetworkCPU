package neural_network.layers.convolution_3d;

import lombok.Getter;
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

public class SNConvolutionTransposeLayer extends ConvolutionNeuralLayer {
    //trainable parts
    private Regularization regularization;
    private Initializer initializer;

    private boolean loadWeight;

    //weightAttention
    @Setter
    private NNTensor4D weight;
    @Getter
    private NNTensor4D derWeight;
    @Setter
    private NNVector threshold;
    private NNVector derThreshold;

    private NNVector u;
    private NNVector v;
    private float sigma;

    private final int paddingY;
    private final int paddingX;
    private final int stride;
    private final int heightKernel;
    private final int widthKernel;
    private final int countKernel;
    private boolean useBias;

    public SNConvolutionTransposeLayer(int countKernel, int sizeKernel) {
        this(countKernel, sizeKernel, sizeKernel, 1, 0, 0);
    }

    public SNConvolutionTransposeLayer(int countKernel, int sizeKernel, int stride) {
        this(countKernel, sizeKernel, sizeKernel, stride, 0, 0);
    }

    public SNConvolutionTransposeLayer(int countKernel, int sizeKernel, int stride, int padding) {
        this(countKernel, sizeKernel, sizeKernel, stride, padding, padding);
    }

    public SNConvolutionTransposeLayer(int countKernel, int heightKernel, int widthKernel, int stride, int paddingY, int paddingX) {
        this.countKernel = countKernel;
        this.paddingX = paddingX;
        this.paddingY = paddingY;
        this.stride = stride;
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

        int h = height, w = width;
        if (stride > 1) {
            h = (height - 1) * stride + heightKernel % stride;
            w = (width - 1) * stride + widthKernel % stride;
        }

        outWidth = widthKernel - 2 - 2 * paddingX + w + stride;
        outHeight = heightKernel - 2 - 2 * paddingY + h + stride;
        outDepth = countKernel;

        derThreshold = new NNVector(countKernel);
        derWeight = new NNTensor4D(depth, heightKernel, widthKernel, countKernel);
        if (!loadWeight) {
            weight = new NNTensor4D(depth, heightKernel, widthKernel, countKernel);
            threshold = new NNVector(countKernel);
            initializer.initialize(weight);

            u = new NNVector(depth);
            v = new NNVector(heightKernel * widthKernel * countKernel);

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
        NNMatrix weightM = new NNMatrix(depth, v.size(), weight.getData());

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
        NNTensor[] inputData = NNArrays.isTensor(inputs);
        input = new NNTensor[inputs.length];
        output = new NNTensor[inputs.length];

        sigma = spectralNorm();
        weight.div(sigma);

        int countC = getCountCores();
        ExecutorService executor = Executors.newFixedThreadPool(countC);
        for (int t = 0; t < countC; t++) {
            final int firstIndex = t * inputData.length / countC;
            final int lastIndex = Math.min(inputData.length, (t + 1) * inputData.length / countC);
            executor.execute(() -> {
                for (int i = firstIndex; i < lastIndex; i++) {
                    input[i] = inputData[i].stride(stride);
                    output[i] = new NNTensor(outHeight, outWidth, outDepth);
                    output[i].transposeConvolution(input[i], weight, paddingY, paddingX);
                    if (useBias) {
                        output[i].add(threshold);
                    }
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

        int countC = getCountCores();
        ExecutorService executor = Executors.newFixedThreadPool(countC);
        for (int t = 0; t < countC; t++) {
            final int firstIndex = t * errors.length / countC;
            final int lastIndex = Math.min(errors.length, (t + 1) * errors.length / countC);
            executor.execute(() -> {
                for (int i = firstIndex; i < lastIndex; i++) {
                    error[i] = new NNTensor(height, width, depth);
                    error[i].convolution(errorNL[i], weight, stride, paddingY, paddingX);

                    if (trainable) {
                        derWeight.convolutionTranspose(input[i], errorNL[i], paddingY, paddingX);
                        if (useBias) {
                            derThreshold.add(errorNL[i]);
                        }
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
        System.out.println("SNTrans conv| " + height + ",\t" + width + ",\t" + depth + "\t|  "
                + outHeight + ",\t" + outWidth + ",\t" + outDepth + "\t|\t" + countParam);
        return countParam;
    }

    @Override
    public void save(FileWriter writer) throws IOException {
        writer.write("Spectral normalization convolution transpose layer 3D\n");
        writer.write(countKernel + " " + heightKernel + " " + widthKernel + " " + stride + " " + paddingY + " " + paddingX + "\n");
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

    public static SNConvolutionTransposeLayer read(Scanner scanner) {
        int[] param = Arrays.stream(scanner.nextLine().split(" ")).mapToInt(Integer::parseInt).toArray();

        SNConvolutionTransposeLayer layer = new SNConvolutionTransposeLayer(param[0], param[1], param[2], param[3], param[4], param[5]);
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

    public SNConvolutionTransposeLayer setRegularization(Regularization regularization) {
        this.regularization = regularization;
        return this;
    }

    public SNConvolutionTransposeLayer setTrainable(boolean trainable) {
        this.trainable = trainable;
        return this;
    }

    public SNConvolutionTransposeLayer setUseBias(boolean useBias) {
        this.useBias = useBias;
        return this;
    }

    public SNConvolutionTransposeLayer setInitializer(Initializer initializer) {
        this.initializer = initializer;
        return this;
    }

    public void setLoadWeight(boolean loadWeight) {
        this.loadWeight = loadWeight;
    }
}

package neural_network.layers.reshape;

import lombok.Setter;
import neural_network.initialization.Initializer;
import neural_network.layers.NeuralLayer;
import neural_network.optimizers.Optimizer;
import neural_network.regularization.Regularization;
import nnarrays.*;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Scanner;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class ImagePatchesLayer extends NeuralLayer {
    //trainable parts
    private Regularization regularization;
    private Initializer initializer;
    @Setter
    private boolean loadWeight;

    protected NNTensor[] input;
    protected NNMatrix[] patches;
    protected NNMatrix[] output;
    protected NNTensor[] error;
    protected NNMatrix[] errorNL;

    private final int sizeKernel;
    private final int lengthVector;
    private boolean returnGradient;

    private NNMatrix weight;
    private NNMatrix derWeight;

    private int height, width, depth;
    private int outWidth, outDepth;

    public ImagePatchesLayer(int sizeKernel, int lengthVector) {
        this.sizeKernel = sizeKernel;
        this.lengthVector = lengthVector;

        initializer = new Initializer.HeNormal();
        trainable = true;
        returnGradient = false;
    }

    @Override
    public void initialize(int[] size) {
        if (size.length != 3) {
            throw new ExceptionInInitializerError("Error size pre layer!");
        }
        height = size[0];
        width = size[1];
        depth = size[2];

        outWidth = (height / sizeKernel) * (width / sizeKernel);
        outDepth = lengthVector;

        derWeight = new NNMatrix(depth * sizeKernel * sizeKernel, lengthVector);

        if (!loadWeight) {
            weight = new NNMatrix(depth * sizeKernel * sizeKernel, lengthVector);
            initializer.initialize(weight);
        }
    }

    @Override
    public void generateOutput(NNArray[] inputs) {
        this.input = NNArrays.isTensor(inputs);
        this.output = new NNMatrix[input.length];
        this.patches = new NNMatrix[input.length];

        ExecutorService executor = Executors.newFixedThreadPool(inputs.length);
        for (int t = 0; t < inputs.length; t++) {
            final int i = t;
            executor.execute(() -> {
                patches[i] = input[i].imageVector(sizeKernel);
                output[i] = patches[i].dot(weight);
            });
        }
        executor.shutdown();
        while (!executor.isTerminated()) {
        }
    }

    @Override
    public void generateTrainOutput(NNArray[] input) {
        generateOutput(input);
    }

    @Override
    public void generateError(NNArray[] errors) {
        errorNL = getErrorNextLayer(errors);
        if (returnGradient)
            error = new NNTensor[errors.length];

        ExecutorService executor = Executors.newFixedThreadPool(input.length);
        for (int t = 0; t < input.length; t++) {
            final int i = t;
            executor.execute(() -> {
                if(returnGradient) {
                    NNMatrix errorImg = errorNL[i].dotT(weight);
                    error[i] = input[i].backImageVector(errorImg, sizeKernel);
                }
                if(trainable){
                    derWeight.add(patches[i].transpose().dot(errorNL[i]));
                }
            });
        }
        executor.shutdown();
        while (!executor.isTerminated()) {
        }

        if (trainable && regularization != null) {
            regularization.regularization(weight);
        }
    }

    @Override
    public int[] size() {
        return new int[]{outWidth, outDepth};
    }

    @Override
    public void initialize(Optimizer optimizer) {
        optimizer.addDataOptimize(weight, derWeight);
    }

    @Override
    public int info() {
        int countParam = weight.size();
        System.out.println("ImgPatches\t| " + height + ",\t" + width + ",\t" + depth + "\t| "
                + outWidth + ",\t" + outDepth + "\t\t|\t" + countParam);
        return countParam;
    }

    @Override
    public void save(FileWriter writer) throws IOException {
        writer.write("Image patches layer\n");
        writer.write( sizeKernel + " " +  sizeKernel + " " + lengthVector + "\n");
        writer.write(this.returnGradient + "\n");
        weight.save(writer);
        if (regularization != null) {
            regularization.write(writer);
        } else {
            writer.write("null\n");
        }
        writer.write(trainable + "\n");
        writer.flush();
    }

    @Override
    public NNArray[] getOutput() {
        return output;
    }

    @Override
    public NNArray[] getError() {
        return error;
    }

    public static ImagePatchesLayer read(Scanner scanner) {
        int[] arr = Arrays.stream(scanner.nextLine().split(" ")).mapToInt(Integer::parseInt).toArray();
        ImagePatchesLayer layer = new ImagePatchesLayer(arr[0], arr[2]);
        layer.setReturnGradient(Boolean.parseBoolean(scanner.nextLine()));
        layer.weight = NNMatrix.read(scanner);
        layer.setRegularization(Regularization.read(scanner));
        layer.setTrainable(Boolean.parseBoolean(scanner.nextLine()));
        layer.loadWeight = true;
        return layer;
    }

    public ImagePatchesLayer setRegularization(Regularization regularization) {
        this.regularization = regularization;
        return this;
    }

    public ImagePatchesLayer setTrainable(boolean trainable) {
        this.trainable = trainable;
        return this;
    }

    public ImagePatchesLayer setInitializer(Initializer initializer) {
        this.initializer = initializer;
        return this;
    }

    public ImagePatchesLayer setReturnGradient(boolean returnGradient) {
        this.returnGradient = returnGradient;

        return this;
    }

    public NNMatrix[] getErrorNextLayer(NNArray[] error) {
        NNMatrix[] errorNL = NNArrays.isMatrix(error);

        if (!nextLayers.isEmpty()) {
            for (int i = 0; i < errorNL.length; i++) {
                for (NeuralLayer nextLayer : nextLayers) {
                    errorNL[i].add(nextLayer.getErrorNL()[i]);
                }
            }
        }
        return errorNL;
    }
}

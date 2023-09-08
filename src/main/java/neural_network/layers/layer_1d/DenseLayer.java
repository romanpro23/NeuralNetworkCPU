package neural_network.layers.layer_1d;

import jcuda.Pointer;
import jcuda.driver.CUfunction;
import jcuda.driver.JCudaDriver;
import lombok.Getter;
import lombok.Setter;
import lombok.SneakyThrows;
import neural_network.initialization.Initializer;
import neural_network.optimizers.Optimizer;
import neural_network.regularization.Regularization;
import nnarrays.NNArray;
import nnarrays.NNArrays;
import nnarrays.NNMatrix;
import nnarrays.NNVector;
import utilities.Use;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static nnarrays.NNArray.BLOCK_SIZE;
import static utilities.GPUInit.helperModule;
import static utilities.JCudaHelper.CONTEXT;

public class DenseLayer extends DenseNeuralLayer {
    //trainable parts
    private Regularization regularization;
    private Initializer initializer;
    @Setter
    private boolean loadWeight;

    //weightAttention and threshold
    @Getter
    private NNMatrix weight;
    private NNMatrix derWeight;

    private NNVector threshold;
    private NNVector derThreshold;

    public DenseLayer(int countNeuron) {
        super();
        this.countNeuron = countNeuron;
        this.trainable = true;
        initializer = new Initializer.HeNormal();
    }

    @Override
    public void initialize(Optimizer optimizer) {
        optimizer.addDataOptimize(weight, derWeight);
        optimizer.addDataOptimize(threshold, derThreshold);
    }

    public DenseLayer setTrainable(boolean trainable) {
        this.trainable = trainable;

        return this;
    }

    public DenseLayer setInitializer(Initializer initializer) {
        this.initializer = initializer;

        return this;
    }

    public DenseLayer setRegularization(Regularization regularization) {
        this.regularization = regularization;

        return this;
    }

    @Override
    public int info() {
        int countParam = weight.size() + threshold.size();
        System.out.println("Dense \t\t|  " + weight.getColumn() + "\t\t\t|  " + countNeuron + "\t\t\t|\t" + countParam);
        return countParam;
    }

    @Override
    public void save(FileWriter writer) throws IOException {
        writer.write("Dense layer\n");
        writer.write(countNeuron + "\n");
        threshold.save(writer);
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
    public void initialize(int[] size) {
        if (size.length != 1) {
            throw new ExceptionInInitializerError("Error size pre layer!");
        }
        derThreshold = new NNVector(countNeuron);
        derWeight = new NNMatrix(countNeuron, size[0]);

        if (!loadWeight) {
            threshold = new NNVector(countNeuron);
            weight = new NNMatrix(countNeuron, size[0]);
            initializer.initialize(weight);
        }
    }

    @SneakyThrows
    @Override
    public void generateOutput(NNArray[] inputs) {
        this.input = NNArrays.isVector(inputs);
        this.output = new NNVector[input.length];


        ExecutorService executor = Executors.newFixedThreadPool(inputs.length);
        for (int t = 0; t < inputs.length; t++) {
            final int i = t;
            executor.execute(() -> {
                if (Use.GPU) {
                    JCudaDriver.cuCtxSetCurrent(CONTEXT);
                }

                output[i] = input[i].dot(weight);
                output[i].add(threshold);
            });
        }
        executor.shutdown();
        while (!executor.isTerminated()) {
        }
    }

    @SneakyThrows
    @Override
    public void generateError(NNArray[] errors) {
        errorNL = getErrorNextLayer(errors);
        this.error = new NNVector[errors.length];

        ExecutorService executor = Executors.newFixedThreadPool(input.length);
        for (int t = 0; t < input.length; t++) {
            final int i = t;
            executor.execute(() -> {
                if (Use.GPU) {
                    JCudaDriver.cuCtxSetCurrent(CONTEXT);
                }

                error[i] = errorNL[i].dotT(weight);
                if (trainable) {
                    derivativeWeight(input[i], errorNL[i]);
                }
            });
        }
        executor.shutdown();
        while (!executor.isTerminated()) {
        }

        if (trainable && regularization != null) {
            regularization.regularization(weight);
            regularization.regularization(threshold);
        }
    }

    private void derivativeWeight(NNVector input, NNVector error) {
        if (Use.CPU) {
            for (int j = 0, index = 0; j < error.size(); j++) {
                for (int k = 0; k < input.size(); k++, index++) {
                    derWeight.getData()[index] += error.getData()[j] * input.getData()[k];
                }
            }
        }

        if (Use.GPU) {
            int row = error.size();
            int column = input.size();
            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, helperModule, "derivativeWeight");
            Pointer kernelParameters = Pointer.to(Pointer.to(input.getData_gpu()), Pointer.to(error.getData_gpu()), Pointer.to(derWeight.getData_gpu()),  Pointer.to(new int[]{row}), Pointer.to(new int[]{column}));
            int blockSizeX = (int) Math.min(row, Math.pow(BLOCK_SIZE, (double) 1 / 2));
            int blockSizeY = (int) Math.min(column, Math.pow(BLOCK_SIZE, (double) 1 / 2));
            int gridSizeX = (int) Math.ceil((double) row / blockSizeX);
            int gridSizeY = (int) Math.ceil((double) column / blockSizeY);

            cuLaunchKernel(function,
                    gridSizeX, gridSizeY, 1,      // Grid dimension
                    blockSizeX, blockSizeY, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
            if (Use.DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
        }
        derThreshold.add(error);
    }

    public static DenseLayer read(Scanner scanner) {
        DenseLayer denseLayer = new DenseLayer(Integer.parseInt(scanner.nextLine()));
        denseLayer.threshold = NNVector.read(scanner);
        denseLayer.weight = NNMatrix.read(scanner);
        denseLayer.setRegularization(Regularization.read(scanner));
        denseLayer.setTrainable(Boolean.parseBoolean(scanner.nextLine()));
        denseLayer.loadWeight = true;
        return denseLayer;
    }
}

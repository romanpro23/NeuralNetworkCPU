package neural_network.layers.layer_1d;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUfunction;
import jcuda.driver.JCudaDriver;
import jcuda.runtime.JCuda;
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
import static jcuda.runtime.JCuda.cudaMalloc;
import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.cudaFuncAttribute.cudaFuncAttributeMaxDynamicSharedMemorySize;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;
import static nnarrays.NNArray.BLOCK_SIZE;
import static utilities.GPUInit.helperModule;
import static utilities.JCudaHelper.CONTEXT;
import static utilities.Use.*;

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
        optimizer.addDataOptimize(weight, derWeight, "Dense layer");
        optimizer.addDataOptimize(threshold, derThreshold, "Dense layer");
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
        derThreshold = new NNVector(countNeuron, half);
        derWeight = new NNMatrix(countNeuron, size[0], half);

        if (!loadWeight) {
            threshold = new NNVector(countNeuron, half);
            weight = new NNMatrix(countNeuron, size[0], half);
            initializer.initialize(weight);
        }
    }

    @SneakyThrows
    @Override
    public void generateOutput(NNArray[] inputs) {
        this.input = NNArrays.isVector(inputs);
        this.output = new NNVector[input.length];

        if ((Use.CPU) && (!Use.GPU)) {
            GPU_Sleep();
            ExecutorService executor = Executors.newFixedThreadPool(inputs.length);
            for (int t = 0; t < inputs.length; t++) {
                final int i = t;
                executor.execute(() -> {
                    output[i] = input[i].dot(weight);
                    output[i].add(threshold);
                });
            }
            executor.shutdown();
            while (!executor.isTerminated()) {
            }
            GPU_WakeUp();
        }

        if (Use.GPU) {
            for (int i = 0; i < inputs.length; i++) {
                output[i] = input[i].dot(weight);
                output[i].add(threshold);
            }

            //DenseLayerForward();
        }
    }

    private void DenseLayerForward()
    {
        int numSteps = input.length;
        Pointer[][] P = new Pointer[2][numSteps];
        int row = weight.getRow();
        int column = weight.getColumn();

        for(int k = 0; k < numSteps; k++)
        {
            output[k] = new NNVector(row);
            P[0][k] = input[k].getData_gpu();
            P[1][k] = output[k].getData_gpu();
        }

        Pointer[] Array = new Pointer[P.length];
        for(int k = 0; k < P.length; k++) {
            Array[k] = new Pointer();
            cudaMalloc(Array[k], (long) numSteps * Sizeof.POINTER);
            cudaMemcpy(Array[k], Pointer.to(P[k]), (long) numSteps * Sizeof.POINTER, cudaMemcpyHostToDevice);
        }

        Pointer PArray = new Pointer();
        cudaMalloc(PArray, (long) numSteps * Sizeof.POINTER);
        cudaMemcpy(PArray, Pointer.to(Array), (long) numSteps * Sizeof.POINTER, cudaMemcpyHostToDevice);

        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, helperModule, "DenseLayerForward");
        Pointer kernelParameters = Pointer.to(Pointer.to(PArray), Pointer.to(weight.getData_gpu()), Pointer.to(threshold.getData_gpu()), Pointer.to(new int[]{row}), Pointer.to(new int[]{column}), Pointer.to(new int[]{numSteps}));
        int blockSizeX = (int) Math.min(numSteps, Math.pow(BLOCK_SIZE, (double) 1 / 2));
        int blockSizeY = (int) Math.min(row, Math.pow(BLOCK_SIZE, (double) 1 / 2));
        int gridSizeX = (int) Math.ceil((double) numSteps / blockSizeX);
        int gridSizeY = (int) Math.ceil((double) row / blockSizeY);

        cuLaunchKernel(function,
                gridSizeX, gridSizeY, 1,      // Grid dimension
                blockSizeX, blockSizeY, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                kernelParameters, null // Kernel- and extra parameters
        );
        if (Use.DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();

        JCuda.cudaFree(PArray);
        for(int k = 0; k < P.length; k++) {
            JCuda.cudaFree(Array[k]);
        }
    }

    @SneakyThrows
    @Override
    public void generateError(NNArray[] errors) {
        errorNL = getErrorNextLayer(errors);
        this.error = new NNVector[errors.length];

        if ((Use.CPU) && (!Use.GPU)) {
            GPU_Sleep();
            ExecutorService executor = Executors.newFixedThreadPool(input.length);
            for (int t = 0; t < input.length; t++) {
                final int i = t;
                executor.execute(() -> {
                    error[i] = errorNL[i].dotT(weight);
                    if (trainable) {
                        derivativeWeight(input[i], errorNL[i]);
                    }
                });
            }
            executor.shutdown();
            while (!executor.isTerminated()) {
            }
            GPU_WakeUp();
        }

        if (Use.GPU) {
            for (int i = 0; i < input.length; i++) {
                error[i] = errorNL[i].dotT(weight);
                if (trainable) {
                    derivativeWeight(input[i], errorNL[i]);
                }
            }
        }

        if (trainable && regularization != null) {
            regularization.regularization(weight);
            regularization.regularization(threshold);
        }
    }

    private void derivativeWeight(NNVector input, NNVector error) {
        if (Use.CPU) {
            if (Use.GPU) {
                for (int j = 0; j < derWeight.getRow(); j++) {
                    for (int k = 0; k < derWeight.getColumn(); k++) {
                        derWeight.set(j, k, 0);
                    }
                }
            }
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
            Pointer kernelParameters = null;
            if (!input.isHalf()) {
                cuModuleGetFunction(function, helperModule, "derivativeWeight");
                kernelParameters = Pointer.to(Pointer.to(input.getData_gpu()), Pointer.to(error.getData_gpu()), Pointer.to(derWeight.getData_gpu()), Pointer.to(new int[]{row}), Pointer.to(new int[]{column}));
            }
            else
            {
                cuModuleGetFunction(function, helperModule, "derivativeWeight_half");
                kernelParameters = Pointer.to(Pointer.to(input.getData_gpu()), Pointer.to(error.getData_gpu()), Pointer.to(derWeight.getData_gpu()), Pointer.to(new int[]{row}), Pointer.to(new int[]{column}));
            }
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
            if (Use.DEBUG_SYNC) {
                JCudaDriver.cuCtxSynchronize();
                if (!input.isHalf()) {
                    input.IsNan_float(input);
                    error.IsNan_float(error);
                    derWeight.IsNan_float(derWeight);
                }
                else
                {
                    input.IsNan(input);
                    error.IsNan(error);
                    derWeight.IsNan(derWeight);
                }
            }
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

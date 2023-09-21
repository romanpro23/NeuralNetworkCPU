package neural_network.layers.layer_2d;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;
import jcuda.runtime.JCuda;
import lombok.Setter;
import neural_network.optimizers.Optimizer;
import neural_network.regularization.Regularization;
import nnarrays.*;
import utilities.Use;

import java.io.FileWriter;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.util.Scanner;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import static jcuda.driver.JCudaDriver.*;
import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;
import static nnarrays.NNArray.BLOCK_SIZE;
import static utilities.GPUInit.helperModule;
import static utilities.JCudaHelper.CONTEXT;
import static utilities.Use.*;

public class NormalizationLayer2D extends NeuralLayer2D {
    //trainable parts
    private Regularization regularization;
    @Setter
    private boolean loadWeight;

    private final float epsilon;

    //betta
    @Setter
    private NNVector betta;
    private NNVector derBetta;
    //gamma
    @Setter
    private NNVector gamma;
    private NNVector derGamma;

    private NNMatrix[] normOutput;
    private NNVector[] mean, var;

    public NormalizationLayer2D() {
        this.trainable = true;
        this.epsilon = 0.001f;
    }

    @Override
    public void initialize(int[] size) {
        if (size.length != 2) {
            throw new ExceptionInInitializerError("Error depth pre layer!");
        }

        this.depth = size[1];
        this.width = size[0];

        outDepth = depth;
        outWidth = width;

        derBetta = new NNVector(depth);
        derGamma = new NNVector(depth);

        if (!loadWeight) {
            betta = new NNVector(depth);
            gamma = new NNVector(depth);

            gamma.fill(1);
        }
    }

    @Override
    public void initialize(Optimizer optimizer) {
        optimizer.addDataOptimize(betta, derBetta);
        optimizer.addDataOptimize(gamma, derGamma);
    }

    @Override
    public void generateOutput(NNArray[] input) {
        this.input = NNArrays.isMatrix(input);
        this.output = new NNMatrix[input.length];
        this.normOutput = new NNMatrix[input.length];
        this.mean = new NNVector[input.length];
        this.var = new NNVector[input.length];

        if (Use.CPU) {
            GPU_Sleep();
            ExecutorService executor = Executors.newFixedThreadPool(input.length);
            for (int t = 0; t < input.length; t++) {
                final int i = t;
                executor.execute(() -> {
                    normOutput[i] = new NNMatrix(outWidth, outDepth);
                    output[i] = new NNMatrix(outWidth, outDepth);
                    findMean(i);
                    findVariance(i);
                    normalization(i);
                });
            }
            executor.shutdown();
            while (!executor.isTerminated()) {
            }
            GPU_WakeUp();
        }

        if (Use.GPU) {
            for (int i = 0; i < input.length; i++) {
                normOutput[i] = new NNMatrix(outWidth, outDepth);
                output[i] = new NNMatrix(outWidth, outDepth);
                findMean(i);
                findVariance(i);
                normalization(i);
            }
            //NormalizationLayerForward2D();
        }
    }

    private void NormalizationLayerForward2D()
    {
        int numSteps = input.length;
        Pointer[][] P = new Pointer[5][numSteps];

        for(int k = 0; k < numSteps; k++)
        {
            normOutput[k] = new NNMatrix(outWidth, outDepth);
            output[k] = new NNMatrix(outWidth, outDepth);
            mean[k] = new NNVector(width);
            var[k] = new NNVector(width);
            P[0][k] = normOutput[k].getData_gpu();
            P[1][k] = output[k].getData_gpu();
            P[2][k] = input[k].getData_gpu();
            P[3][k] = mean[k].getData_gpu();
            P[4][k] = var[k].getData_gpu();
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
        cuModuleGetFunction(function, helperModule, "NormalizationLayerForward2D");
        Pointer kernelParameters = Pointer.to(Pointer.to(PArray), Pointer.to(gamma.getData_gpu()), Pointer.to(betta.getData_gpu()), Pointer.to(new int[]{width}), Pointer.to(new int[]{depth}), Pointer.to(new int[]{numSteps}));
        int blockSizeX = (int) Math.min(numSteps, Math.pow(BLOCK_SIZE, (double) 1 / 2));
        int blockSizeY = (int) Math.min(width, Math.pow(BLOCK_SIZE, (double) 1 / 2));
        int gridSizeX = (int) Math.ceil((double) numSteps / blockSizeX);
        int gridSizeY = (int) Math.ceil((double) width / blockSizeY);

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

    private void normalization(int n) {
        if (Use.CPU) {
            float[] varSqrt = new float[var[n].size()];
            for (int i = 0; i < var[n].size(); i++) {
                varSqrt[i] = (float) (Math.sqrt(var[n].getData()[i] + epsilon));
            }
            int index = 0;
            for (int j = 0; j < width; j++) {
                for (int k = 0; k < depth; k++, index++) {
                    normOutput[n].getData()[index] = ((float)(input[n].getData()[index] - mean[n].get(j))) / varSqrt[j];
                    output[n].getData()[index] = normOutput[n].getData()[index] * gamma.get(k) + betta.get(k);
                }
            }
        }

        if (Use.GPU) {
            int p = var[n].size();
            Pointer varSqrt_Pointer = new Pointer();
            cudaMalloc(varSqrt_Pointer, (long) p * Sizeof.FLOAT);
            cudaMemset(varSqrt_Pointer, 0, (long) p * Sizeof.FLOAT);

            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, helperModule, "normalization_part_1");
            Pointer kernelParameters = Pointer.to(Pointer.to(varSqrt_Pointer), Pointer.to(var[n].getData_gpu()), Pointer.to(new int[]{p}));
            int blockSize = Math.min(p, BLOCK_SIZE);
            int gridSizeX = (int) Math.ceil((double) p / blockSize);
            cuLaunchKernel(function,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSize, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );

            if (Use.DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();


            CUfunction function2 = new CUfunction();
            cuModuleGetFunction(function2, helperModule, "normalization_part_2");
            Pointer kernelParameters2 = Pointer.to(Pointer.to(input[n].getData_gpu()), Pointer.to(mean[n].getData_gpu()), Pointer.to(varSqrt_Pointer), Pointer.to(normOutput[n].getData_gpu()), Pointer.to(gamma.getData_gpu()), Pointer.to(betta.getData_gpu()), Pointer.to(output[n].getData_gpu()), Pointer.to(new int[]{width}), Pointer.to(new int[]{depth}));
            int blockSizeX = (int) Math.min(width, Math.pow(BLOCK_SIZE, (double) 1 / 2));
            int blockSizeY = (int) Math.min(depth, Math.pow(BLOCK_SIZE, (double) 1 / 2));
            gridSizeX = (int) Math.ceil((double) width / blockSizeX);
            int gridSizeY = (int) Math.ceil((double) depth / blockSizeY);

            cuLaunchKernel(function2,
                    gridSizeX, gridSizeY, 1,      // Grid dimension
                    blockSizeX, blockSizeY, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters2, null // Kernel- and extra parameters
            );

            if (Use.DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();

            JCuda.cudaFree(varSqrt_Pointer);
        }
    }

    private void findMean(int n) {
        mean[n] = new NNVector(width);

        if (Use.CPU) {
            int index = 0;
            for (int j = 0; j < width; j++) {
                for (int k = 0; k < depth; k++, index++) {
                    mean[n].getData()[j] += input[n].getData()[index];
                }
            }
        }

        if (Use.GPU) {
            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, helperModule, "findMean_part");
            Pointer kernelParameters = Pointer.to(Pointer.to(input[n].getData_gpu()), Pointer.to(mean[n].getData_gpu()),  Pointer.to(new int[]{width}), Pointer.to(new int[]{depth}));
            int blockSizeX = (int) Math.min(width, Math.pow(BLOCK_SIZE, (double) 1));
            int gridSizeX = (int) Math.ceil((double) width / blockSizeX);

            cuLaunchKernel(function,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSizeX, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
            if (Use.DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
        }
        mean[n].div(depth);
    }

    private void findVariance(int n) {
        var[n] = new NNVector(width);
        if (Use.CPU) {
            float sub;
            int index = 0;
            for (int j = 0; j < width; j++) {
                for (int k = 0; k < depth; k++, index++) {
                    sub = (float)(input[n].getData()[index] - mean[n].getData()[j]);
                    var[n].getData()[j] += sub * sub;
                }
            }
        }
        if (Use.GPU) {
            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, helperModule, "findVariance_part");
            Pointer kernelParameters = Pointer.to(Pointer.to(input[n].getData_gpu()), Pointer.to(mean[n].getData_gpu()), Pointer.to(var[n].getData_gpu()), Pointer.to(new int[]{width}), Pointer.to(new int[]{depth}));
            int blockSizeX = (int) Math.min(width, Math.pow(BLOCK_SIZE, 1));
            int gridSizeX = (int) Math.ceil((double) width / blockSizeX);

            cuLaunchKernel(function,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSizeX, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
            if (Use.DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
        }
        var[n].div(depth);
    }

    @Override
    public void generateError(NNArray[] errors) {
        errorNL = getErrorNextLayer(errors);
        this.error = new NNMatrix[errors.length];

        if (Use.CPU) {
            GPU_Sleep();
            ExecutorService executor = Executors.newFixedThreadPool(input.length);
            for (int t = 0; t < input.length; t++) {
                final int i = t;
                executor.execute(() -> {
                    error[i] = new NNMatrix(outWidth, outDepth);
                    NNMatrix errorNorm = generateErrorNorm(i);
                    NNVector errorVariance = derVar(errorNorm, i);
                    NNVector errorMean = derMean(errorNorm, errorVariance, i);

                    derNorm(errorNorm, errorMean, errorVariance, i);

                    if (trainable) {
                        derivativeWeight(errorNL[i], i);
                    }
                });
            }
            executor.shutdown();
            while (!executor.isTerminated()) {
            }
            GPU_WakeUp();
        }

        if (Use.GPU)
        {
            for (int i = 0; i < input.length; i++) {
                error[i] = new NNMatrix(outWidth, outDepth);
                NNMatrix errorNorm = generateErrorNorm(i);
                NNVector errorVariance = derVar(errorNorm, i);
                NNVector errorMean = derMean(errorNorm, errorVariance, i);

                derNorm(errorNorm, errorMean, errorVariance, i);

                if (trainable) {
                    derivativeWeight(errorNL[i], i);
                }
            }
        }

        if (trainable && regularization != null) {
            regularization.regularization(betta);
            regularization.regularization(gamma);
        }
    }

    private NNMatrix generateErrorNorm(int n) {
        NNMatrix errorNorm = new NNMatrix(outWidth, outDepth);

        if (Use.CPU) {
            int index = 0;
            for (int j = 0; j < width; j++) {
                for (int k = 0; k < depth; k++, index++) {
                    errorNorm.getData()[index] = errorNL[n].getData()[index] * gamma.get(k);
                }
            }
        }

        if (Use.GPU) {
            int row = width;
            int column = depth;
            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, helperModule, "generateErrorNorm");
            Pointer kernelParameters = Pointer.to(Pointer.to(errorNL[n].getData_gpu()), Pointer.to(gamma.getData_gpu()), Pointer.to(errorNorm.getData_gpu()),  Pointer.to(new int[]{row}), Pointer.to(new int[]{column}));
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

        return errorNorm;
    }

    private NNVector derVar(NNMatrix error, int n) {
        NNVector derVariance = new NNVector(var[n].size());

        if (Use.CPU) {
            float[] dVar = new float[var[n].size()];
            for (int i = 0; i < var[n].size(); i++) {
                dVar[i] = (float) (-0.5 * Math.pow(var[n].get(i) + epsilon, -1.5));
            }

            int index = 0;
            for (int j = 0; j < width; j++) {
                for (int k = 0; k < depth; k++, index++) {
                    derVariance.getData()[j] += (float)(error.get(index) * (input[n].get(index) - mean[n].get(j)));
                }
            }

            for (int i = 0; i < derVariance.size(); i++) {
                derVariance.getData()[i] *= dVar[i];
            }
        }

        if (Use.GPU) {
            Pointer dVar_Pointer = new Pointer();
            cudaMalloc(dVar_Pointer, (long) var[n].size() * Sizeof.FLOAT);
            cudaMemset(dVar_Pointer, 0,(long) var[n].size() * Sizeof.FLOAT);

            int p = var[n].size();
            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, helperModule, "derVar_part_1");
            Pointer kernelParameters = Pointer.to(Pointer.to(var[n].getData_gpu()), Pointer.to(new float[]{epsilon}), Pointer.to(dVar_Pointer), Pointer.to(new int[]{p}));
            int blockSize = Math.min(p, BLOCK_SIZE);
            int gridSizeX = (int) Math.ceil((double) p / blockSize);
            cuLaunchKernel(function,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSize, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
            if (Use.DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();

            var[n].IsNan(var[n]);

            CUfunction function2 = new CUfunction();
            cuModuleGetFunction(function2, helperModule, "derVar_part_2");
            Pointer kernelParameters2 = Pointer.to(Pointer.to(error.getData_gpu()), Pointer.to(input[n].getData_gpu()), Pointer.to(mean[n].getData_gpu()), Pointer.to(derVariance.getData_gpu()),  Pointer.to(new int[]{width}), Pointer.to(new int[]{depth}));
            int blockSizeX = (int) Math.min(width, Math.pow(BLOCK_SIZE, 1));
            gridSizeX = (int) Math.ceil((double) width / blockSizeX);

            cuLaunchKernel(function2,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSizeX, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters2, null // Kernel- and extra parameters
            );

            if (Use.DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();

            input[n].IsNan(input[n]);
            mean[n].IsNan(mean[n]);
            derVariance.IsNan(derVariance);

            p = derVariance.size();
            CUfunction function3 = new CUfunction();
            cuModuleGetFunction(function3, helperModule, "derVar_part_3");
            Pointer kernelParameters3 = Pointer.to(Pointer.to(dVar_Pointer), Pointer.to(derVariance.getData_gpu()), Pointer.to(new int[]{p}));
            blockSize = Math.min(p, BLOCK_SIZE);
            gridSizeX = (int) Math.ceil((double) p / blockSize);
            cuLaunchKernel(function3,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSize, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters3, null // Kernel- and extra parameters
            );
            if (Use.DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();

            JCuda.cudaFree(dVar_Pointer);

            derVariance.IsNan(derVariance);
        }
        return derVariance;
    }

    private NNVector derMean(NNMatrix error, NNVector derVar, int n) {
        NNVector derMean = new NNVector(mean[n].size());

        if (Use.CPU) {
            float[] dMean = new float[mean[n].size()];
            double[] dVar = new double[var[n].size()];
            for (int i = 0; i < var[n].size(); i++) {
                dMean[i] = (float) (-1.0f / Math.sqrt(var[n].get(i) + epsilon));
            }

            int index = 0;
            for (int j = 0; j < width; j++) {
                for (int k = 0; k < depth; k++, index++) {
                    derMean.getData()[j] += error.get(index);
                    dVar[j] += ((double)input[n].get(index)) - (mean[n].get(j));
                }
            }

            for (int i = 0; i < derMean.size(); i++) {
                derMean.getData()[i] *= dMean[i];
                derMean.getData()[i] += (-2.0f * derVar.get(i) * dVar[i]) / (depth);
            }
        }

        if (Use.GPU) {

            Pointer dMean_Pointer = new Pointer();
            cudaMalloc(dMean_Pointer, (long) mean[n].size() * Sizeof.FLOAT);
            cudaMemset(dMean_Pointer, 0, (long) mean[n].size() * Sizeof.FLOAT);

            int p = var[n].size();

            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, helperModule, "derMean_part_1");
            Pointer kernelParameters = Pointer.to(Pointer.to(var[n].getData_gpu()), Pointer.to(new float[]{epsilon}), Pointer.to(dMean_Pointer), Pointer.to(new int[]{p}));
            int blockSizeX = Math.min(p, BLOCK_SIZE);
            int gridSizeX = (int) Math.ceil((double) p / blockSizeX);
            cuLaunchKernel(function,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSizeX, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
            if (Use.DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();

            Pointer dVar_Pointer = new Pointer();
            cudaMalloc(dVar_Pointer, (long) p * Sizeof.DOUBLE);
            cudaMemset(dVar_Pointer, 0, (long) p * Sizeof.DOUBLE);

            CUfunction function2 = new CUfunction();
            cuModuleGetFunction(function2, helperModule, "derMean_part_2");
            Pointer kernelParameters2 = Pointer.to(Pointer.to(error.getData_gpu()), Pointer.to(input[n].getData_gpu()), Pointer.to(mean[n].getData_gpu()), Pointer.to(derMean.getData_gpu()), Pointer.to(dVar_Pointer), Pointer.to(new int[]{width}), Pointer.to(new int[]{depth}));
            blockSizeX = Math.min(width, BLOCK_SIZE);
            gridSizeX = (int) Math.ceil((double) width / blockSizeX);

            cuLaunchKernel(function2,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSizeX, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters2, null // Kernel- and extra parameters
            );
            if (Use.DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();

            p = derMean.size();
            CUfunction function3 = new CUfunction();
            cuModuleGetFunction(function3, helperModule, "derMean_part_3");
            Pointer kernelParameters3 = Pointer.to(Pointer.to(dMean_Pointer), Pointer.to(derVar.getData_gpu()), Pointer.to(dVar_Pointer), Pointer.to(new int[]{depth}), Pointer.to(derMean.getData_gpu()), Pointer.to(new int[]{p}));
            blockSizeX = Math.min(p, BLOCK_SIZE);
            gridSizeX = (int) Math.ceil((double) p / blockSizeX);
            cuLaunchKernel(function3,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSizeX, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters3, null // Kernel- and extra parameters
            );
            if (Use.DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();

            JCuda.cudaFree(dMean_Pointer);
            JCuda.cudaFree(dVar_Pointer);
        }
        return derMean;
    }

    private void derNorm(NNMatrix errors, NNVector errorMean, NNVector errorVar, int n) {
        errorMean.div(depth);
        errorVar.mul(2.0f / (depth));

        if (Use.CPU) {
            float[] dVar = new float[var[n].size()];
            for (int i = 0; i < var[n].size(); i++) {
                dVar[i] = (float) (1.0 / Math.sqrt(var[n].getData()[i] + epsilon));
            }

            int index = 0;
            for (int j = 0; j < width; j++) {
                for (int k = 0; k < depth; k++, index++) {
                    error[n].getData()[index] = errors.getData()[index] * dVar[j] + errorVar.get(j) *
                            ((float)(input[n].get(index) - mean[n].get(j))) + errorMean.get(j);
                }
            }
        }

        if (Use.GPU) {
            int p = var[n].size();

            Pointer dVar_Pointer = new Pointer();
            float[] init2 = new float[p];
            cudaMalloc(dVar_Pointer, (long) p * Sizeof.FLOAT);
            cudaMemcpy(dVar_Pointer, Pointer.to(init2), (long) p * Sizeof.FLOAT, cudaMemcpyHostToDevice);

            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, helperModule, "derNorm_part_1");
            Pointer kernelParameters = Pointer.to(Pointer.to(var[n].getData_gpu()), Pointer.to(new float[]{epsilon}), Pointer.to(dVar_Pointer), Pointer.to(new int[]{p}));
            int blockSize = Math.min(p, BLOCK_SIZE);
            int gridSizeX = (int) Math.ceil((double) p / blockSize);
            cuLaunchKernel(function,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSize, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
            if (Use.DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();

            var[n].IsNan(var[n]);

            int row = width;
            int column = depth;
            CUfunction function2 = new CUfunction();
            cuModuleGetFunction(function2, helperModule, "derNorm_part_2");
            Pointer kernelParameters2 = Pointer.to(Pointer.to(errors.getData_gpu()), Pointer.to(dVar_Pointer), Pointer.to(errorVar.getData_gpu()), Pointer.to(input[n].getData_gpu()), Pointer.to(mean[n].getData_gpu()), Pointer.to(errorMean.getData_gpu()), Pointer.to(error[n].getData_gpu()), Pointer.to(new int[]{row}), Pointer.to(new int[]{column}));
            int blockSizeX = (int) Math.min(row, Math.pow(BLOCK_SIZE, (double) 1 / 2));
            int blockSizeY = (int) Math.min(column, Math.pow(BLOCK_SIZE, (double) 1 / 2));
            gridSizeX = (int) Math.ceil((double) row / blockSizeX);
            int gridSizeY = (int) Math.ceil((double) column / blockSizeY);

            cuLaunchKernel(function2,
                    gridSizeX, gridSizeY, 1,      // Grid dimension
                    blockSizeX, blockSizeY, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters2, null // Kernel- and extra parameters
            );
            if (Use.DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();

            JCuda.cudaFree(dVar_Pointer);

            errors.IsNan(errors);
            errorVar.IsNan(errorVar);
            input[n].IsNan(input[n]);
        }
    }

    private void derivativeWeight(NNMatrix error, int n) {
        if (Use.CPU) {
            int index = 0;
            for (int j = 0; j < width; j++) {
                for (int k = 0; k < depth; k++, index++) {
                    derBetta.getData()[k] += error.getData()[index];
                    derGamma.getData()[k] += error.getData()[index] * ((output[n].getData()[index] - betta.get(k)) / gamma.get(k));
                    //derGamma.getData()[k] += error.getData()[index] * ((output[n].getData()[index] - mean[n].getData()[j]) * var[n].getData()[j]);
                }
            }
        }
        if (Use.GPU) {
            int row = width;
            int column = depth;
            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, helperModule, "derivativeWeight_2");
            Pointer kernelParameters = Pointer.to(Pointer.to(error.getData_gpu()), Pointer.to(output[n].getData_gpu()), Pointer.to(betta.getData_gpu()), Pointer.to(gamma.getData_gpu()), Pointer.to(derBetta.getData_gpu()), Pointer.to(derGamma.getData_gpu()), Pointer.to(new int[]{row}), Pointer.to(new int[]{column}));
            int blockSizeX = (int) Math.min(column, BLOCK_SIZE);
            int gridSizeX = (int) Math.ceil((double) column / blockSizeX);

            cuLaunchKernel(function,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSizeX, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
            if (Use.DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
        }
    }

    public NormalizationLayer2D setTrainable(boolean trainable) {
        this.trainable = trainable;

        return this;
    }

    public NormalizationLayer2D setRegularization(Regularization regularization) {
        this.regularization = regularization;

        return this;
    }

    @Override
    public int info() {
        int countParam = betta.size() * 2;
        System.out.println("Layer norm\t| " + width + ",\t" + depth + "\t\t| " + outWidth + ",\t" + outDepth + "\t\t|\t" + countParam);

        return countParam;
    }

    @Override
    public void save(FileWriter writer) throws IOException {
        writer.write("Normalization layer 2D\n");
        gamma.save(writer);
        betta.save(writer);

        if (regularization != null) {
            regularization.write(writer);
        } else {
            writer.write("null\n");
        }
        writer.write(trainable + "\n");
        writer.flush();
    }

    public static NormalizationLayer2D read(Scanner scanner) {
        NormalizationLayer2D layer = new NormalizationLayer2D();
        layer.loadWeight = false;
        layer.gamma = NNVector.read(scanner);
        layer.betta = NNVector.read(scanner);
        layer.setRegularization(Regularization.read(scanner));
        layer.setTrainable(Boolean.parseBoolean(scanner.nextLine()));
        layer.loadWeight = true;
        return layer;
    }
}

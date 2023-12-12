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

    private NNVector[] mean, var;

    public NormalizationLayer2D(boolean TYPE) {
        this.trainable = true;
        this.epsilon = 0.00000001f;
        this.TYPE = TYPE;
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

        derBetta = new NNVector(depth, TYPE);
        derGamma = new NNVector(depth, TYPE);

        if (!loadWeight) {
            betta = new NNVector(depth, TYPE);
            gamma = new NNVector(depth, TYPE);

            gamma.fill(1.0f);
            //.fill(0.01f);
        }
    }

    @Override
    public void initialize(Optimizer optimizer) {
        optimizer.addDataOptimize(betta, derBetta, "Normalization layer 2D");
        optimizer.addDataOptimize(gamma, derGamma, "Normalization layer 2D");
    }

    @Override
    public void generateOutput(NNArray[] input) {
        this.input = NNArrays.isMatrix(input);
        this.output = new NNMatrix[input.length];
        this.mean = new NNVector[input.length];
        this.var = new NNVector[input.length];

        for (int i = 0; i < input.length; i++) {
            output[i] = new NNMatrix(outWidth, outDepth, TYPE);
            mean[i] = new NNVector(width, TYPE);
            var[i] = new NNVector(width, TYPE);
        }

        if (Use.CPU) {
            GPU_Sleep();
            ExecutorService executor = Executors.newFixedThreadPool(input.length);
            for (int t = 0; t < input.length; t++) {
                final int i = t;
                executor.execute(() -> {
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
            NormalizationLayerForward2D();
        }
    }

    private void NormalizationLayerForward2D()
    {
        int numSteps = input.length;
        Pointer[][] P = new Pointer[4][numSteps];

        for(int k = 0; k < numSteps; k++)
        {
            P[0][k] = input[k].getData_gpu();
            P[1][k] = mean[k].getData_gpu();
            P[2][k] = var[k].getData_gpu();
            P[3][k] = output[k].getData_gpu();
        }

        Pointer[] Array = new Pointer[P.length];
        for(int k = 0; k < P.length; k++) {
            Array[k] = new Pointer();
            cudaMalloc(Array[k], (long) numSteps * Sizeof.POINTER);
            cudaMemcpy(Array[k], Pointer.to(P[k]), (long) numSteps * Sizeof.POINTER, cudaMemcpyHostToDevice);
        }

        Pointer PArray = new Pointer();
        cudaMalloc(PArray, (long) P.length * Sizeof.POINTER);
        cudaMemcpy(PArray, Pointer.to(Array), (long) P.length * Sizeof.POINTER, cudaMemcpyHostToDevice);

        CUfunction function = new CUfunction();
        if (!TYPE) {
            cuModuleGetFunction(function, helperModule, "NormalizationLayerForward2D");
        }
        else
        {
            cuModuleGetFunction(function, helperModule, "NormalizationLayerForward_TYPE_2D");
        }
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
        JCuda.cudaFree(PArray);
        for(int k = 0; k < P.length; k++) {
            JCuda.cudaFree(Array[k]);
        }

        if (Use.DEBUG_SYNC) {
            JCudaDriver.cuCtxSynchronize();
            for(int k = 0; k < numSteps; k++) {
                input[k].IsNan_float();
                mean[k].IsNan_float();
                var[k].IsNan_float();
                output[k].IsNan_float();
            }
        }
    }

    private void NormalizationLayerBackward2D()
    {
        int numSteps = input.length;
        Pointer[][] P = new Pointer[6][numSteps];

        for(int k = 0; k < numSteps; k++)
        {
            P[0][k] = errorNL[k].getData_gpu();
            P[1][k] = var[k].getData_gpu();
            P[2][k] = input[k].getData_gpu();
            P[3][k] = mean[k].getData_gpu();
            P[4][k] = error[k].getData_gpu();
            P[5][k] = output[k].getData_gpu();
        }

        Pointer[] Array = new Pointer[P.length];
        for(int k = 0; k < P.length; k++) {
            Array[k] = new Pointer();
            cudaMalloc(Array[k], (long) numSteps * Sizeof.POINTER);
            cudaMemcpy(Array[k], Pointer.to(P[k]), (long) numSteps * Sizeof.POINTER, cudaMemcpyHostToDevice);
        }

        Pointer PArray = new Pointer();
        cudaMalloc(PArray, (long) P.length * Sizeof.POINTER);
        cudaMemcpy(PArray, Pointer.to(Array), (long) P.length * Sizeof.POINTER, cudaMemcpyHostToDevice);

        CUfunction function = new CUfunction();
        if (!TYPE) {
            cuModuleGetFunction(function, helperModule, "NormalizationLayerBackward2D");
        }
        else
        {
            cuModuleGetFunction(function, helperModule, "NormalizationLayerBackward_TYPE_2D");
        }
        Pointer kernelParameters = Pointer.to(Pointer.to(PArray), Pointer.to(gamma.getData_gpu()), Pointer.to(betta.getData_gpu()), Pointer.to(derGamma.getData_gpu()), Pointer.to(derBetta.getData_gpu()), Pointer.to(new int[]{outWidth}), Pointer.to(new int[]{outDepth}), Pointer.to(new int[]{width}), Pointer.to(new int[]{depth}), Pointer.to(new int[]{numSteps}));
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
        JCuda.cudaFree(PArray);
        for(int k = 0; k < P.length; k++) {
            JCuda.cudaFree(Array[k]);
        }

        if (Use.DEBUG_SYNC) {
            JCudaDriver.cuCtxSynchronize();

            for(int k = 0; k < numSteps; k++) {
                if (!TYPE) {
                    errorNL[k].IsNan_float();
                    var[k].IsNan_float();
                    input[k].IsNan_float();
                    mean[k].IsNan_float();
                    error[k].IsNan_float();
                    output[k].IsNan_float();
                }
                else
                {
                    errorNL[k].IsNan();
                    var[k].IsNan();
                    input[k].IsNan();
                    mean[k].IsNan();
                    error[k].IsNan();
                    output[k].IsNan();
                }
            }
        }
    }

    //среднее_mean = X.sum () / n
    //std = ((( X-mean)** 2 ) .sum () / n).sqrt()
    //z_scores = (X - среднее_mean) / std

    private void normalization(int n) {
        float[] varSqrt = new float[var[n].size()];
        for (int i = 0; i < var[n].size(); i++) {
            varSqrt[i] = (float) (Math.sqrt(var[n].getData()[i] + epsilon));
        }
        int index = 0;
        for (int j = 0; j < width; j++) {
            for (int k = 0; k < depth; k++, index++) {
                output[n].getData()[index] = ((input[n].getData()[index] - mean[n].get(j)) / varSqrt[j]) * gamma.get(k) + betta.get(k);
            }
        }
    }

    //საშუალო მნიშვნელობის გამოთვლა
    private void findMean(int n) {
        int index = 0;
        for (int j = 0; j < width; j++) {
            for (int k = 0; k < depth; k++, index++) {
                mean[n].getData()[j] += input[n].getData()[index];
            }
        }
        mean[n].div(depth);
    }

    //среднее_mean = X.sum () / n
    //std = ((( X-mean)** 2 ) .sum () / n).sqrt()
    //z_scores = (X - среднее_mean) / std

    private void findVariance(int n) {
        float sub;
        int index = 0;
        for (int j = 0; j < width; j++) {
            for (int k = 0; k < depth; k++, index++) {
                sub = (float)(input[n].getData()[index] - mean[n].getData()[j]);
                var[n].getData()[j] += sub * sub;
            }
        }
        var[n].div(depth);
    }

    @Override
    public void generateError(NNArray[] errors) {
        errorNL = getErrorNextLayer(errors);
        this.error = new NNMatrix[errors.length];

        for (int i = 0; i < input.length; i++) {
            error[i] = new NNMatrix(outWidth, outDepth, TYPE);
        }

        if (Use.CPU) {
            GPU_Sleep();
            ExecutorService executor = Executors.newFixedThreadPool(input.length);
            for (int t = 0; t < input.length; t++) {
                final int i = t;
                executor.execute(() -> {
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
            NormalizationLayerBackward2D();
        }

        if (trainable && regularization != null) {
            regularization.regularization(betta);
            regularization.regularization(gamma);
        }
    }

    private NNMatrix generateErrorNorm(int n) {
        NNMatrix errorNorm = new NNMatrix(outWidth, outDepth, TYPE);

        int index = 0;
        for (int j = 0; j < width; j++) {
            for (int k = 0; k < depth; k++, index++) {
                errorNorm.getData()[index] = errorNL[n].getData()[index] * gamma.get(k);
            }
        }

        return errorNorm;
    }

    private NNVector derVar(NNMatrix error, int n) {
        NNVector derVariance = new NNVector(var[n].size(), TYPE);

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

        return derVariance;
    }

    private NNVector derMean(NNMatrix error, NNVector derVar, int n) {
        NNVector derMean = new NNVector(mean[n].size(), TYPE);

        float[] dMean = new float[mean[n].size()];
        float[] dVar = new float[var[n].size()];

        for (int i = 0; i < var[n].size(); i++) {
            dMean[i] = (float) (-1.0f / Math.sqrt(var[n].get(i) + epsilon));
        }

        int index = 0;
        for (int j = 0; j < width; j++) {
            for (int k = 0; k < depth; k++, index++) {
                derMean.getData()[j] += error.get(index);
                dVar[j] += input[n].get(index) - mean[n].get(j);
            }
        }

        for (int i = 0; i < derMean.size(); i++) {
            derMean.getData()[i] *= dMean[i];
            derMean.getData()[i] += ((-2.0f * derVar.get(i) * dVar[i]) / (depth));
        }
        return derMean;
    }

    private void derNorm(NNMatrix errors, NNVector errorMean, NNVector errorVar, int n) {
        errorMean.div(depth);
        errorVar.mul(2.0f / (depth));

        float[] dVar = new float[var[n].size()];
        for (int i = 0; i < var[n].size(); i++) {
            dVar[i] = (float) (1.0f / Math.sqrt(var[n].getData()[i] + epsilon));
        }

        int index = 0;
        for (int j = 0; j < width; j++) {
            for (int k = 0; k < depth; k++, index++) {
                error[n].getData()[index] = errors.getData()[index] * dVar[j] + errorVar.get(j) *
                        ((float)(input[n].get(index) - mean[n].get(j))) + errorMean.get(j);
            }
        }
    }

    private void derivativeWeight(NNMatrix error, int n) {
        int index = 0;
        for (int j = 0; j < width; j++) {
            for (int k = 0; k < depth; k++, index++) {
                derBetta.getData()[k] += error.getData()[index];
                derGamma.getData()[k] += error.getData()[index] * ((output[n].getData()[index] - betta.get(k)) / gamma.get(k));
            }
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
        writer.write(this.TYPE + "\n");
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
        NormalizationLayer2D layer = new NormalizationLayer2D(Boolean.parseBoolean(scanner.nextLine()));
        layer.loadWeight = false;
        layer.gamma = NNVector.read(scanner);
        layer.betta = NNVector.read(scanner);
        layer.setRegularization(Regularization.read(scanner));
        layer.setTrainable(Boolean.parseBoolean(scanner.nextLine()));
        layer.loadWeight = true;

        //layer.gamma = new NNVector(layer.gamma.size());
        //layer.gamma.fill(0.01f);

        return layer;
    }
}

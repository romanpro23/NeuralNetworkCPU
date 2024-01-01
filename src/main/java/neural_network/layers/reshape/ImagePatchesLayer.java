package neural_network.layers.reshape;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUfunction;
import jcuda.driver.JCudaDriver;
import jcuda.runtime.JCuda;
import lombok.Setter;
import neural_network.initialization.Initializer;
import neural_network.layers.NeuralLayer;
import neural_network.optimizers.Optimizer;
import neural_network.regularization.Regularization;
import nnarrays.*;
import utilities.Use;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Scanner;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.runtime.JCuda.cudaMalloc;
import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;
import static nnarrays.NNArray.BLOCK_SIZE;
import static utilities.GPUInit.helperModule;
import static utilities.JCudaHelper.CONTEXT;
import static utilities.Use.*;

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

        if ((Use.CPU) && (!Use.GPU)) {
            GPU_Sleep();
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
            GPU_WakeUp();
        }

        if (Use.GPU) {
            for (int i = 0; i < inputs.length; i++) {
                patches[i] = input[i].imageVector(sizeKernel);
                output[i] = patches[i].dot(weight);
            }

            //ImagePatchesLayerForward();
        }
    }

    public NNMatrix dotT(NNMatrix a, NNMatrix b, int row1, int col1, int row2, int col2)
    {
        NNMatrix results = new NNMatrix(row1, col2);
        for (int i = 0; i < row1; i++) {
            for (int j = 0; j < col2; j++) {
                float sum = 0;
                for (int k = 0; k < col1; k++)
                    sum = sum + a.getData()[i * col1 + k] * b.getData()[k * col2 + j];
                results.getData()[i * col2 + j] = sum;
            }
        }
        return results;
    }

    public NNMatrix transpose(NNMatrix data, int row, int col)
    {
        NNMatrix results = new NNMatrix(col, row);
        int index;
        for (int i = 0; i < row; i++) {
            index = i * col;
            for (int j = 0; j < col; j++, index++) {
                results.getData()[i + j * col] = data.getData()[index];
            }
        }
        return results;
    }

    private void ImagePatchesLayerForward()
    {
        int numSteps = this.input.length;
        Pointer[][] P = new Pointer[3][numSteps];

        int row = this.input[0].getRows();
        int col = this.input[0].getColumns();
        int depth = this.input[0].getDepth();

        int patch_row = (row / sizeKernel) * (col / sizeKernel);
        int patch_col = sizeKernel * sizeKernel * depth;

        for(int k = 0; k < numSteps; k++)
        {
            this.patches[k] = new NNMatrix(patch_row, patch_col);
            this.output[k] = new NNMatrix(row, weight.getColumn());
            P[0][k] = this.input[k].getData_gpu();
            P[1][k] = this.patches[k].getData_gpu();
            P[2][k] = this.output[k].getData_gpu();
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
        cuModuleGetFunction(function, helperModule, "ImagePatchesLayerForward");
        Pointer kernelParameters = Pointer.to(Pointer.to(PArray), Pointer.to(weight.getData_gpu()), Pointer.to(new int[]{row}), Pointer.to(new int[]{col}), Pointer.to(new int[]{depth}), Pointer.to(new int[]{patch_row}), Pointer.to(new int[]{patch_col}), Pointer.to(new int[]{weight.getRow()}), Pointer.to(new int[]{weight.getColumn()}), Pointer.to(new int[]{sizeKernel}), Pointer.to(new int[]{numSteps}));
        int blockSizeX = (int) Math.min(numSteps, Math.pow(BLOCK_SIZE, (double) 1 / 2));
        int blockSizeY = (int) Math.min(row, Math.pow(BLOCK_SIZE, (double) 1 / 2));
        int gridSizeX = (int) Math.ceil((double) numSteps / blockSizeX);
        int gridSizeY = (int) Math.ceil((double) row / blockSizeY);

        cuLaunchKernel(function,
                gridSizeX, 1, 1,      // Grid dimension
                blockSizeX, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                kernelParameters, null // Kernel- and extra parameters
        );
        if (Use.DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();

        JCuda.cudaFree(PArray);
        for(int k = 0; k < P.length; k++) {
            JCuda.cudaFree(Array[k]);
        }
        if (Use.DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
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

        if ((Use.CPU) && (!Use.GPU)) {
            GPU_Sleep();
            ExecutorService executor = Executors.newFixedThreadPool(input.length);
            for (int t = 0; t < input.length; t++) {
                final int i = t;
                executor.execute(() -> {
                    if (returnGradient) {
                        NNMatrix errorImg = errorNL[i].dotT(weight);
                        error[i] = input[i].backImageVector(errorImg, sizeKernel);
                    }
                    if (trainable) {
                        derWeight.add(patches[i].transpose().dot(errorNL[i]));
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
                if (returnGradient) {
                    NNMatrix errorImg = errorNL[i].dotT(weight);
                    error[i] = input[i].backImageVector(errorImg, sizeKernel);
                }
                if (trainable) {
                    derWeight.add(patches[i].transpose().dot(errorNL[i]));
                }
            }
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

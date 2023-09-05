package nnarrays;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUfunction;
import jcuda.driver.JCudaDriver;
import jcuda.jcublas.JCublas2;
import jcuda.runtime.JCuda;
import lombok.SneakyThrows;
import utilities.Use;

import java.util.Arrays;

import static java.lang.Math.log;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.jcublas.JCublas2.cublasSasum;
import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;
import static nnarrays.NNArray.BLOCK_SIZE;
import static utilities.GPUInit.cublasHandle;
import static utilities.GPUInit.helperModule;

public final class NNArrays {

    public static NNVector[] isVector(NNArray[] batch) {
        return (NNVector[]) batch;
    }

    public static NNVector[] toVector(NNArray[] batch) {
        NNVector[] arr = new NNVector[batch.length];
        for (int i = 0; i < arr.length; i++) {
            if (!Use.GPU) {
                arr[i] = new NNVector(batch[i]);
            }
            else
            {
                arr[i] = new NNVector(batch[i].size);
                arr[i].copy(batch[i]);
            }
        }

        return arr;
    }

    public static NNArray[] empty() {
        return null;
    }

    public static NNMatrix[] empty(NNMatrix[] out) {
        NNMatrix[] res = new NNMatrix[out.length];

        for (int i = 0; i < res.length; i++) {
            res[i] = new NNMatrix(out[i]);
        }
        return res;
    }

    public static NNVector[] subVectors(NNVector[] input, int start, int size) {
        NNVector[] output = new NNVector[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = input[i].subVector(start, size);
        }

        return output;
    }

    public static NNMatrix[] toHotVector(NNArray[] batch, int sizeVoc) {
        NNMatrix[] arr = new NNMatrix[batch.length];
        for (int i = 0; i < arr.length; i++) {
            arr[i] = new NNMatrix(batch[i].size, sizeVoc);
            for (int j = 0; j < batch[i].size; j++) {
                arr[i].set(j, (int) batch[i].data[j], 1);
            }
        }

        return arr;
    }

    public static NNTensor[] toTensor(NNArray[] batch, int[] size) {
        return toTensor(batch, size[0], size[1], size[2]);
    }

    public static NNTensor[] toTensor(NNArray[] batch, int depth, int height, int width) {
        NNTensor[] arr = new NNTensor[batch.length];
        for (int i = 0; i < arr.length; i++) {
            arr[i] = new NNTensor(depth, height, width, batch[i].data);
        }

        return arr;
    }

    public static NNMatrix[] toMatrix(NNArray[] batch, int height, int width) {
        NNMatrix[] arr = new NNMatrix[batch.length];
        for (int i = 0; i < arr.length; i++) {
            arr[i] = new NNMatrix(height, width, batch[i].data);
        }

        return arr;
    }

    public static NNArray[] concat(NNArray[] first, NNArray[] second) {
        if (first[0].countAxes == 1) {
            return concatVector(first, second);
        } else if (first[0].countAxes == 2) {
            return concatMatrix(first, second);
        } else if (first[0].countAxes == 3) {
            return concatTensor(first, second);
        }
        return null;
    }

    public static NNArray[] subArray(NNArray[] first, NNArray[] second) {
        return subArray(first, second, 0);
    }

    public static NNArray[] subArray(NNArray[] first, NNArray[] second, int startIndex) {
        if (first[0].countAxes == 1) {
            return subVector(first, second, startIndex);
        } else if (first[0].countAxes == 2) {
            return subMatrix(first, second, startIndex);
        } else if (first[0].countAxes == 3) {
            return subTensor(first, second, startIndex);
        }
        return null;
    }

    @SneakyThrows
    public static NNVector[] concatVector(NNArray[] first, NNArray[] second) {
        if (first.length != second.length) {
            throw new Exception("Vector has difference size");
        }
        NNVector[] result = new NNVector[first.length];
        for (int i = 0; i < first.length; i++) {
            float[] data = new float[first[i].size + second[i].size];

            System.arraycopy(first[i].data, 0, data, 0, first[i].size);
            System.arraycopy(second[i].data, 0, data, first[i].size, second[i].size);
            result[i] = new NNVector(data);
        }
        return result;
    }

    @SneakyThrows
    public static NNVector[] subVector(NNArray[] first, NNArray[] second, int startIndex) {
        NNVector[] result = new NNVector[first.length];
        for (int i = 0; i < first.length; i++) {
            float[] data = new float[second[i].size];

            System.arraycopy(first[i].data, startIndex, data, 0, second[i].size);
            result[i] = new NNVector(data);
        }
        return result;
    }

    @SneakyThrows
    public static NNMatrix[] subMatrix(NNArray[] first, NNArray[] second, int startIndex) {
        NNMatrix[] result = new NNMatrix[first.length];
        for (int i = 0; i < first.length; i++) {
            float[] data = new float[second[i].size];

            int size = second[i].shape()[0];
            int startDepth = startIndex / size;
            int depth = second[i].shape()[0];
            int column = second[i].shape()[1];

            int index = 0, indexF;

            for (int j = 0; j < size; j++) {
                indexF = j * first[i].shape()[1] + startDepth;
                for (int k = 0; k < column; k++, index++, indexF++) {
                    data[index] = first[i].data[indexF];
                }
            }

            result[i] = new NNMatrix(depth, column, data);
        }
        return result;
    }

    @SneakyThrows
    public static NNMatrix[] concatMatrix(NNArray[] first, NNArray[] second) {
        if (first.length != second.length) {
            throw new Exception("Vector has difference size");
        }
        NNMatrix[] result = new NNMatrix[first.length];
        for (int i = 0; i < first.length; i++) {
            float[] data = new float[first[i].size + second[i].size];

            int size = first[i].shape()[0];
            int columnF = first[i].shape()[1], columnS = second[i].shape()[1];
            int column = columnF + columnS;
            int index = 0;
            int indexF = 0;
            int indexS = 0;

            for (int j = 0; j < size; j++) {
                for (int k = 0; k < columnF; k++, index++, indexF++) {
                    data[index] = first[i].data[indexF];
                }
                for (int k = 0; k < columnS; k++, index++, indexS++) {
                    data[index] = second[i].data[indexS];
                }
            }

            int row = first[i].shape()[0];
            result[i] = new NNMatrix(row, column, data);
        }
        return result;
    }

    @SneakyThrows
    public static NNTensor[] subTensor(NNArray[] first, NNArray[] second, int startIndex) {
        NNTensor[] result = new NNTensor[first.length];
        for (int i = 0; i < first.length; i++) {
            float[] data = new float[second[i].size];

            int size = second[i].shape()[0] * second[i].shape()[1];
            int startDepth = startIndex / size;
            int depth = second[i].shape()[0];
            int row = second[i].shape()[1];
            int column = second[i].shape()[2];

            int index = 0, indexF;

            for (int j = 0; j < size; j++) {
                indexF = j * first[i].shape()[2] + startDepth;
                for (int k = 0; k < column; k++, index++, indexF++) {
                    data[index] = first[i].data[indexF];
                }
            }

            result[i] = new NNTensor(depth, row, column, data);
        }
        return result;
    }

    @SneakyThrows
    public static void subTensor(NNArray[] tensor, NNArray[] subTensor, int x0, int y0) {
        NNTensor[] tensors = NNArrays.isTensor(tensor);
        NNTensor[] subTensors = NNArrays.isTensor(subTensor);
        for (int i = 0; i < tensors.length; i++) {
            for (int x = x0, h = 0; x < subTensors[i].getRows() + x0; x++, h++) {
                for (int y = y0, w = 0; y < subTensors[i].getColumns() + y0; y++, w++) {
                    int indexT = tensors[i].getRowsIndex()[x] + tensors[i].getColumnsIndex()[y];
                    int indexST = subTensors[i].getRowsIndex()[h] + subTensors[i].getColumnsIndex()[w];
                    System.arraycopy(tensors[i].getData(), indexT, subTensors[i].getData(), indexST, subTensors[i].getDepth());
                }
            }
        }
    }

    public static NNMatrix[] reverse(NNMatrix[] input) {
        NNMatrix[] result = new NNMatrix[input.length];

        for (int i = 0; i < input.length; i++) {
            result[i] = new NNMatrix(input[i]);
            for (int j = 0; j < input[i].getRow(); j++) {
                for (int k = 0, l = input[i].getColumn() - 1; k < input[i].getColumn(); k++, l--) {
                    result[i].set(j, l, input[i].get(j, k));
                }
            }
        }

        return result;
    }

    @SneakyThrows
    public static void addSubTensor(NNArray[] tensor, NNArray[] subTensor, int x0, int y0) {
        NNTensor[] tensors = NNArrays.isTensor(tensor);
        NNTensor[] subTensors = NNArrays.isTensor(subTensor);
        for (int i = 0; i < tensors.length; i++) {
            for (int x = x0, h = 0; x < subTensors[i].getRows() + x0; x++, h++) {
                for (int y = y0, w = 0; y < subTensors[i].getColumns() + y0; y++, w++) {
                    int indexT = tensors[i].getRowsIndex()[x] + tensors[i].getColumnsIndex()[y];
                    int indexST = subTensors[i].getRowsIndex()[h] + subTensors[i].getColumnsIndex()[w];
                    System.arraycopy(subTensors[i].getData(), indexST, tensors[i].getData(), indexT, subTensors[i].getDepth());
                }
            }
        }
    }

    @SneakyThrows
    public static NNTensor[] concatTensor(NNArray[] first, NNArray[] second) {
        if (first.length != second.length) {
            throw new Exception("Tensor has difference size");
        }
        NNTensor[] result = new NNTensor[first.length];
        for (int i = 0; i < first.length; i++) {
            float[] data = new float[first[i].size + second[i].size];

            int size = first[i].shape()[0] * first[i].shape()[1];
            int columnF = first[i].shape()[2], columnS = second[i].shape()[2];
            int column = columnF + columnS;
            int index = 0;
            int indexF = 0;
            int indexS = 0;

            for (int j = 0; j < size; j++) {
                for (int k = 0; k < columnF; k++, index++, indexF++) {
                    data[index] = first[i].data[indexF];
                }
                for (int k = 0; k < columnS; k++, index++, indexS++) {
                    data[index] = second[i].data[indexS];
                }
            }

            int depth = first[i].shape()[0];
            int row = first[i].shape()[1];
            result[i] = new NNTensor(depth, row, column, data);
        }
        return result;
    }

    public static void mul(NNArray[] arrays, float lambda){
        for (NNArray array : arrays) {
            array.mul(lambda);
        }
    }

    public static NNMatrix mul(NNVector[] first, NNVector[] second) {
        NNMatrix result = new NNMatrix(first[0].size, second[0].size);

        for (int i = 0; i < first.length; i++) {
            for (int j = 0, index = 0; j < first[i].size(); j++) {
                for (int k = 0; k < second[i].size(); k++, index++) {
                    result.data[index] += first[i].data[j] * second[i].data[k];
                }
            }
        }

        return result;
    }

    public static NNMatrix[] isMatrix(NNArray[] batch) {
        return (NNMatrix[]) batch;
    }

    public static NNTensor[] isTensor(NNArray[] batch) {
        return (NNTensor[]) batch;
    }

    public static NNTensor4D[] isTensor4D(NNArray[] batch) {
        return (NNTensor4D[]) batch;
    }

    public static float sum(NNArray array) {
        float sum = 0;
        if (!Use.GPU) {
            for (int i = 0; i < array.size; i++) {
                sum += array.data[i];
            }
        }
        else
        {
            float[] sumArray = new float[1];
            Pointer sum_gpu = new Pointer();
            cublasSasum(cublasHandle, array.size, array.data_gpu, 1, sum_gpu);

            /*cudaMalloc(sum_gpu, (long) Sizeof.FLOAT);

            int n = array.size;
            CUfunction function = new CUfunction();
            cu
            cuModuleGetFunction(function, helperModule, "sum");
            Pointer kernelParameters = Pointer.to(Pointer.to(array.data_gpu), Pointer.to(sum_gpu), Pointer.to(new int[]{n}));
            int blockSize = Math.min(n, BLOCK_SIZE);
            int gridSizeX = (int) Math.ceil((double) n / blockSize);
            cuLaunchKernel(function,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSize, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );*/
            JCublas2.cublasGetVector(sumArray.length, Sizeof.FLOAT, sum_gpu, 1, Pointer.to(sumArray), 1);
            sum = sumArray[0];
            if (Use.DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
            JCuda.cudaFree(sum_gpu);
        }

        return sum;
    }

    @SneakyThrows
    public static NNArray sub(NNArray first, NNArray second) {
        if (first.size != second.size) {
            throw new Exception("Vector has difference size");
        }
        NNArray result = new NNArray(first.size);

        if (!Use.GPU) {
            for (int i = 0; i < result.size; i++) {
                result.data[i] = first.data[i] - second.data[i];
            }
        }
        else
        {
            int n = result.size;
            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, helperModule, "sub");
            Pointer kernelParameters = Pointer.to(Pointer.to(first.data_gpu), Pointer.to(second.data_gpu), Pointer.to(result.data_gpu), Pointer.to(new int[]{n}));
            int blockSize = Math.min(n, BLOCK_SIZE);
            int gridSizeX = (int) Math.ceil((double) n / blockSize);
            cuLaunchKernel(function,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSize, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
            if (Use.DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
        }

        return result;
    }

    @SneakyThrows
    public static NNArray capsLoss(NNArray first, NNArray second) {
        if (first.size != second.size) {
            throw new Exception("Vector has difference size");
        }
        NNArray result = new NNArray(first.size);
        for (int i = 0; i < result.size; i++) {
            result.data[i] = (float) (first.data[i] * Math.pow(Math.max(0, 0.9f - second.data[i]), 2) +
                    0.5f * (1f - first.data[i]) * Math.pow(Math.max(0, second.data[i] - 0.1f), 2));
        }

        return result;
    }

    @SneakyThrows
    public static NNArray derCapsLoss(NNArray first, NNArray second) {
        if (first.size != second.size) {
            throw new Exception("Vector has difference size");
        }
        NNArray result = new NNArray(first.size);
        for (int i = 0; i < result.size; i++) {
            result.data[i] = first.data[i] * -2 * Math.max(0, 0.9f - second.data[i])
                    + (1f - first.data[i]) * Math.max(0, second.data[i] - 0.1f);
        }

        return result;
    }

    @SneakyThrows
    public static NNArray crossEntropy(NNArray first, NNArray second) {
        if (first.size != second.size) {
            throw new Exception("Vector has difference size");
        }
        NNArray result = new NNArray(first.size);
        for (int i = 0; i < result.size; i++) {
            result.data[i] = (float) (first.data[i] * log(second.data[i] + 0.00000001f));
        }

        return result;
    }

    @SneakyThrows
    public static NNArray poisson(NNArray first, NNArray second) {
        if (first.size != second.size) {
            throw new Exception("Vector has difference size");
        }
        NNArray result = new NNArray(first.size);
        for (int i = 0; i < result.size; i++) {
            result.data[i] = (float) (second.data[i] - first.data[i] * log(second.data[i] + 0.00000001f));
        }

        return result;
    }

    @SneakyThrows
    public static NNArray klDivergence(NNArray first, NNArray second) {
        if (first.size != second.size) {
            throw new Exception("Vector has difference size");
        }
        NNArray result = new NNArray(first.size);
        for (int i = 0; i < result.size; i++) {
            result.data[i] = (float) (first.data[i] * log(first.data[i] / (second.data[i] + 0.00000001f) + 0.00000001f));
        }

        return result;
    }

    @SneakyThrows
    public static NNArray hinge(NNArray first, NNArray second) {
        if (first.size != second.size) {
            throw new Exception("Vector has difference size");
        }
        NNArray result = new NNArray(first.size);
        for (int i = 0; i < result.size; i++) {
            result.data[i] = Math.max(1.0f - first.data[i] * second.data[i], 0);
        }

        return result;
    }

    @SneakyThrows
    public static NNArray logCosh(NNArray first, NNArray second) {
        if (first.size != second.size) {
            throw new Exception("Vector has difference size");
        }
        NNArray result = new NNArray(first.size);
        for (int i = 0; i < result.size; i++) {
            result.data[i] = (float) log(Math.cosh(second.data[i] - first.data[i]));
        }

        return result;
    }

    @SneakyThrows
    public static NNArray derLogCosh(NNArray first, NNArray second) {
        if (first.size != second.size) {
            throw new Exception("Vector has difference size");
        }
        NNArray result = new NNArray(first.size);
        for (int i = 0; i < result.size; i++) {
            result.data[i] = (float) (Math.tanh(second.data[i] - first.data[i]));
        }

        return result;
    }

    @SneakyThrows
    public static NNArray derHinge(NNArray first, NNArray second) {
        if (first.size != second.size) {
            throw new Exception("Vector has difference size");
        }
        NNArray result = new NNArray(first.size);
        for (int i = 0; i < result.size; i++) {
            if (1.0f - first.data[i] * second.data[i] > 0) {
                result.data[i] = -first.data[i];
            }
        }

        return result;
    }

    @SneakyThrows
    public static NNArray subAbs(NNArray first, NNArray second) {
        if (first.size != second.size) {
            throw new Exception("Vector has difference size");
        }
        NNArray result = new NNArray(first.size);
        for (int i = 0; i < result.size; i++) {
            result.data[i] = Math.abs(first.data[i] - second.data[i]);
        }

        return result;
    }

    @SneakyThrows
    public static NNArray binaryCrossEntropy(NNArray first, NNArray second) {
        if (first.size != second.size) {
            throw new Exception("Vector has difference size");
        }
        NNArray result = new NNArray(first.size);
        for (int i = 0; i < result.size; i++) {
            result.data[i] = (float) (first.data[i] * log(second.data[i] + 0.00000001f) + (1 - first.data[i]) * log(1.0000001f - second.data[i]));
        }

        return result;
    }

    @SneakyThrows
    public static NNArray derAbs(NNArray first, NNArray second) {
        if (first.size != second.size) {
            throw new Exception("Vector has difference size");
        }
        NNArray result = new NNArray(first.size);
        for (int i = 0; i < result.size; i++) {
            float diff = first.data[i] - second.data[i];
            result.data[i] = diff / (Math.abs(diff) + 0.00000001f);
        }

        return result;
    }

    @SneakyThrows
    public static NNVector div(NNVector first, NNVector second) {
        if (first.size != second.size) {
            throw new Exception("Vector has difference size");
        }
        NNVector result = new NNVector(first.size);
        for (int i = 0; i < result.size; i++) {
            result.data[i] = first.data[i] / second.data[i];
        }

        return result;
    }

    @SneakyThrows
    public static NNArray derBinaryCrossEntropy(NNArray outputs, NNArray idealOutputs) {
        if (outputs.size != idealOutputs.size) {
            throw new Exception("Vector has difference size");
        }
        NNArray result = new NNArray(outputs.size);
        for (int i = 0; i < result.size; i++) {
            result.data[i] = (outputs.data[i] - idealOutputs.data[i]) / ((1 - outputs.data[i]) * outputs.data[i] + 0.00000001f);
        }

        return result;
    }

    @SneakyThrows
    public static NNArray derCrossEntropy(NNArray outputs, NNArray idealOutputs) {
        if (outputs.size != idealOutputs.size) {
            throw new Exception("Vector has difference size");
        }
        NNArray result = new NNArray(outputs.size);
        for (int i = 0; i < result.size; i++) {
            result.data[i] = -idealOutputs.data[i] / (outputs.data[i] + 0.00000001f);
        }

        return result;
    }

    @SneakyThrows
    public static NNArray derPoisson(NNArray outputs, NNArray idealOutputs) {
        if (outputs.size != idealOutputs.size) {
            throw new Exception("Vector has difference size");
        }
        NNArray result = new NNArray(outputs.size);
        for (int i = 0; i < result.size; i++) {
            result.data[i] = 1 - idealOutputs.data[i] / (outputs.data[i] + 0.00000001f);
        }

        return result;
    }

    public static NNMatrix[] create(NNMatrix[] input) {
        NNMatrix[] result = new NNMatrix[input.length];

        for (int i = 0; i < input.length; i++) {
            result[i] = new NNMatrix(input[i]);
        }

        return result;
    }

    public static NNVector[] create(NNVector[] input) {
        NNVector[] result = new NNVector[input.length];

        for (int i = 0; i < input.length; i++) {
            result[i] = new NNVector(input[i]);
        }

        return result;
    }

    public static void add(NNArray[] first, NNArray[] second) {
        for (int i = 0; i < second.length; i++) {
            first[i].add(second[i]);
        }
    }
}


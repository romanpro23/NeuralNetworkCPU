package nnarrays;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUfunction;
import jcuda.driver.JCudaDriver;
import jcuda.jcublas.JCublas2;
import jcuda.runtime.JCuda;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.SneakyThrows;
import utilities.Use;

import java.io.FileWriter;
import java.io.IOException;
import java.lang.ref.WeakReference;
import java.util.Arrays;
import java.util.Scanner;

import static java.lang.Math.pow;
import static java.lang.Math.signum;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;
import static utilities.GPUInit.*;
import static utilities.GPUInit.allocatedUse;

@NoArgsConstructor
public class NNArray {
    @Getter
    public float[] data;
    @Getter
    protected Pointer data_gpu;
    protected int size;
    @Getter
    protected int countAxes;
    public static int BLOCK_SIZE = 1024;
    public static int SharedMemorySizeGPU = 64 * 1024;

    public NNArray(int size) {
        this.size = size;

        if (Use.CPU) {
            this.data = new float[size];
        }

        if (Use.GPU) {
            this.data_gpu = new Pointer();
            cudaMalloc(this.data_gpu, (long) size * Sizeof.FLOAT);
            cudaMemset(this.data_gpu, 0, (long) size * Sizeof.FLOAT);

            allocatedPut();
        }
    }

    public int[] shape() {
        return new int[]{size};
    }

    public NNArray(float[] data) {
        this.size = data.length;

        if (Use.CPU) {
            this.data = data;
        }

        if (Use.GPU) {
            this.data_gpu = new Pointer();
            cudaMalloc(this.data_gpu, (long) Sizeof.FLOAT * this.size);
            cudaMemcpy(this.data_gpu, Pointer.to(data), (long) Sizeof.FLOAT * this.size, cudaMemcpyHostToDevice);

            allocatedPut();
        }
    }

    public NNArray(int[] data) {
        this.size = data.length;

        if (Use.CPU) {
            this.data = new float[size];
            for (int i = 0; i < data.length; i++) {
                this.data[i] = data[i];
            }
        }
        if (Use.GPU) {
            this.data_gpu = new Pointer();
            cudaMalloc(data_gpu, (long) Sizeof.INT * this.size);
            cudaMemcpy(data_gpu, Pointer.to(data), (long) Sizeof.INT * this.size, cudaMemcpyHostToDevice);

            allocatedPut();
        }
    }

    public void allocatedPut() {
        Use U = new Use();
        U.data_gpu = this.data_gpu;
        U.HashCode = this.hashCode();
        allocated.put(String.valueOf(this.hashCode()), new WeakReference<>(this));
        allocatedUse.put(String.valueOf(this.hashCode()), U);
    }

    public int size() {
        return size;
    }

    public void set(int i, float value) {
        if (Use.CPU) {
            data[i] = value;
        }
        /*if (Use.GPU) {
            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, helperModule, "set");
            Pointer kernelParameters = Pointer.to(Pointer.to(data_gpu), Pointer.to(new int[]{i}), Pointer.to(new float[]{value}));
            int blockSize = 1;
            int gridSizeX = 1;
            cuLaunchKernel(function,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSize, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
            //if (Use.DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
        }*/
    }

    public float get(int i) {
        return data[i];
    }

    public void div(float val) {
        if (Use.CPU) {
            for (int i = 0; i < size; i++) {
                data[i] /= val;
            }
        }

        if (Use.GPU) {
            int n = this.size();
            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, helperModule, "matrixDiv");
            Pointer kernelParameters = Pointer.to(Pointer.to(data_gpu), Pointer.to(new float[]{val}), Pointer.to(new int[]{n}));
            int blockSize = Math.min(n, BLOCK_SIZE);
            int gridSizeX = (int) Math.ceil((double) n / blockSize);
            cuLaunchKernel(function,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSize, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );

            if (Use.DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
            IsNan();
        }
    }

    public NNVector subVector(int startPos, int size) {
        NNVector result = new NNVector(size);
        System.arraycopy(data, startPos, result.data, 0, size);

        return result;
    }

    public NNArray pow2() {
        if (Use.CPU) {
            for (int i = 0; i < size; i++) {
                data[i] *= data[i];
            }
        }

        if (Use.GPU) {
            int n = size;
            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, helperModule, "pow2");
            Pointer kernelParameters = Pointer.to(Pointer.to(data_gpu), Pointer.to(new int[]{n}));
            int blockSize = Math.min(n, BLOCK_SIZE);
            int gridSizeX = (int) Math.ceil((double) n / blockSize);
            cuLaunchKernel(function,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSize, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
            if (Use.DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();

            IsNan();
        }

        return this;
    }

    public void clip(float val) {
        clip(-val, val);
    }

    public void clip(float min, float max) {
        float a;
        for (int i = 0; i < size; i++) {
            a = data[i];
            if (a > max) {
                data[i] = max;
            } else if (a < min) {
                data[i] = min;
            }
        }
    }

    public void sqrt() {
        for (int i = 0; i < size; i++) {
            data[i] = (float) Math.sqrt(data[i] + 0.00000001f);
        }
    }

    public NNArray mul(float val) {
        if (Use.CPU) {
            for (int i = 0; i < size; i++) {
                data[i] *= val;
            }
        }
        if (Use.GPU) {
            int n = size;
            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, helperModule, "mul");
            Pointer kernelParameters = Pointer.to(Pointer.to(data_gpu), Pointer.to(new float[]{val}), Pointer.to(new int[]{n}));
            int blockSize = Math.min(n, BLOCK_SIZE);
            int gridSizeX = (int) Math.ceil((double) n / blockSize);
            cuLaunchKernel(function,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSize, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
            if (Use.DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();

            IsNan();
        }

        return this;
    }

    public NNArray mul(NNArray array) {
        for (int i = 0; i < size; i++) {
            data[i] *= array.data[i];
        }

        return this;
    }

    public NNArray addMul(NNArray array, float val) {
        for (int i = 0; i < size; i++) {
            data[i] += array.data[i] * val;
        }

        return this;
    }

    public void clear() {
        if (Use.CPU) {
            for (int i = 0; i < size; i++) {
                data[i] = 0;
            }
        }
        if (Use.GPU) {
            cudaMemset(data_gpu, 0, (long) size * Sizeof.FLOAT);
        }
    }

    public void sub(float val) {
        add(-val);
    }

    @SneakyThrows
    public void sub(NNArray array) {
        if (size != array.size) {
            throw new Exception("Array has difference size");
        }

        for (int i = 0; i < size; i++) {
            data[i] -= array.data[i];
        }
    }

    @SneakyThrows
    public void copy(NNArray array) {
        if (size != array.size) {
            throw new Exception("Array has difference size");
        }

        if (Use.CPU) {
            System.arraycopy(array.data, 0, data, 0, size);
        }
        if (Use.GPU) {
            JCublas2.cublasScopy(cublasHandle, this.size, array.data_gpu, 1, this.data_gpu, 1);

            IsNan();
        }
    }

    @SneakyThrows
    public void add(NNArray array) {
        if (size != array.size) {
            throw new Exception("Array has difference size");
        }

        if (Use.CPU) {
            for (int i = 0; i < size; i++) {
                data[i] += array.data[i];
            }
        }
        if (Use.GPU) {
            MatAdd(array);
        }
    }

    public void MatAdd(NNArray matrix) {
        int n = size;
        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, helperModule, "MatAdd");
        Pointer kernelParameters = Pointer.to(Pointer.to(this.data_gpu), Pointer.to(matrix.data_gpu), Pointer.to(new int[]{n}));
        int blockSize = Math.min(n, BLOCK_SIZE);
        int gridSizeX = (int) Math.ceil((double) n / blockSize);
        cuLaunchKernel(function,
                gridSizeX, 1, 1,      // Grid dimension
                blockSize, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                kernelParameters, null // Kernel- and extra parameters
        );
        if (Use.DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();

        matrix.IsNan(matrix);
    }

    public void add(float val) {
        for (int i = 0; i < size; i++) {
            data[i] += val;
        }
    }

    public void oneSub() {
        for (int i = 0; i < size; i++) {
            data[i] = 1 - data[i];
        }
    }

    public void subSign(float val) {
        float a;
        for (int i = 0; i < size; i++) {
            a = data[i];
            if (a > 0) {
                data[i] -= val;
            } else if (a < 0) {
                data[i] += val;
            }
        }
    }

    public void fill(float value) {
        if (Use.CPU) {
            for (int i = 0; i < size; i++) {
                data[i] = value;
            }
        }

        if (Use.GPU) {
            int n = size;
            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, helperModule, "fill");
            Pointer kernelParameters = Pointer.to(Pointer.to(data_gpu), Pointer.to(new float[]{value}), Pointer.to(new int[]{n}));
            int blockSize = Math.min(n, BLOCK_SIZE);
            int gridSizeX = (int) Math.ceil((double) n / blockSize);
            cuLaunchKernel(function,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSize, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
            if (Use.DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();

            IsNan();
        }
    }

    public void relu(NNArray input) {
        for (int i = 0; i < size; i++) {
            data[i] = Math.max(0, input.data[i]);
        }
    }

    public void relu() {
        for (int i = 0; i < size; i++) {
            data[i] = Math.max(0, data[i]);
        }
    }

    public void gelu(NNArray input) {
        if (Use.CPU) {
            for (int i = 0; i < size; i++) {
                float x = input.data[i];
                data[i] = (float) (0.5f * x * (1f + Math.tanh(0.7978846f * x + 0.0356774f * Math.pow(x, 3))));
            }
        }

        if (Use.GPU) {
            int n = size;
            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, helperModule, "gelu");
            Pointer kernelParameters = Pointer.to(Pointer.to(input.data_gpu), Pointer.to(data_gpu), Pointer.to(new int[]{n}));
            int blockSize = Math.min(n, BLOCK_SIZE);
            int gridSizeX = (int) Math.ceil((double) n / blockSize);
            cuLaunchKernel(function,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSize, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
            if (Use.DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();

            IsNan();
        }
    }

    public void derGelu(NNArray input, NNArray error) {
        if (Use.CPU) {
            for (int i = 0; i < size; i++) {
                float x = input.data[i];
                float val = (float) Math.tanh(0.7978846f * x + 0.0356774f * Math.pow(x, 3));
                data[i] = error.data[i] * 0.5f * (1f + val + x * (1f - val * val) * (0.79788846f + 0.1070322f * x * x));
            }
        }

        if (Use.GPU) {
            int p = size;

            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, helperModule, "derGelu");
            Pointer kernelParameters = Pointer.to(Pointer.to(input.data_gpu), Pointer.to(error.data_gpu), Pointer.to(data_gpu), Pointer.to(new int[]{p}));
            int blockSize = Math.min(p, BLOCK_SIZE);
            int gridSizeX = (int) Math.ceil((double) p / blockSize);
            cuLaunchKernel(function,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSize, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
            if (Use.DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();

            IsNan();
        }
    }

    public void relu_max(NNArray input, float max) {
        for (int i = 0; i < size; i++) {
            data[i] = Math.min(max, Math.max(0, input.data[i]));
        }
    }

    public void randomrelu(NNArray input, NNArray alpha) {
        for (int i = 0; i < size; i++) {
            if (data[i] >= 0) {
                data[i] = input.data[i];
            } else {
                data[i] = input.data[i] * alpha.data[i];
            }
        }
    }

    public void fillRandom(float min, float max) {
        float sub = max - min;
        for (int i = 0; i < size; i++) {
            data[i] = (float) (Math.random() * sub + min);
        }
    }

    public void prelu(NNArray input, NNArray alpha) {
        for (int index = 0; index < size; index++) {
            if (input.data[index] < 0) {
                data[index] = input.data[index] * alpha.data[index];
            } else {
                data[index] = input.data[index];
            }
        }
    }

    public void derPrelu(NNArray input, NNArray error, NNArray alpha) {
        for (int index = 0; index < size; index++) {
            if (input.data[index] < 0) {
                data[index] = error.data[index] * alpha.data[index];
            } else {
                data[index] = error.data[index];
            }
        }
    }

    public void derRandomRelu(NNArray input, NNArray error, NNArray alpha) {
        for (int i = 0; i < size; i++) {
            if (input.data[i] >= 0) {
                data[i] = error.data[i];
            } else {
                data[i] = alpha.data[i] * error.data[i];
            }
        }
    }

    public void sineRelu(NNArray input, float epsilon) {
        for (int i = 0; i < size; i++) {
            if (input.data[i] > 0) {
                data[i] = input.data[i];
            } else {
                data[i] = (float) (epsilon * (Math.sin(input.data[i]) - Math.cos(input.data[i])));
            }
        }
    }

    public void derSineRelu(NNArray input, NNArray error, float param) {
        for (int i = 0; i < size; i++) {
            if (input.data[i] > 0) {
                data[i] = error.data[i];
            } else {
                data[i] = (float) (param * (Math.cos(input.data[i]) + Math.sin(input.data[i])));
            }
        }
    }

    public void silu(NNArray input) {
        for (int i = 0; i < size; i++) {
            data[i] = (float) (data[i] / (1 + pow(Math.E, -data[i])));
        }
    }

    public void derRelu(NNArray input, NNArray error) {
        for (int i = 0; i < size; i++) {
            if (input.data[i] > 0) {
                data[i] = error.data[i];
            }
        }
    }

    public void derRelu(NNArray output) {
        for (int i = 0; i < size; i++) {
            if (output.data[i] == 0) {
                data[i] = 0;
            }
        }
    }

    public void derReluMax(NNArray input, NNArray error, float max) {
        for (int i = 0; i < size; i++) {
            if (input.data[i] > 0 && input.data[i] <= max) {
                data[i] = error.data[i];
            }
        }
    }

    public void derSilu(NNArray input, NNArray error) {
        for (int i = 0; i < size; i++) {
            data[i] = (float) (error.data[i] * ((1 + pow(Math.E, -input.data[i]) + input.data[i] * pow(Math.E, -input.data[i]))
                    / Math.pow(1 + pow(Math.E, -input.data[i]), 2)));
        }
    }

    public void derSigmoid(NNArray output, NNArray error) {
        for (int i = 0; i < size; i++) {
            data[i] = output.data[i] * (1 - output.data[i]) * error.data[i];
        }
    }

    public void derTanh(NNArray output, NNArray error) {
        for (int i = 0; i < size; i++) {
            data[i] = (1 - output.data[i] * output.data[i]) * error.data[i];
        }
    }

    public void derLeakyRelu(NNArray input, NNArray error, float param) {
        for (int i = 0; i < size; i++) {
            if (input.data[i] > 0) {
                data[i] = error.data[i];
            } else {
                data[i] = param * error.data[i];
            }
        }
    }

    public void derElu(NNArray input, NNArray error, float param) {
        for (int i = 0; i < size; i++) {
            if (input.data[i] > 0) {
                data[i] = error.data[i];
            } else {
                data[i] = (float) (param * Math.pow(Math.E, input.data[i]) * error.data[i]);
            }
        }
    }

    public void sigmoid(NNArray input) {
        for (int i = 0; i < size; i++) {
            data[i] = (float) (1.0 / (1 + Math.pow(Math.E, -input.data[i])));
        }
    }

    public void tanh(NNArray input) {
        for (int i = 0; i < size; i++) {
            data[i] = (float) Math.tanh(input.data[i]);
        }
    }

    public void linear(NNArray input) {
        if (Use.CPU) {
            System.arraycopy(input.data, 0, data, 0, size);
        }

        if (Use.GPU) {
            this.copy(input);

            IsNan();
        }
    }

    public void elu(NNArray input, float param) {
        for (int i = 0; i < size; i++) {
            if (input.data[i] > 0) {
                data[i] = input.data[i];
            } else {
                data[i] = (float) ((Math.pow(Math.E, input.data[i]) - 1) * param);
            }
        }
    }

    public void softplus(NNArray input) {
        for (int i = 0; i < size; i++) {
            data[i] = (float) Math.log(Math.pow(Math.E, input.data[i]) + 1);
        }
    }

    public void hardSigmoid(NNArray input) {
        for (int i = 0; i < size; i++) {
            data[i] = Math.max(0, Math.min(1, input.data[i] * 0.2f + 0.5f));
        }
    }

    public void derHardSigmoid(NNArray output, NNArray error) {
        for (int i = 0; i < size; i++) {
            if (output.data[i] >= 0 && output.data[i] <= 1) {
                data[i] = 0.2f * error.data[i];
            }
        }
    }

    public void leakyRelu(NNArray input, float param) {
        for (int i = 0; i < size; i++) {
            if (input.data[i] > 0) {
                data[i] = input.data[i];
            } else {
                data[i] = input.data[i] * param;
            }
        }
    }

    public void l2norm() {
        float norm = 0;
        for (int i = 0; i < size; i++) {
            norm += data[i] * data[i];
        }
        div((float) Math.sqrt(norm) + 0.0000001f);
    }

    public void gaussian(NNArray input) {
        for (int i = 0; i < size; i++) {
            data[i] = (float) (Math.pow(Math.E, -input.data[i] * input.data[i]));
        }
    }

    public void derGaussian(NNArray input, NNArray error) {
        for (int i = 0; i < size; i++) {
            data[i] = (float) (-2 * input.data[i] * Math.pow(Math.E, -input.data[i] * input.data[i]) * error.data[i]);
        }
    }

    public float max() {
        float max = data[0];
        for (int i = 1; i < size; i++) {
            if (data[i] > max) {
                max = data[i];
            }
        }
        return max;
    }

    public int indexMaxElement() {
        int index = 0;

        if (Use.CPU) {
            float max = data[0];
            for (int i = 1; i < size; i++) {
                if (max < data[i]) {
                    index = i;
                    max = data[i];
                }
            }
        }
        if (Use.GPU) {
            IsNan();

            int[] maxIndex = new int[1];
            Pointer maxIndex_gpu = new Pointer();
            cudaMalloc(maxIndex_gpu, (long) Sizeof.INT);
            cudaMemset(maxIndex_gpu, 0, (long) Sizeof.INT);
            Pointer maxValue_gpu = new Pointer();
            cudaMalloc(maxValue_gpu, (long) Sizeof.INT);
            cudaMemset(maxValue_gpu, 0, (long) Sizeof.INT);
            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, helperModule, "indexMaxElement");
            Pointer kernelParameters = Pointer.to(Pointer.to(data_gpu), Pointer.to(maxValue_gpu), Pointer.to(maxIndex_gpu), Pointer.to(new int[]{size}));
            int blockSize = Math.min(size, BLOCK_SIZE);
            int gridSizeX = (int) Math.ceil((double) size / blockSize);
            cuLaunchKernel(function,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSize, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );

            cudaMemcpy(Pointer.to(maxIndex), maxIndex_gpu, (long) Sizeof.INT, cudaMemcpyDeviceToHost);
            index = maxIndex[0];

            if (Use.DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();

            JCuda.cudaFree(maxIndex_gpu);

            IsNan();
        }

        return index;
    }

    public int[] indexMaxElement(int count) {
        int[] index = new int[count];
        for (int m = 0; m < count; m++) {
            float max = 0;
            for (int i = 0; i < size; i++) {
                if (max < data[i] && m == 0) {
                    index[m] = i;
                    max = data[i];
                } else if (max < data[i]) {
                    boolean is = false;
                    for (int j = m - 1; j >= 0; j--) {
                        if (data[i] == data[index[j]]) {
                            is = true;
                            break;
                        }
                    }
                    if (is) {
                        continue;
                    }
                    index[m] = i;
                    max = data[i];
                }
            }
        }

        return index;
    }

    public void softmax(NNArray input) {
        float sum = 0;
        float max = input.max();

        for (int i = 0; i < size; i++) {
            data[i] = (float) (Math.pow(Math.E, input.data[i] - max));
            sum += data[i];
        }
        sum += 0.00000001f;

        for (int i = 0; i < size; i++) {
            data[i] /= sum;
        }
    }

    public void derSoftmax(NNArray output, NNArray error) {
        float value;
        for (int i = 0; i < size; i++) {
            data[i] = 0;
            for (int j = 0; j < size; j++) {
                if (i != j) {
                    value = output.data[i] * -output.data[j];
                } else {
                    value = output.data[i] * (1 - output.data[i]);
                }
                data[i] += error.getData()[j] * value;
            }
        }
    }

    public void momentum(NNArray array, final float decay) {
        if (Use.CPU) {
            final float rt = 1.0f - decay;
            for (int i = 0; i < size; i++) {
                data[i] = decay * data[i] + array.data[i] * rt;
            }
        }

        if (Use.GPU) {
            int p = size;

            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, helperModule, "momentum");
            Pointer kernelParameters = Pointer.to(Pointer.to(data_gpu), Pointer.to(array.data_gpu), Pointer.to(new float[]{decay}), Pointer.to(new int[]{p}));
            int blockSize = Math.min(p, BLOCK_SIZE);
            int gridSizeX = (int) Math.ceil((double) p / blockSize);
            cuLaunchKernel(function,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSize, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
            if (Use.DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();

            IsNan();
        }
    }

    public void momentumAbs(NNArray array, final float decay) {
        final float rt = 1.0f - decay;
        for (int i = 0; i < size; i++) {
            data[i] = decay * data[i] + Math.abs(array.data[i]) * rt;
        }
    }

    public void momentumNorm(NNArray array, NNArray e_array, final float decay) {
        final float rt = 1.0f - decay;
        for (int i = 0; i < size; i++) {
            data[i] = decay * data[i] + array.data[i] * rt * Math.max(1, e_array.data[i] / (Math.abs(array.data[i]) + 0.0000001f));
        }
    }

    public void momentumInject(NNArray array, NNArray deltaWeight, final float decay, float k) {
        final float rt = (1.0f - decay) / k;
        for (int i = 0; i < size; i++) {
            data[i] = decay * data[i] + (array.data[i] + deltaWeight.data[i] * array.data[i] * array.data[i]) * rt;
        }
    }

    public void subAndMul(NNArray vector, float val) {
        for (int i = 0; i < size; i++) {
            data[i] -= val * vector.data[i];
        }
    }

    public void subAndMulQH(NNArray vector, NNArray delta, float val, float v) {
        float v_ = 1f - v;
        for (int i = 0; i < size; i++) {
            data[i] -= val * (v * vector.data[i] + v_ * delta.data[i]);
        }
    }

    public void momentumPow2(NNArray vector, final float decay) {
        if (Use.CPU) {
            final float dr = 1 - decay;
            for (int i = 0; i < size; i++) {
                data[i] = decay * data[i] + dr * vector.data[i] * vector.data[i];
            }
        }

        if (Use.GPU) {
            int p = size;

            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, helperModule, "momentumPow2");
            Pointer kernelParameters = Pointer.to(Pointer.to(data_gpu), Pointer.to(vector.data_gpu), Pointer.to(new float[]{decay}), Pointer.to(new int[]{p}));
            int blockSize = Math.min(p, BLOCK_SIZE);
            int gridSizeX = (int) Math.ceil((double) p / blockSize);
            cuLaunchKernel(function,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSize, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
            if (Use.DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();

            IsNan();
        }
    }

    public void momentumPow2Sign(NNArray vector, final float decay) {
        final float dr = 1 - decay;
        float val;
        for (int i = 0; i < size; i++) {
            val = vector.data[i] * vector.data[i];
            data[i] -= dr * signum(data[i] - val) * val;
        }
    }

    public void subDivSqrt(NNArray nominator, NNArray denominator, float lr) {
        for (int i = 0; i < size; i++) {
            data[i] -= lr * nominator.data[i] / (Math.sqrt(denominator.data[i]) + 0.0000001f);
        }
    }

    public void subDivSqrtNorm(NNArray nominator, NNArray denominator, float lr, float normN, float normD) {
        if (Use.CPU) {
            float cur_lr = lr / (normN + 0.0000001f);
            for (int i = 0; i < size; i++) {
                data[i] -= cur_lr * (nominator.data[i]) / (Math.sqrt(denominator.data[i] / normD) + 0.0000001f);
            }
        }

        if (Use.GPU) {
            int p = size;

            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, helperModule, "subDivSqrtNorm");
            Pointer kernelParameters = Pointer.to(Pointer.to(nominator.data_gpu), Pointer.to(denominator.data_gpu), Pointer.to(new float[]{lr}), Pointer.to(new float[]{normN}), Pointer.to(new float[]{normD}), Pointer.to(data_gpu), Pointer.to(new int[]{p}));
            int blockSize = Math.min(p, BLOCK_SIZE);
            int gridSizeX = (int) Math.ceil((double) p / blockSize);
            cuLaunchKernel(function,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSize, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
            if (Use.DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();

            IsNan();
        }
    }

    private float absSigmoid(float val) {
        return (float) (1.0f / (1.0f + Math.pow(Math.E, -Math.abs(val))));
    }

    public void subDivSqrtNormDiff(NNArray nominator, NNArray denominator, NNArray der, NNArray derPre, float lr, float normN, float normD) {
        float cur_lr = lr / (normN + 0.0000001f);
        for (int i = 0; i < size; i++) {
            data[i] -= cur_lr * absSigmoid(derPre.data[i] - der.data[i]) * (nominator.data[i])
                    / (Math.sqrt(denominator.data[i] / normD) + 0.0000001f);
        }
    }

    public void subDivSqrtNorm(NNArray nominator, NNArray denominator, NNArray phi, float lr, float normN, float normD) {
        float cur_lr = lr / (normN + 0.0000001f);
        for (int i = 0; i < size; i++) {
            data[i] -= cur_lr * phi.data[i] * (nominator.data[i])
                    / (Math.sqrt(denominator.data[i] / normD) + 0.0000001f);
        }
    }

    public void subDivPowNorm(NNArray nominator, NNArray denominator, float lr, float normN, float normD, float p) {
        float cur_lr = lr / (normN + 0.0000001f);
        for (int i = 0; i < size; i++) {
            data[i] -= cur_lr * (nominator.data[i]) / (Math.pow(denominator.data[i] / normD, p) + 0.0000001f);
        }
    }

    public void subDivSqrtNormClip(NNArray nominator, NNArray denominator, float lr, float normN, float normD, float min, float max) {
        float cur_lr = 1 / (normN + 0.0000001f);
        for (int i = 0; i < size; i++) {
            data[i] -= clip((float) (lr / (Math.sqrt(denominator.data[i] / normD) + 0.0000001f)), min, max) * cur_lr * (nominator.data[i]);
        }
    }

    public float clip(float val, float min, float max) {
        if (val < min) {
            return min;
        }
        return Math.min(val, max);
    }

    public void deltaSubDivSqrtNorm(NNArray nominator, NNArray denominator, float lr, float normN, float normD) {
        float cur_lr = lr / (normN + 0.0000001f);
        for (int i = 0; i < size; i++) {
            data[i] += cur_lr * (nominator.data[i]) / (Math.sqrt(denominator.data[i] / normD) + 0.0000001f);
        }
    }

    public void subDivSqrtNormQH(NNArray gradient, NNArray nominator, NNArray denominator, float lr, float normN, float normD, float v1, float v2) {
        float cur_lr = lr / (normN + 0.0000001f);
        float v_1 = 1f - v1;
        float v_2 = 1f - v2;
        for (int i = 0; i < size; i++) {
            data[i] -= cur_lr * (v_1 * gradient.data[i] + v1 * nominator.data[i])
                    / (Math.sqrt(v_2 * gradient.data[i] * gradient.data[i] + v2 * denominator.data[i] / normD) + 0.0000001f);
        }
    }

    public void subDivSqrtNormNesterov(NNArray nominator, NNArray denominator, NNArray grad, float lr, float beta1, float normN, float normD) {
        float bt = (1.0f - beta1) / (normN);
        for (int i = 0; i < size; i++) {
            data[i] -= lr * (beta1 * nominator.data[i] + bt * grad.data[i]) / (Math.sqrt(denominator.data[i] / normD) + 0.0000001f);
        }
    }

    public void subDivNormNesterov(NNArray nominator, NNArray denominator, NNArray grad, float lr, float beta1, float normN) {
        float bt = (1.0f - beta1) / (normN);
        for (int i = 0; i < size; i++) {
            data[i] -= lr * (beta1 * nominator.data[i] + bt * grad.data[i]) / (denominator.data[i] + 0.0000001f);
        }
    }

    public void addPow2(NNArray vector) {
        for (int i = 0; i < size; i++) {
            data[i] += vector.data[i] * vector.data[i];
        }
    }

    public void momentumN(NNArray array, final float decay, final float lr) {
        for (int i = 0; i < size; i++) {
            data[i] = decay * data[i] - array.data[i] * lr;
        }
    }

    public void addMomentumN(NNArray derivative, NNArray decay, final float decayR, final float lr) {
        for (int i = 0; i < size; i++) {
            data[i] += decayR * decay.data[i] - derivative.data[i] * lr;
        }
    }

    public NNArray divSqrt(NNArray nominator, NNArray denominator) {
        NNArray result = new NNArray(nominator.size);
        for (int i = 0; i < size; i++) {
            result.data[i] = (float) (data[i] * Math.sqrt(nominator.data[i] + 0.0000001f) / (Math.sqrt(denominator.data[i]) + 0.0000001f));
        }
        return result;
    }

    public NNArray angularGrad(NNArray array) {
        NNArray result = new NNArray(array.size);
        for (int i = 0; i < size; i++) {
            result.data[i] = (float) Math.atan((data[i] - array.data[i]) / (1 + data[i] * array.data[i]));
        }
        return result;
    }

    public NNArray angularCos(NNArray array, float lambda1, float lambda2) {
        NNArray result = new NNArray(array.size);
        for (int i = 0; i < size; i++) {
            result.data[i] = (float) Math.tanh(Math.cos(Math.min(data[i], array.data[i]))) * lambda1 + lambda2;
        }
        return result;
    }

    public NNArray angularTan(NNArray array, float lambda1, float lambda2) {
        NNArray result = new NNArray(array.size);
        for (int i = 0; i < size; i++) {
            result.data[i] = (float) Math.tanh(Math.tan(Math.min(data[i], array.data[i]))) * lambda1 + lambda2;
        }
        return result;
    }

    public void dropout(NNArray input, double chanceDrop) {
        if (Use.CPU) {
            float drop = (float) (1.0f / (1.0f - chanceDrop));
            for (int i = 0; i < size; i++) {
                if (Math.random() > chanceDrop) {
                    data[i] = input.data[i] * drop;
                }
            }
        }

        if (Use.GPU) {
            dropout_GPU(input, chanceDrop);
        }
    }

    private static void dropout_GPU(NNArray A, double chanceDrop) {
        int n = A.size;
        int blockSize = Math.min(n, BLOCK_SIZE);
        int gridSizeX = (int) Math.ceil((double) n / blockSize);
        CUfunction function = new CUfunction();
        float[] randomArray = new float[n];
        for (int i = 0; i < n; i++) {
            randomArray[i] = (float) Math.random();
        }

        NNArray randomArrayGPU = new NNArray(randomArray);

        cuModuleGetFunction(function, helperModule, "dropout");
        Pointer kernelParameters = Pointer.to(Pointer.to(A.data_gpu), Pointer.to(randomArrayGPU.data_gpu), Pointer.to(new double[]{chanceDrop}), Pointer.to(new int[]{n}));
        cuLaunchKernel(function,
                gridSizeX, 1, 1,      // Grid dimension
                blockSize, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                kernelParameters, null // Kernel- and extra parameters
        );

        if (Use.DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
    }

    public void dropoutBack(NNArray output, NNArray error, double chanceDrop) {
        float drop = (float) (1.0f / (1.0f - chanceDrop));

        if (Use.CPU) {
            for (int i = 0; i < size; i++) {
                if (output.data[i] != 0) {
                    data[i] = error.data[i] * drop;
                }
            }
        }

        if (Use.GPU) {
            int p = size;

            Pointer dVar_Pointer = new Pointer();
            cudaMalloc(dVar_Pointer, (long) p * Sizeof.FLOAT);
            cudaMemset(dVar_Pointer, 0, (long) p * Sizeof.FLOAT);

            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, helperModule, "dropoutBack");
            Pointer kernelParameters = Pointer.to(Pointer.to(output.data_gpu), Pointer.to(error.data_gpu), Pointer.to(new float[]{drop}), Pointer.to(data_gpu), Pointer.to(new int[]{p}));
            int blockSize = Math.min(p, BLOCK_SIZE);
            int gridSizeX = (int) Math.ceil((double) p / blockSize);
            cuLaunchKernel(function,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSize, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );

            if (Use.DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
            JCuda.cudaFree(dVar_Pointer);


            IsNan();
        }
    }

    public void save(FileWriter writer) throws IOException {
        float[] hostData = null;
        if (Use.GPU) {
            hostData = GetFirstSingleValueFloat(data_gpu, size);
        }

        writer.write(size + "\n");
        int row = (int) Math.ceil(size / 1024.0);
        int column = 1024;
        for (int i = 0; i < row; i++) {
            int i_index = i * 1024;
            if (size - i_index < 1024) {
                column = size - i_index;
            }
            for (int j = 0; j < column; j++, i_index++) {
                if (Use.CPU) {
                    writer.write(data[i_index] + " ");
                } else {
                    writer.write(hostData[i_index] + " ");
                }
            }
            writer.write("\n");
            writer.flush();
        }
        writer.flush();
    }

    public static NNArray read(Scanner scanner) {
        NNArray array = new NNArray(Integer.parseInt(scanner.nextLine()));
        int row = (int) Math.ceil(array.size / 1024.0);
        if (Use.CPU) {
            for (int i = 0; i < row; i++) {
                int i_index = i * 1024;
                double[] arr = Arrays.stream(scanner.nextLine().split(" ")).mapToDouble(Float::parseFloat).toArray();
                for (int j = 0; j < arr.length; j++, i_index++) {
                    array.data[i_index] = (float) arr[j];
                }
            }
        } else {
            float[] hostdata = new float[array.size];
            for (int i = 0; i < row; i++) {
                int i_index = i * 1024;
                double[] arr = Arrays.stream(scanner.nextLine().split(" ")).mapToDouble(Float::parseFloat).toArray();
                for (int j = 0; j < arr.length; j++, i_index++) {
                    hostdata[i_index] = (float) arr[j];
                }
            }
            cudaMemcpy(array.data_gpu, Pointer.to(hostdata), (long) Sizeof.FLOAT * array.size, cudaMemcpyHostToDevice);
        }
        return array;
    }

    // A = alpha
    static void scalarSet(NNArray A, float alpha) {
        int n = A.size;
        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, helperModule, "vectorScalarSet");
        Pointer kernelParameters = Pointer.to(Pointer.to(A.data_gpu), Pointer.to(new float[]{alpha}), Pointer.to(new int[]{n}));
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

    public Pointer sum_gpu(NNArray array) {
        Pointer sum_gpu = new Pointer();
        cudaMalloc(sum_gpu, (long) Sizeof.FLOAT);
        cudaMemset(sum_gpu, 0, (long) Sizeof.FLOAT);

        //cublasSasum(cublasHandle, array.size, array.data_gpu, 1,  Pointer.to(sum_gpu));

        int n = array.size;
        CUfunction function = new CUfunction();

        cuModuleGetFunction(function, helperModule, "sum");
        Pointer kernelParameters = Pointer.to(Pointer.to(array.data_gpu), Pointer.to(sum_gpu), Pointer.to(new int[]{n}));
        int blockSize = Math.min(n, BLOCK_SIZE);
        int gridSizeX = (int) Math.ceil((double) n / blockSize);
        cuLaunchKernel(function,
                gridSizeX, 1, 1,      // Grid dimension
                blockSize, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                kernelParameters, null // Kernel- and extra parameters
        );

        if (Use.DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();

        return sum_gpu;
    }

    public float[] toArray() {
        float[] data_h = new float[size];
        JCublas2.cublasGetVector(data_h.length, Sizeof.FLOAT, Pointer.to(data_gpu), 1, Pointer.to(data_h), 1);
        return data_h;
    }

    public float GetFirstSingleValue(Pointer data) {
        float[] data_h = new float[1];
        JCublas2.cublasGetVector(data_h.length, Sizeof.FLOAT, data, 1, Pointer.to(data_h), 1);
        return data_h[0];
    }

    public static float[] GetFirstSingleValueFloatStatic(Pointer data_gpu, int n) {
        float[] data_h = new float[n];
        cudaMemcpy(Pointer.to(data_h), data_gpu, (long) n * Sizeof.FLOAT, cudaMemcpyDeviceToHost);
        return data_h;
    }

    public float[] GetFirstSingleValueFloat(Pointer data_gpu, int n) {
        float[] data_h = new float[n];
        cudaMemcpy(Pointer.to(data_h), data_gpu, (long) n * Sizeof.FLOAT, cudaMemcpyDeviceToHost);
        return data_h;
    }

    public double[] GetFirstSingleValueDouble(Pointer data_gpu, int n) {
        double[] data_h = new double[n];
        cudaMemcpy(Pointer.to(data_h), data_gpu, (long) n * Sizeof.DOUBLE, cudaMemcpyDeviceToHost);
        return data_h;
    }

    public int[] GetFirstSingleValueInt(Pointer data_gpu, int n) {
        int[] data_h = new int[n];
        JCublas2.cublasGetVector(n, Sizeof.INT, data_gpu, 1, Pointer.to(data_h), 1);
        return data_h;
    }

    public static float[] toFloatArray(double[] arr) {
        if (arr == null) return null;
        int n = arr.length;
        float[] ret = new float[n];
        for (int i = 0; i < n; i++) {
            ret[i] = (float) arr[i];
        }
        return ret;
    }

    public static int fromFloat( float fval )
    {
        int fbits = Float.floatToIntBits( fval );
        int sign = fbits >>> 16 & 0x8000;          // sign only
        int val = ( fbits & 0x7fffffff ) + 0x1000; // rounded value

        if( val >= 0x47800000 )               // might be or become NaN/Inf
        {                                     // avoid Inf due to rounding
            if( ( fbits & 0x7fffffff ) >= 0x47800000 )
            {                                 // is or must become NaN/Inf
                if( val < 0x7f800000 )        // was value but too large
                    return sign | 0x7c00;     // make it +/-Inf
                return sign | 0x7c00 |        // remains +/-Inf or NaN
                        ( fbits & 0x007fffff ) >>> 13; // keep NaN (and Inf) bits
            }
            return sign | 0x7bff;             // unrounded not quite Inf
        }
        if( val >= 0x38800000 )               // remains normalized value
            return sign | val - 0x38000000 >>> 13; // exp - 127 + 15
        if( val < 0x33000000 )                // too small for subnormal
            return sign;                      // becomes +/-0
        val = ( fbits & 0x7fffffff ) >>> 23;  // tmp exp for subnormal calc
        return sign | ( ( fbits & 0x7fffff | 0x800000 ) // add subnormal bit
                + ( 0x800000 >>> val - 102 )     // round depending on cut off
                >>> 126 - val );   // div by 2^(1-(exp-127+15)) and >> 13 | exp=0
    }

    public void free() {
        //if (data_gpu != null) JCuda.cudaFree(data_gpu);
    }

    public boolean IsNan(NNArray data) {
        /*if (Use.DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
        int p = data.size;

        int[] result = new int[1];
        Pointer result_gpu = new Pointer();
        cudaMalloc(result_gpu, (long) Sizeof.INT);
        //cudaMemset(result_gpu, 0, Sizeof.INT);
        int[] init = new int[1];
        cudaMemcpy(result_gpu, Pointer.to(init), (long) Sizeof.INT, cudaMemcpyHostToDevice);

        if (Use.DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();

        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, helperModule, "isnan");
        Pointer kernelParameters = Pointer.to(Pointer.to(data.data_gpu), Pointer.to(result_gpu), Pointer.to(new int[]{p}));
        int blockSize = Math.min(p, BLOCK_SIZE);
        int gridSizeX = (int) Math.ceil((double) p / blockSize);
        cuLaunchKernel(function,
                gridSizeX, 1, 1,      // Grid dimension
                blockSize, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                kernelParameters, null // Kernel- and extra parameters
        );

        JCublas2.cublasGetVector(result.length, Sizeof.INT, result_gpu, 1, Pointer.to(result), 1);
        if (result[0] == 1)
        {
            if (Use.DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
            System.out.println("Error!!");
            return true;
        }
        if (Use.DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();*/
        return false;
    }

    public boolean IsNan() {
        /*if (Use.DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
        int p = size;

        int[] result = new int[1];
        Pointer result_gpu = new Pointer();
        cudaMalloc(result_gpu, (long) Sizeof.INT);
        //cudaMemset(result_gpu, 0, Sizeof.INT);
        int[] init = new int[1];
        cudaMemcpy(result_gpu, Pointer.to(init), (long) Sizeof.INT, cudaMemcpyHostToDevice);

        if (Use.DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();

        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, helperModule, "isnan");
        Pointer kernelParameters = Pointer.to(Pointer.to(data_gpu), Pointer.to(result_gpu), Pointer.to(new int[]{p}));
        int blockSize = Math.min(p, BLOCK_SIZE);
        int gridSizeX = (int) Math.ceil((double) p / blockSize);
        cuLaunchKernel(function,
                gridSizeX, 1, 1,      // Grid dimension
                blockSize, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                kernelParameters, null // Kernel- and extra parameters
        );

        JCublas2.cublasGetVector(result.length, Sizeof.INT, result_gpu, 1, Pointer.to(result), 1);
        if (result[0] == 1)
        {
            if (Use.DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
            System.out.println("Error!!");
            return true;
        }
        if (Use.DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();*/
        return false;
    }

    public static final String kernels =
                    "const __device__ float epsilon = 0.001f;\n" +
                    "#define MAX_FLOAT_EXP 		80\n" +
                    "#include <cuda_fp16.h>\n" +

                    "__device__ int SharedMemorySize = 64 * 1024 / 4;\n" +
                    "__device__ const int BLOCK_DIM = 32;\n" +

                    "extern \"C\"\n" +
                    "__global__ void fill(float* A, float alpha, int numElements)\n" +
                    "{\n" +
                    "    const int i = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (i < numElements) {\n" +
                    "        A[i] = alpha;\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void gelu(const float* __restrict__ A, float* C, int numElements)\n" +
                    "{\n" +
                    "    const int i = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (i < numElements) {\n" +
                    "        float a = A[i];\n" +
                    "        C[i] = 0.5f * a * (1.0f + tanh(0.7978846f * a + 0.0356774f * (a * a * a)));\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void set(float* A, int i, float alpha)\n" +
                    "{\n" +
                    "    A[i] = alpha;\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void multiplyIndex(float* A, int alpha, int numElements)\n" +
                    "{\n" +
                    "    const int i = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (i < numElements) {\n" +
                    "        A[i] = i * alpha;\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void imageVector(const float* __restrict__ A, float* C, int rows, int columns, int depth, int sizeKernel)\n" +
                    "{\n" +
                    "    const int h = (blockDim.x * blockIdx.x + threadIdx.x) * sizeKernel;\n" +
                    "    const int w = (blockDim.y * blockIdx.y + threadIdx.y) * sizeKernel;\n" +
                    "    const int z = blockDim.z * blockIdx.z + threadIdx.z;\n" +
                    "    if (h < rows && w < columns && z < sizeKernel)\n" +
                    "    {\n" +
                    "        int sizeKernel_X_depth = sizeKernel * depth;\n" +
                    "        int sizeKernel_X_sizeKernel_X_depth_ = sizeKernel_X_depth * sizeKernel;\n" +
                    "        int columns_X_sizeKernel_X_sizeKernel_X_depth = sizeKernel_X_sizeKernel_X_depth_ * columns / sizeKernel;\n" +
                    "        int index = z * sizeKernel_X_depth + w / sizeKernel * sizeKernel_X_sizeKernel_X_depth_ + h / sizeKernel * columns_X_sizeKernel_X_sizeKernel_X_depth;\n" +
                    "        for (int k = 0; k < sizeKernel; k++) {\n" +
                    "            int indexInput = (h + z) * depth * columns + (w + k) * depth;\n" +
                    "            for (int c = 0; c < depth; c++, index++, indexInput++) {\n" +
                    "                C[index] = A[indexInput];\n" +
                    "            }\n" +
                    "        }\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void backImageVector(const float* __restrict__ A, float* C, int rows, int columns, int depth, int sizeKernel)\n" +
                    "{\n" +
                    "    const int h = (blockDim.x * blockIdx.x + threadIdx.x) * sizeKernel;\n" +
                    "    const int w = (blockDim.y * blockIdx.y + threadIdx.y) * sizeKernel;\n" +
                    "    const int z = blockDim.z * blockIdx.z + threadIdx.z;\n" +
                    "    if (h < rows && w < columns && z < sizeKernel)\n" +
                    "    {\n" +
                    "        int sizeKernel_X_depth = sizeKernel * depth;\n" +
                    "        int sizeKernel_X_sizeKernel_X_depth_ = sizeKernel_X_depth * sizeKernel;\n" +
                    "        int columns_X_sizeKernel_X_sizeKernel_X_depth = sizeKernel_X_sizeKernel_X_depth_ * columns / sizeKernel;\n" +
                    "        int index = z * sizeKernel_X_depth + w / sizeKernel * sizeKernel_X_sizeKernel_X_depth_ + h / sizeKernel * columns_X_sizeKernel_X_sizeKernel_X_depth;\n" +
                    "        for (int k = 0; k < sizeKernel; k++) {\n" +
                    "            int indexInput = (h + z) * depth * columns + (w + k) * depth;\n" +
                    "            for (int c = 0; c < depth; c++, index++, indexInput++) {\n" +
                    "                C[indexInput] = A[index];\n" +
                    "            }\n" +
                    "        }\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void add3(const float* __restrict__ A, float* C, int rows, int columns)\n" +
                    "{\n" +
                    //"    extern __shared__ float shared[];\n" +
                    "    int h = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    int w = blockDim.y * blockIdx.y + threadIdx.y;\n" +
                    "    if (h < rows && w < columns) {\n" +
                    /*"       if (w < SharedMemorySize) {\n" +
                    "           shared[w] = A[w];\n" +
                    "       }\n" +
                    "       __syncthreads();\n" +*/
                    "       int index = h * blockDim.y * gridDim.y + w;\n" +
                    "       w = blockDim.y * blockIdx.y + threadIdx.y;\n" +
                    //"       if (w < SharedMemorySize) {\n" +
                    //"           C[index] += shared[w];\n" +
                    //"       }\n" +
                    //"       else {\n" +
                    "          C[index] += A[w];\n" +
                    //"       }\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void dot_VectorAndMatrix(const float* __restrict__ A, const float* __restrict__ B, float* C, int rows, int columns)\n" +
                    "{\n" +
                    "    int h = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (h < rows) {\n" +
                    "       float s = 0.0f;\n" +
                    "       for (int j = 0; j < columns; j++) {\n" +
                    "           s += B[j] * A[h * columns + j];\n" +
                    "       }\n" +
                    "       C[h] = s;\n" +
                    "    }\n" +
                    "}\n" +

                    /*"extern \"C\"\n" +
                    "__global__ void gpu_coalescing_shared_multipication(float* a, float* x, float* res, int n, int m) {\n" +
                    "    __shared__ float xx[1024];\n" +
                    "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n" +
                    "    if (idx < n) res[idx] = 0;\n" +
                    "    for (int i = 0; i < m; i += blockDim.x) {\n" +
                    "        if (i + threadIdx.x < m)\n" +
                    "            xx[threadIdx.x] = x[i + threadIdx.x];\n" +
                    "        __syncthreads();\n" +
                    "        if (idx < n)\n" +
                    "            for (int j = 0; j < blockDim.x && i + j < m; j++)\n" +
                    "                res[idx] += a[idx + (i + j)*n] * xx[j];\n" +
                    "        __syncthreads();\n" +
                    "    }\n" +
                    "}\n" +*/

                    /*"extern \"C\"\n" +
                    "__global__ void dot_VectorAndMatrix(const float* __restrict__ A, const float* __restrict__ x, float* y, int m, int n) {\n" +
                    "    const int TILE_SIZE = 32;\n" +
                    "    __shared__ float shared_A[TILE_SIZE][TILE_SIZE];\n" +
                    "    __shared__ float shared_x[TILE_SIZE];\n" +

                    "    int tile_id = blockIdx.x;\n" +
                    "    int row = threadIdx.x;\n" +

                    "    float sum = 0.0f;\n" +
                    "    int num_tiles = (n - 1) / TILE_SIZE + 1;\n" +

                    "    for (int i = 0; i < num_tiles; i++) {\n" +
                    "        int tile_start = i * TILE_SIZE;\n" +
                    "        int tile_end = min(tile_start + TILE_SIZE, n);\n" +

                    "        if (tile_start + row < m) {\n" +
                    "            shared_x[row] = x[tile_start + row];\n" +
                    "        }\n" +

                    "        for (int j = 0; j < TILE_SIZE; j++) {\n" +
                    "            if (tile_start + j < tile_end && tile_start + row < m) {\n" +
                    "                shared_A[row][j] = A[(tile_start + row) * n + tile_start + j];\n" +
                    "            }\n" +
                    "        }\n" +

                    "        __syncthreads();\n" +

                    "        for (int j = 0; j < TILE_SIZE; j++) {\n" +
                    "            if (tile_start + j < tile_end && tile_start + row < m) {\n" +
                    "                sum += shared_A[row][j] * shared_x[j];\n" +
                    "            }\n" +
                    "        }\n" +

                    "        __syncthreads();\n" +
                    "    }\n" +

                    "    if (TILE_SIZE * tile_id + row < m) {\n" +
                    "        y[TILE_SIZE * tile_id + row] = sum;\n" +
                    "    }\n" +
                    "}\n" +*/

                    "extern \"C\"\n" +
                    "__global__ void dotT_VectorAndMatrix(const float* __restrict__ A, const float* __restrict__ B, float* C, int rows, int columns)\n" +
                    "{\n" +
                    "    int j = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (j < columns) {\n" +
                    "       float sum = 0.0f;\n" +
                    "       for (int i = 0; i < rows; i++) {\n" +
                    "            int index = floorf(i * columns + j);\n" +
                    "            sum += A[i] * B[index];\n" +
                    "       }\n" +
                    "       C[j] = sum;\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void derivativeWeight(const float* __restrict__ input, const float* __restrict__ error, float* derWeight, int rows, int columns)\n" +
                    "{\n" +
                    "    unsigned idx = threadIdx.x + blockDim.x * blockIdx.x;\n" +
                    "    if (idx < rows * columns) {\n" +
                    "        derWeight[idx] += error[idx / columns] * input[idx % columns];\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void findMean_part(const float* __restrict__ A, float* C, int width, int depth)\n" +
                    "{\n" +
                    "    int j = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (j < width) {\n" +
                    "       float s = 0.0f;\n" +
                    "       int index = j * depth;\n" +
                    "       for (int k = 0; k < depth; k++, index++) {\n" +
                    "           s += A[index];\n" +
                    "       }\n" +
                    "       C[j] = s;\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void generateErrorNorm(const float* __restrict__ errorNL, const float* __restrict__ gamma, float* errorNorm, int width, int depth)\n" +
                    "{\n" +
                    "    const int j = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    const int k = blockDim.y * blockIdx.y + threadIdx.y;\n" +
                    "    const int index = j * blockDim.y * gridDim.y + k;\n" +
                    "    if (index < width * depth) {\n" +
                    "       errorNorm[index] = errorNL[index] * gamma[k];\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void derVar_part_2(const float* __restrict__ error, const float* __restrict__ input, const float* __restrict__ mean, float* derVariance, int width, int depth)\n" +
                    "{\n" +
                    "    const int j = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (j < width) {\n" +
                    "        int index = j * depth;\n" +
                    "        float m = mean[j];\n" +
                    "        float s = 0.0f;\n" +
                    "        for (int k = 0; k < depth; k++, index++) {\n" +
                    "           s += (float)(error[index] * (input[index] - m));\n" +
                    "        }\n" +
                    "        derVariance[j] = s;\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void derMean_part_2(const float* __restrict__ error, const float* __restrict__ input, const float* __restrict__ mean, float* derMean, float* dVar, int width, int depth)\n" +
                    "{\n" +
                    "    const int j = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (j < width) {\n" +
                    "        int index = j * depth;\n" +
                    "        float DM = 0.0f;\n" +
                    "        float DV = 0.0f;\n" +
                    "        float m = mean[j];\n" +
                    "        for (int k = 0; k < depth; k++, index++) {\n" +
                    "           DM += error[index];\n" +
                    "           DV += input[index] - m;\n" +
                    "        }\n" +
                    "        derMean[j] = DM;\n" +
                    "        dVar[j] = DV;\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void derNorm_part_2(const float* __restrict__ errors, const float* __restrict__ dVar, const float* __restrict__ errorVar, const float* __restrict__ input, const float* __restrict__ mean, const float* __restrict__ errorMean, float* error, int width, int depth)\n" +
                    "{\n" +
                    "    const int j = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    const int k = blockDim.y * blockIdx.y + threadIdx.y;\n" +
                    "    const int index = j * blockDim.y * gridDim.y + k;\n" +
                    "    if (index < width * depth) {\n" +
                    "        error[index] = errors[index] * dVar[j] + errorVar[j] * (input[index] - mean[j]) + errorMean[j];\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void derivativeWeight_2(const float* __restrict__ error, const float* __restrict__ output, const float* __restrict__ betta, const float* __restrict__ gamma, float* derBetta, float* derGamma, int width, int depth)\n" +
                    "{\n" +
                    "    const int i = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (i < depth) {\n" +
                    "       float dB = derBetta[i];\n" +
                    "       float dG = derGamma[i];\n" +
                    "       float g = gamma[i];\n" +
                    "       float b = betta[i];\n" +
                    "       for (int j = 0; j < width; j++) { \n" +
                    "           int ind = floorf(j * depth + i);\n" +
                    "           dB += error[ind];\n" +
                    "           dG += error[ind] * ((output[ind] - b) / g);\n" +
                    "       }\n" +
                    "       derBetta[i] = dB;\n" +
                    "       derGamma[i] = dG;\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void findVariance_part(const float* __restrict__ input, const float* __restrict__ mean, float* var, int width, int depth)\n" +
                    "{\n" +
                    "    const int j = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (j < width) {\n" +
                    "        float s = 0.0f;\n" +
                    "        float m = mean[j];\n" +
                    "        int index = j * depth;\n" +
                    "        for (int k = 0; k < depth; k++, index++) {\n" +
                    "           float sub = input[index] - m;\n" +
                    "           s += sub * sub;\n" +
                    "        }\n" +
                    "        var[j] = s;\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void normalization_part_2(const float* __restrict__ input, const float* __restrict__ mean, const float* __restrict__ varSqrt, float* normOutput, const float* __restrict__ gamma, const float* __restrict__ betta, float* output, int width, int depth)\n" +
                    "{\n" +
                    "    int j = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    int k = blockDim.y * blockIdx.y + threadIdx.y;\n" +
                    "    if (j < width && k < depth) {\n" +
                    "        int index = j * blockDim.y * gridDim.y + k;\n" +
                    "        float nO = (input[index] - mean[j]) / varSqrt[j];\n" +
                    "        output[index] = nO * gamma[k] + betta[k];\n" +
                    "        normOutput[index] = nO;\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void add_NNMatrix(const float* __restrict__ matrix, float* data, int width, int depth)\n" +
                    "{\n" +
                    "    int k = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (k < depth) {\n" +
                    "       double d = (double)data[k];\n" +
                    "       for (int i = 0; i < width; i++) { \n" +
                    "           int	index = floorf(i * depth + k);\n" +
                    "           d += matrix[index];\n" +
                    "       }\n" +
                    "       data[k] = (float)d;\n" +
                    "    }\n" +
                    "  }\n" +

                    "extern \"C\"\n" +
                    "__global__ void reverse(float* A, int rows, int columns, int depth)\n" +
                    "{\n" +
                    "    const int i = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    const int j = blockDim.y * blockIdx.y + threadIdx.y;\n" +
                    "    const int k = blockDim.z * blockIdx.z + threadIdx.z;\n" +
                    "    const int index = i * blockDim.y * gridDim.y + j;\n" +
                    "    if (index < rows * columns  && (k < depth))\n" +
                    "    {\n" +
                    "       const int index = rows - 1 - i;\n" +
                    "       float valf = A[i * depth * columns + j * depth + k];\n" +
                    "       float vals = A[index  * depth * columns + j * depth + k];\n" +
                    "       A[i  * depth * columns + j * depth + k] = valf;\n" +
                    "       A[index  * depth * columns + j * depth + k] = vals;\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void set2(float* A, int i, int j, int k, int columns, int depth, float value)\n" +
                    "{\n" +
                    "    A[i * depth * columns + j * depth + k] = value;\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void MatAdd(float* A, const float* __restrict__ B, int numElements)\n" +
                    "{\n" +
                    "    const int k = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (k < numElements) {\n" +
                    "       atomicAdd(&A[k], B[k]);\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void matrixDiv(float* A, float B, int numElements)\n" +
                    "{\n" +
                    "    const int i = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (i < numElements) {\n" +
                    "        A[i] /= B;\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void vectorScalarSet(float* A, float alpha, int numElements)\n" +
                    "{\n" +
                    "    const int i = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (i < numElements) {\n" +
                    "        A[i] = alpha;\n" +
                    "    }\n" +
                    "}\n" +

                    /*"extern \"C\"\n"+
                    "__global__ void Softmax(const float* __restrict__ A, float* auxE, int sample_dim, float* N, int numElements)\n"+
                    "{\n"+
                    //This way of programing allow no warp syncronization as we only need kernel optimization.
                    //Maybe use thrust library but could be tricky. Should study if it fit well with this problem. Think is not, we would need one thrust vector per row.
                    //On the other hand possibly implement reduction as in http://www.cuvilib.com/Reduction.pdf. Will need to call another function. This could be complecated also as we need to see which thread id implements softmax and which one computes maximum. For now simple approximation.
                    "   float C_value = 0;\n"+
                    "   int thread_id_x = blockDim.x * blockIdx.x + threadIdx.x;\n"+
                    "   float maxCoef = A[thread_id_x*sample_dim];\n"+
                    "   float actualCoef = 0;\n"+
                    "   double E = 2.718281828459045;\n"+
                    "   if (thread_id_x < numElements)\n"+
                    "   {\n"+
                    ///REALLY HIGH PROBABILITY OF BRANCH DIVERGENCE.
                    //Description: All of the threads that lie under one condition execute first (stalling the others) and then next. Assuming one clock cycle per operation we would need double time to execute one warp.
                    //Warping divergence: study reduction options for getting the maximum
                    "#pragma omp parallel for\n"+
                    "       for (int cA = 1; cA < sample_dim; cA++)\n"+
                    "       if (A[thread_id_x * sample_dim + cA] > maxCoef)\n"+
                    "           maxCoef = A[thread_id_x * sample_dim+cA];\n"+

                    //No warping divergence as all threads execute the same
                    "#pragma omp parallel for\n"+
                "           for (int cA = 0; cA < sample_dim; cA++)\n"+
                    "       {\n"+
                    "           actualCoef = (float) pow(E, (double)(A[thread_id_x * sample_dim + cA] - maxCoef));\n"+
                    "           auxE[thread_id_x * sample_dim + cA] = actualCoef;\n"+
                    "           C_value += actualCoef;\n"+
                    "       }\n"+
                    "#pragma omp parallel for\n"+
                    "       C_value += 0.00000001f;\n" +
                    "       for (int cA = 0; cA < sample_dim; cA++)\n"+
                    "       {\n"+
                    "           N[thread_id_x * sample_dim + cA] = auxE[thread_id_x * sample_dim + cA] / C_value;\n"+
                    "       }\n"+
                    "   }\n"+
                    "}\n" +*/

                    "extern \"C\"\n" +
                    "__global__ void addCopy(const float* __restrict__ matrix, float* data, int row, int col, int m_col, int start) \n" +
                    "{\n" +
                    "    const int x = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    const int y = blockDim.y * blockIdx.y + threadIdx.y;\n" +
                    "    const int index = x * blockDim.y * gridDim.y + y;\n" +
                    "    if (index < row * m_col)\n" +
                    "    {\n" +
                    "        const int indexIn = x * col + start * m_col + y;\n" +
                    "        data[indexIn] = matrix[index];\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void normalization_part_1(float* A, const float* __restrict__ var, int numElements)\n" +
                    "{\n" +
                    "    const int i = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (i < numElements) {\n" +
                    "       A[i] = (float)sqrt(var[i] + epsilon);\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void NormalizationLayerForward2D(float*** P, const float* __restrict__ gamma, const float* __restrict__ betta, int width, int depth, int n)\n" +
                    "{\n" +
                    "    int x = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    int y = blockDim.y * blockIdx.y + threadIdx.y;\n" +
                    "    if (x < n && y < width) {\n" +
                    "       float mean = P[3][x][y];\n" +
                    "       for (int index = y * depth, int k = 0; k < depth; k++, index++) {\n" +
                    "           mean += P[2][x][index];\n" +
                    "       }\n" +
                    "       P[3][x][y] = mean;\n" +

                    "       float var = P[4][x][y];\n" +
                    "       float sub;\n" +
                    "       for (int index = y * depth, int k = 0; k < depth; k++, index++) {\n" +
                    "           sub = (float)(P[2][x][index] - mean);\n" +
                    "           var += sub * sub;\n" +
                    "       }\n" +
                    "       var = var / depth;\n" +
                    "       P[4][x][y] = var;\n" +

                    "       float varSqrt = (float) (sqrt(var + epsilon));\n" +
                    "       for (int index = y * depth, int k = 0; k < depth; k++, index++) {\n" +
                    "           float nO = ((float)(P[2][x][index] - mean)) / varSqrt;\n" +
                    "           P[0][x][index] = nO;\n" +
                    "           P[1][x][index] = nO * gamma[k] + betta[k];\n" +
                    "       }\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void DenseLayerForward(float*** P, float* weight, const float* __restrict__ threshold, int row, int column, int n)\n" +
                    "{\n" +
                    "    int x = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    int y = blockDim.y * blockIdx.y + threadIdx.y;\n" +
                    "    if (x < n && y < row) {\n" +
                    "       int index = y * column;\n" +
                    "       float sum = P[1][x][y];\n" +
                    "       for (int j = 0; j < column; j++, index++) {\n" +
                    "           P[1][x][y] += P[0][x][j] * weight[index];\n" +
                    "       }\n" +
                    "       P[1][x][y] = sum;\n" +
                    "       atomicAdd(&P[1][x][y], threshold[y]);\n" +
                    "    }\n" +
                    "}\n" +

                    /*"extern \"C\"\n" +
                    "__device__ void dot(float* const __restrict__ a, const float* __restrict__ b, float* result, int row1, int col1, int row2, int col2)\n" +
                    "{\n" +
                    "    for (int i = 0; i < row1; i++) {\n" +
                    "        for (int j = 0; j < col2; j++) {\n" +
                    "            float sum = 0;\n" +
                    "            for (int k = 0; k < col1; k++)\n" +
                    "                sum = sum + a[i * col1 + k] * b[k * col2 + j];\n" +
                    "            result[i * col2 + j] = sum;\n" +
                    "        }\n" +
                    "    }\n" +
                    "}\n" +*/

                    "extern \"C\"\n" +
                    "__global__ void dot(float* const __restrict__ a, const float* __restrict__ b, float* result, int row1, int col1, int row2, int col2)\n" +
                    "{\n" +
                    "    int i = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    int j = blockDim.y * blockIdx.y + threadIdx.y;\n" +
                    "    if (i < row1 && j < col2) {\n" +
                    "       float sum = 0.0f;\n" +
                    "       for (int k = 0; k < col1; k++) {\n" +
                    "           sum = sum + a[i * col1 + k] * b[k * col2 + j];\n" +
                    "       }\n" +
                    "       result[i * col2 + j] = sum;\n" +
                    "    }\n" +
                    "}\n" +

                    /*"__device__ void transpose(const float* __restrict__ data, float* result, int row, int col)\n" +
                    "{\n" +
                    "    int index;\n" +
                    "    for (int i = 0; i < row; i++) {\n" +
                    "        index = i * col;\n" +
                    "        for (int j = 0; j < col; j++, index++) {\n" +
                    "            result[i + j * col] = data[index];\n" +
                    "        }\n" +
                    "    }\n" +
                    "}\n" +*/

                    "extern \"C\"\n" +
                    "__global__ void transpose(float* odata, const float* __restrict__ idata, int width, int height)\n" +
                    "{\n" +
                    "    __shared__ float block[BLOCK_DIM][BLOCK_DIM+1];\n" +
                    "    unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;\n" +
                    "    unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;\n" +
                    "    if((xIndex < width) && (yIndex < height))\n" +
                    "    {\n" +
                    "        unsigned int index_in = yIndex * width + xIndex;\n" +
                    "        block[threadIdx.y][threadIdx.x] = idata[index_in];\n" +
                    "    }\n" +
                    "    __syncthreads();\n" +
                    "    xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;\n" +
                    "    yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;\n" +
                    "    if((xIndex < height) && (yIndex < width))\n" +
                    "    {\n" +
                    "        unsigned int index_out = yIndex * height + xIndex;\n" +
                    "        odata[index_out] = block[threadIdx.x][threadIdx.y];\n" +
                    "    }\n" +
                    "}\n" +

                    /*"extern \"C\"\n" +
                    "__device__ void dot(float* const __restrict__ a, const float* __restrict__ b, float* result, int row, int col, int row2, int col2)\n" +
                    "{\n" +
                    "    float* T = new float[row * col2];\n" +
                    "    transpose(b, T, row2, col2);\n" +
                    "    dotT(a, T, result, row, col, row2, col2);\n" +
                    "    free(result);\n" +
                    "}\n" +*/

                    /*"extern \"C\"\n" +
                    "__global__ void ImagePatchesLayerForward(float*** P, const float* __restrict__ weight, int row, int col, int depth, int patch_row, int patch_col, int weight_row, int weight_col, int sizeKernel, int n)\n" +
                    "{\n" +
                    "    int x = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    int h = blockDim.y * blockIdx.y + threadIdx.y;\n" +
                    "    if (x < n && h < row) {\n" +
                    "        int indexInput;\n" +
                    "        int index_h = h * sizeKernel;\n" +
                    "        if (index_h < row) {\n" +
                    "           for (int w = 0; w < col; w += sizeKernel) {\n" +
                    "               int index_w = j * sizeKernel + index_h;\n" +
                    "               for (int j = 0; j < sizeKernel; j++) {\n" +
                    "                   int rI = (h + j) * depth * col;\n" +
                    "                   int index_j = j * sizeKernel + index_w;\n" +
                    "                   for (int k = 0; k < sizeKernel; k++) {\n" +
                    "                       int cI = (w + k) * depth;\n" +
                    "                       indexInput = rI + cI;\n" +
                    "                       int index_k = k * depth + index_j;\n" +
                    "                       for (int c = 0; c < depth; c++, indexInput++) {\n" +
                    "                           int index_c = c + index_k;\n" +
                    "                           P[1][x][index] = P[0][x][indexInput];\n" +
                    "                       }\n" +
                    "                   }\n" +
                    "               }\n" +
                    "           }\n" +
                    "        }\n" +
                    "        __syncthreads();\n" +
                    "        dot(P[1][x], weight, P[2][x], patch_row, patch_col, weight_row, weight_col);\n" +
                    "    }\n" +
                    "}\n" +*/

                    "__device__ size_t getGlobalIdx_3D_3D()\n" +
                    "{\n" +
                    "    size_t blockId = blockIdx.x + blockIdx.y * gridDim.x\n" +
                    "            + gridDim.x * gridDim.y * blockIdx.z;\n" +
                    "    size_t threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)\n" +
                    "            + (threadIdx.z * (blockDim.x * blockDim.y))\n" +
                    "            + (threadIdx.y * blockDim.x) + threadIdx.x;\n" +
                    "    return threadId;\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void dropout(float* A, float* random, double chanceDrop, int numElements)\n" +
                    "{\n" +
                    "    const int idx = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (idx < numElements) {\n" +
                    "       float drop = (float) (1.0f / (1.0f - chanceDrop));\n" +
                    "       if (random[idx] > chanceDrop)\n" +
                    "       {\n" +
                    "           A[idx] = A[idx] * drop;\n" +
                    "       }\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void sub_gpu(const float* __restrict__ first, const float* __restrict__ second, float* result, int numElements)\n" +
                    "{\n" +
                    "    const int idx = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (idx < numElements) {\n" +
                    "       result[idx] = first[idx] - second[idx];\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void mul(float* result, float val, int numElements)\n" +
                    "{\n" +
                    "    int idx = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (idx < numElements) {\n" +
                    "       result[idx] *= val;\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void indexMaxElement(float* data, float* max_value, int* result, int numElements)\n" +
                    "{\n" +
                    "    int idx = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (idx < numElements) {\n" +
                    "        if (data[idx] > *max_value) {\n" +
                    "            *max_value = data[idx];\n" +
                    "            *result = idx;\n" +
                    "        }\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void derVar_part_1(const float* __restrict__ var, float epsilon, float* dVar, int numElements)\n" +
                    "{\n" +
                    "    int idx = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (idx < numElements) {\n" +
                    "       dVar[idx] = (float) (-0.5f * pow((var[idx] + epsilon), -1.5f));\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void derVar_part_3(const float* __restrict__ dVar, float* derVariance, int numElements)\n" +
                    "{\n" +
                    "    int idx = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (idx < numElements) {\n" +
                    "       derVariance[idx] *= dVar[idx];\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void derMean_part_1(const float* __restrict__ var, float epsilon, float* dMean, int numElements)\n" +
                    "{\n" +
                    "    int idx = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (idx < numElements) {\n" +
                    "       dMean[idx] = (float) (-1.0f / sqrt(var[idx] + epsilon));\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void derMean_part_3(const float* __restrict__ dMean, const float* __restrict__ derVar, const float* __restrict__ dVar, int depth, float* derMean, int numElements)\n" +
                    "{\n" +
                    "    int idx = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (idx < numElements) {\n" +
                    "       float dM = derMean[idx];\n" +
                    "       dM *= dMean[idx];\n" +
                    "       dM += (float) ((-2.0f * derVar[idx] * dVar[idx]) / (depth));\n" +
                    "       derMean[idx] = dM;\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void derNorm_part_1(const float* __restrict__ var, float epsilon, float* dVar, int numElements)\n" +
                    "{\n" +
                    "    int idx = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (idx < numElements) {\n" +
                    "       dVar[idx] = (float) (1.0f / sqrt(var[idx] + epsilon));\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void pow2(float* data, int numElements)\n" +
                    "{\n" +
                    "    int idx = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (idx < numElements) {\n" +
                    "       data[idx] *= data[idx];\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void sum(const float* __restrict__ data, float* sum, int numElements)\n" +
                    "{\n" +
                    "    int idx = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (idx < numElements) {\n" +
                    "       atomicAdd(sum, data[idx]);\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void isnan(const float* __restrict__ data, int* result, int numElements)\n" +
                    "{\n" +
                    "    int idx = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (idx < numElements) {\n" +
                    "       if (result[0] == 0) {\n" +
                    "           if (isnan(data[idx])) {\n" +
                    "               result[0] = 1;\n" +
                    "           }\n" +
                    "       }\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void divide_add(float* A, const float* B, const int C)\n" +
                    "{\n" +
                    "   atomicAdd(A, B[0] / C);\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void momentum(float* data, const float* __restrict__ array, float decay, int numElements)\n" +
                    "{\n" +
                    "    int idx = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (idx < numElements) {\n" +
                    "       data[idx] = decay * data[idx] + array[idx] * (1.0f - decay);\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void momentumPow2(float* data, const float* __restrict__ vector, float decay, int numElements)\n" +
                    "{\n" +
                    "    int idx = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (idx < numElements) {\n" +
                    "       data[idx] = decay * data[idx] + (1.0f - decay) * vector[idx] * vector[idx];\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void subDivSqrtNorm(const float* __restrict__ nominator, const float* __restrict__ denominator, float lr, float normN, float normD, float* data, int numElements)\n" +
                    "{\n" +
                    "    int idx = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (idx < numElements) {\n" +
                    "        atomicAdd(&data[idx], -(lr / (normN + 0.0000001f)) * (nominator[idx]) / (sqrt(denominator[idx] / normD) + 0.0000001f));\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void addBackCopy(const float* __restrict__ matrix, int m_column, int row, int column, int start, float* data)\n" +
                    "{\n" +
                    "    const int x = blockIdx.x * blockDim.x + threadIdx.x;\n" +
                    "    const int y = blockIdx.y * blockDim.y + threadIdx.y;\n" +
                    "    const int index = x * blockDim.y * gridDim.y + y;\n" +
                    "    if (index < row * column) {\n" +
                    "       const int indexOut = x * m_column + start * column + y;\n" +
                    "       data[index] = matrix[indexOut];\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void dropoutBack(const float* __restrict__ output, const float* __restrict__ error, float drop, float* data, int numElements)\n" +
                    "{\n" +
                    "    const int idx = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (idx < numElements) {\n" +
                    "       if (output[idx] != 0) {\n" +
                    "            data[idx] = error[idx] * drop;\n" +
                    "       };\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void mask(const float* __restrict__ A, float val, float newVal, float* C, int numElements)\n" +
                    "{\n" +
                    "    const int i = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (i < numElements) {\n" +
                    "       if (A[i] == val) {\n" +
                    "           C[i] = newVal;\n" +
                    "       }\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void fillUnderDiagonal(int column, float val, float* data, int numElements)\n" +
                    "{\n" +
                    "    int i = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (i < numElements) {\n" +
                    "        for (int j = 0; j < i + 1; j++) {\n" +
                    "            data[i * column + j] = val;\n" +
                    "        }\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void derGelu(const float* __restrict__ input, const float* __restrict__ error, float* data, int numElements)\n" +
                    "{\n" +
                    "    int idx = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (idx < numElements) {\n" +
                    "        float x = input[idx];\n" +
                    "        float val = tanh(0.7978846f * x + 0.0356774f * (x * x * x));\n" +
                    "        data[idx] = (float)(error[idx] * 0.5f * (1.0f + val + x * (1.0f - val * val) * (0.79788846f + 0.1070322f * x * x)));\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void matrixMultiplicationKernel(const float* __restrict__ A, const float* __restrict__ B, float* C, int width, int P, int Q) {\n" +
                    "    int r = blockIdx.y * blockDim.y + threadIdx.y;\n" +
                    "    int c = blockIdx.x * blockDim.x + threadIdx.x;\n" +
                    "    const int index = r * blockDim.y * gridDim.y + c;\n" +
                    "    if (index < P * Q) {\n" +
                    "        float value = (float)(0.0);\n" +
                    "        for(int k = 0; k < width; k++) {\n" +
                    "            value += A[r * width + k] * B[k * Q + c];\n" +
                    "        }\n" +
                    "        C[r * Q + c] = value;\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void Softmax(const float* __restrict__ input, float* data, int column, int numElements)\n" +
                    "{\n" +
                    "    int k = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (k < numElements)\n" +
                    "    {\n" +
                    "       float sum = 0;\n" +
                    "       int index = k * column;\n" +
                    "       float max = input[index];\n" +
                    "       for (int i = 1; i < column; i++, index++) {\n" +
                    "           if (max < input[index])\n" +
                    "               max = input[index];\n" +
                    "       }\n" +
                    "       index = k * column;\n" +
                    "       float E = 2.718281828459045f;\n" +
                    "       for (int i = 0; i < column; i++, index++) {\n" +
                    "           data[index] = (float)(pow(E, input[index] - max));\n" +
                    //"           if (sum + data[index] + 0.00000001 > Float.MAX_VALUE) {\n" +
                    //"               sum = Float.MAX_VALUE;\n" +
                    //"           } else {\n" +
                    "               sum += data[index];\n" +
                    //"           }\n" +
                    "       }\n" +
                    "       sum += 0.00000001f;\n" +
                    "       index = k * column;\n" +
                    "       for (int i = 0; i < column; i++, index++) {\n" +
                    "           data[index] /= sum;\n" +
                    "       }\n" +
                    "    }\n" +
                    "}\n" +

                    /*"const int TILE_DIM_Y = 32;  // Tile dimension for rows\n" +
                    "const int TILE_DIM_X = 32;  // Tile dimension for columns// must be 32 for this method\n" +

                    "extern \"C\"\n" +
                    "__global__ void softmaxKernel2D_rows(const float* input, float* exp_sums, int N, int M) {\n" +
                    "    int row = blockIdx.y * TILE_DIM_Y + threadIdx.y;\n" +
                    "    int col = blockIdx.x * TILE_DIM_X + threadIdx.x;\n" +
                    "    float val = 0;\n" +
                        // Copy data from global memory to shared memory
                    "    if (row < N && col < M) {\n" +
                    "        if (sizeof(float) == 8)\n" +
                    "            val = exp(input[row * M + col]);\n" +
                    "        else\n" +
                    "            val = expf(input[row * M + col]);\n" +
                    "    }\n" +
                        // warp shuffle reduction
                        // Use XOR mode to perform butterfly reduction
                    "    for (int i=16; i>=1; i>>=1)\n" +
                    "        val += __shfl_xor_sync(0xffffffff, val, i, 32);\n" +
                        // update global value for row
                    "    if ((threadIdx.x == 0) && (row < N)) atomicAdd(exp_sums+row, val);\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void softmaxKernel2D_elementwise(const float* input, const float* exp_sums, float* output, int N, int M) {\n" +
                    "    int row = blockIdx.y * TILE_DIM_Y + threadIdx.y;\n" +
                    "    int col = blockIdx.x * TILE_DIM_X + threadIdx.x;\n" +
                        // Compute the softmax values
                    "    if (row < N && col < M) {\n" +
                    "        if (sizeof(float) == 8)\n" +
                    "            output[row * M + col] = exp(input[row * M + col])/ exp_sums[row];\n" +
                    "        else\n" +
                    "            output[row * M + col] = expf(input[row * M + col])/ exp_sums[row];\n" +
                    "    }\n" +
                    "}\n" +*/

                    /*"extern \"C\"\n" +
                    "__global__ void derSoftmax(const float* __restrict__ A, const float* __restrict__ error, float* C, int row, int column)\n" +
                    "{\n" +
                    "    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;\n" +
                    "    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;\n" +
                    //"    const int index = x + y * row + z * row * column;\n" +

                    //"    unsigned int x = threadIdx.x + (blockDim.x * blockIdx.x);  // just like a 2D grid of 2D blocks\n" +
                    //"    unsigned int s = (blockDim.y*blockIdx.y) / column;				 // slice index\n" +
                    //"    unsigned int y = s = 0  ?  threadIdx.y + (blockDim.y * blockIdx.y)  :  ( threadIdx.y +  (blockDim.y*blockIdx.y) ) % column;\n" +
                    //"    unsigned int z = threadIdx.z + (s * blockDim.z);\n" +

                    //"    const int ndx = dX + dY*c.dataX + dZ*c.dataX*c.dataY;
                    //"    const int index = x + y + z * blockDim.x * gridDim.x;\n" +
                    //"    const int gridIndex = blockIdx.y * gridDim.x + blockIdx.x;\n" +
                    //"    const int index = ( gridIndex * blockDim.y + threadIdx.y ) * blockDim.x + threadIdx.x;\n" +
                    "    if ((x < row) && (y < column) && (z < column))\n" +
                    "    {\n" +
                    //"        unsigned int index = x + y * row + z * row * column;\n" +
                    //"        unsigned int index = y + x * column;\n" +
                    //"        unsigned int index2 = z + x * column;\n" +
                    "        const unsigned int indexI = x * column + y;\n" +
                    "        const unsigned int indexJ = x * column + z;\n" +
                    //"        const unsigned int index = x + y * blockDim.x * gridDim.x;\n" +
                    //"        const unsigned int index2 = x + z * blockDim.x * gridDim.x;\n" +
                    "        float value;\n" +
                    "        if (y != z) {\n" +
                    "            value = A[indexI] * -A[indexJ];\n" +
                    "        } \n" +
                    "        else {\n" +
                    "            value = A[indexI] * (((float)1.0) - A[indexI]);\n" +
                    "        }\n" +
                    "        C[indexI] += error[indexJ] * value;\n" +
                    "   }\n" +
                    "}\n" +*/

                     /*int block_id =
                         blockIdx.x +
                         blockIdx.y * gridDim.x +
                         blockIdx.z * gridDim.x + gridDim.y;

                     int block_offset =
                         block_id *
                         blockDim.x * gridDim.y * gridDim.z;

                    int thread_offset =
                        threadIdx.x +
                        threadIdx.y * blockDim.x +
                        threadIdx.z * blockDim.x + blockDim.y;

                    int id = block_offset + thread_offset;*/

                    /*"extern \"C\"\n" +
                    "__global__ void derSoftmax(const float* __restrict__ A, const float* __restrict__ error, float* r, int row, int column)\n" +
                    "{\n" +
                    "    int x = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    int y = blockDim.y * blockIdx.y + threadIdx.y;\n" +
                    "    if (x < row && y < column)\n" +
                    "    {\n" +
                    "        int indexI = x * column + y;\n" +
                    "        float value = 0.0f;\n" +
                    "        float AindexI = A[indexI];\n" +
                    "        int indexJ = x * column;\n" +
                    "        float rV = 0.0f;\n" +
                    "        for (int j = 0; j < column; j++, indexJ++) {\n" +
                    "           if (y != j) {\n" +
                    "               value = (AindexI * -A[indexJ]);\n" +
                    "           } \n" +
                    "           else {\n" +
                    "               value = (AindexI * (1.0f - AindexI));\n" +
                    "           }\n" +
                    "        }\n" +
                    "        atomicAdd(&r[indexI], value * error[indexJ]);\n" +
                    "   }\n" +
                    "}\n";*/

                    /*"__device__ float SharedMemorySize = 110 * 110;\n" +
                    "__device__ extern __shared__ float shared_error[];\n" +
                    "extern \"C\"\n" +
                    "__global__ void derSoftmax(const float* __restrict__ A, const float* __restrict__ error, float* r, int row, int column)\n" +
                    "{\n" +
                    "    int x = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    int y = blockDim.y * blockIdx.y + threadIdx.y;\n" +
                    "    if (x < row)\n" +
                    "    {\n" +
                    "        const int index = x * blockDim.y * gridDim.y + y;\n" +
                    "        if (index < SharedMemorySize) {\n" +
                    "            shared_error[index] = error[index];\n" +
                    "        }\n" +
                    "        __syncthreads();\n" +
                    "        int indexI = x * column + y;\n" +
                    "        float value = 0.0f;\n" +
                    "        float er;\n" +
                    "        float AindexI = A[indexI];\n" +
                    "        int indexJ = x * column;\n" +
                    "        for (int j = 0; j < column; j++, indexJ++) {\n" +
                    "            if (y != j) {\n" +
                    "                value = AindexI * -A[indexJ];\n" +
                    "            } \n" +
                    "            else {\n" +
                    "                value = AindexI * (1.0f - AindexI);\n" +
                    "            }\n" +
                    //"            if (indexJ < SharedMemorySize) {\n" +
                    //"                er = shared_error[indexJ];\n" +
                    //"            }\n" +
                    //"            else {\n" +
                    "                er = error[indexJ];\n" +
                    //"            }\n" +
                    "            atomicAdd(&r[indexI], value * er);\n" +
                    "        }\n" +
                    "   }\n" +
                    "}\n";*/

                    /*"extern \"C\"\n" +
                    __global__ void SoftmaxBwKernel(float *input,float *input_grad,int tgt_len,float *output,float scale)
                    {

                        int tid = threadIdx.x;
                        int base = blockIdx.x * tgt_len;
                        //int baseInput = ;
                        //int baseOutput = blockIdx.x * m ;
                        float outp=0.0f;
                        if (tid < tgt_len)
                            outp = input[base + tid] * input_grad[base + tid];
                        //printf("%d %d %.3f\n",blockIdx.y,tid,ds[tid]);
                        __syncthreads();

                        __shared__ float reduce[TILE_WIDTH];
                        outp += __shfl_xor_sync(WARP_REDUCE_MASK, outp, 16, 32);
                        outp += __shfl_xor_sync(WARP_REDUCE_MASK, outp, 8, 32);
                        outp += __shfl_xor_sync(WARP_REDUCE_MASK, outp, 4, 32);
                        outp += __shfl_xor_sync(WARP_REDUCE_MASK, outp, 2, 32);
                        outp += __shfl_xor_sync(WARP_REDUCE_MASK, outp, 1, 32);
                        __syncthreads();
                        if ((tid & 0x1f) == 0)
                        {
                            reduce[tid>>5] = outp;
                        }
                        __syncthreads();
                        if (tid < (blockDim.x>>5))
                            outp = reduce[tid];
                        else
                            outp = 0.0f;
                        __syncthreads();

                        outp += __shfl_xor_sync(WARP_REDUCE_MASK, outp, 16, 32);
                        outp += __shfl_xor_sync(WARP_REDUCE_MASK, outp, 8, 32);
                        outp += __shfl_xor_sync(WARP_REDUCE_MASK, outp, 4, 32);
                        outp += __shfl_xor_sync(WARP_REDUCE_MASK, outp, 2, 32);
                        outp += __shfl_xor_sync(WARP_REDUCE_MASK, outp, 1, 32);
                        __syncthreads();

                        __shared__ float sum;
                        if (tid==0)
                            sum = outp;
                        __syncthreads();

                        if (tid<tgt_len)
                            output[base + tid] = scale * input[base + tid] * (input_grad[base + tid] - sum);

                    }*/

                    "extern \"C\"\n" +
                    "__global__ void derSoftmax(const float* __restrict__ output, const float* __restrict__ error, float* data, int row, int column)\n" +
                    "{\n" +
                    //"    extern __shared__ float shared[];\n" +
                    "    int k = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    int i = blockDim.y * blockIdx.y + threadIdx.y;\n" +
                    "    int idx = k * blockDim.y * gridDim.y + i;\n" +
                    "    if (k < row && i < column)\n" +
                    "    {\n" +
                    /*"       if (idx < SharedMemorySize) {\n" +
                    "           shared[idx] = error[idx];\n" +
                    "       }\n" +
                    "       __syncthreads();\n" +*/
                    "       k = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "       i = blockDim.y * blockIdx.y + threadIdx.y;\n" +
                    "       float value = 0.0f;\n" +
                    "       int index = k * column;\n" +
                    "       int indexI = index + i;\n" +
                    "       data[indexI] = 0.0f;\n" +
                    "       float o = output[indexI];\n" +
                    "       float sum = 0.0f;\n" +
                    "       for (int j = 0; j < column; j++) {\n" +
                    "           int indexJ = index + j;\n" +
                    "           if (i != j) {\n" +
                    "               value = o * -output[indexJ];\n" +
                    "           } else {\n" +
                    "               value = o * (1.0f - o);\n" +
                    "           }\n" +
                    //"           if (indexJ < SharedMemorySize) {\n" +
                    //"               sum += shared[indexJ] * value;\n" +
                    //"           } else {\n" +
                    "               sum += error[indexJ] * value;\n" +
                    //"           }\n" +
                    "        }\n" +
                    "        data[indexI] = sum;\n" +
                    "    }\n" +
                    "}\n";

                    /*"extern \"C\"\n" +
                    "__global__ void derSoftmax(const float* A, const float* error, float* r, const int row, const int column, const int n)\n" +
                    "{\n" +
                    "    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;\n" +
                    "    unsigned int indexI = x * blockDim.y * gridDim.y + y;\n" +
                    "    if (x < row && y < column)\n" +
                    "    {\n" +
                    "        float value = ((float)(0.0));\n" +
                    "        float temp = ((float)(0.0));\n" +
                    "        float AindexI = A[indexI];\n" +
                    "        int indexJ = x * blockDim.y * gridDim.y;\n" +
                    "        for (int j = 0; j < column; j++, indexJ++) {\n" +
                    "           if (y == j) {\n" +
                    "               value = (float)(AindexI * (((float)1.0) - AindexI));\n" +
                    "           } \n" +
                    "           else {\n" +
                    "               value = AindexI * -A[indexJ];\n" +
                    "           }\n" +
                    "            temp += error[indexJ] * value;\n" +
                    "            r[indexI] += error[indexJ] * value;\n" +
                    "        }\n" +
                    //"        r[indexI] = temp;\n" +
                    "   }\n" +
                    "}\n" */

                    /*"extern \"C\"\n" +
                    "__global__ void derSoftmax(const float* A, const float* error, float* r, const int row, int column, int n)\n" +
                    "{\n" +
                    "    unsigned int y = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    unsigned int z = blockDim.y * blockIdx.y + threadIdx.y;\n" +
                    "    if (y < column && z < column) {\n" +
                    "       for (int k = 0; k < row; k++) {\n" +
                    "           float value = 0;\n" +
                    "           unsigned int indexI = k * blockDim.y * gridDim.y + y;\n" +
                    "           unsigned int indexJ = k * blockDim.y * gridDim.y + z;\n" +
                    "           float AindexI = A[indexI];\n" +
                    "           if (y != z) {\n" +
                    "              value += error[indexJ] * (AindexI * -A[indexJ]);\n" +
                    "           } \n" +
                    "           else {\n" +
                    "              value += error[indexJ] * (AindexI * (((float)1.0) - AindexI));\n" +
                    "           }\n" +
                    "           r[indexI] += error[indexJ] * value;\n" +
                    "       };\n" +
                    "    };\n" +
                    "}\n";*/

}
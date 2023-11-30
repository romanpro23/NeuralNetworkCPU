package nnarrays;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.cudaDataType;
import jcuda.driver.CUfunction;
import jcuda.driver.JCudaDriver;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasStatus;
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
import static jcuda.jcublas.JCublas2.*;
import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.cudaMemcpyKind.*;
import static utilities.GPUInit.*;
import static utilities.GPUInit.allocatedUse;
import static utilities.JCudaHelper.CONTEXT;

@NoArgsConstructor
public class NNArray {
    @Getter
    protected float[] data;
    @Getter
    protected short[] sdata;
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
            this.sdata = new short[size];
        }

        if (Use.GPU) {
            this.data_gpu = new Pointer();
            cudaMalloc(this.data_gpu, (long) size * Sizeof.SHORT);
            cudaMemset(this.data_gpu, 0, (long) size * Sizeof.SHORT);

            allocatedPut();
        }
    }

    public int[] shape() {
        return new int[]{size};
    }

    public NNArray(float[] _data, short[] _sdata) {
        this.size = _data.length;

        if (Use.CPU) {
            this.data = _data;
            this.sdata = _sdata;
        }

        if (Use.GPU) {
            this.data_gpu = new Pointer();
            cudaMalloc(this.data_gpu, (long) Sizeof.SHORT * this.size);
            cudaMemcpy(this.data_gpu, Pointer.to(_sdata), (long) Sizeof.SHORT * this.size, cudaMemcpyHostToDevice);

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
        data[i] = value;
        sdata[i] = Float.floatToFloat16(value);
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
            Pointer kernelParameters = Pointer.to(Pointer.to(data_gpu), Pointer.to(new short[]{Float.floatToFloat16(val)}), Pointer.to(new int[]{n}));
            int blockSize = Math.min(n, BLOCK_SIZE);
            int gridSizeX = (int) Math.ceil((double) n / blockSize);
            cuLaunchKernel(function,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSize, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );

            if (Use.DEBUG_SYNC) {
                JCudaDriver.cuCtxSynchronize();
                IsNan();
            }
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
            if (Use.DEBUG_SYNC)
            {
                JCudaDriver.cuCtxSynchronize();
                IsNan();
            }
        }

        return this;
    }

    public void clip(float val) {
        clip(-val, val);
    }

    public void clip(float min, float max) {
        float a;
        if (Use.CPU) {
            for (int i = 0; i < size; i++) {
                a = data[i];
                if (a > max) {
                    data[i] = max;
                } else if (a < min) {
                    data[i] = min;
                }
            }
        }
        if (Use.GPU) {
            int n = size;
            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, helperModule, "clip");
            Pointer kernelParameters = Pointer.to(Pointer.to(data_gpu), Pointer.to(new short[]{Float.floatToFloat16(min)}), Pointer.to(new short[]{Float.floatToFloat16(max)}), Pointer.to(new int[]{n}));
            int blockSize = Math.min(n, BLOCK_SIZE);
            int gridSizeX = (int) Math.ceil((double) n / blockSize);
            cuLaunchKernel(function,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSize, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );

            if (Use.DEBUG_SYNC) {
                JCudaDriver.cuCtxSynchronize();
                IsNan();
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
            Pointer kernelParameters = Pointer.to(Pointer.to(data_gpu), Pointer.to(new short[]{Float.floatToFloat16(val)}), Pointer.to(new int[]{n}));
            int blockSize = Math.min(n, BLOCK_SIZE);
            int gridSizeX = (int) Math.ceil((double) n / blockSize);
            cuLaunchKernel(function,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSize, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );

            if (Use.DEBUG_SYNC) {
                JCudaDriver.cuCtxSynchronize();
                IsNan();
            }
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
            cudaMemset(data_gpu, 0, (long) size * Sizeof.SHORT);
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
            cudaMemcpy(data_gpu, array.data_gpu, (long)array.size * Sizeof.SHORT, cudaMemcpyDeviceToDevice);

            if (Use.DEBUG_SYNC) {
                JCudaDriver.cuCtxSynchronize();
                array.IsNan(array);
                IsNan();
            }
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
            if (array.size % 2 != 0)
            {
                throw new Exception("Error size for half2 calculation!");
            }

            int n = size / 2;
            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, helperModule, "MatAdd");
            Pointer kernelParameters = Pointer.to(Pointer.to(this.data_gpu), Pointer.to(array.data_gpu), Pointer.to(new int[]{n}));
            int blockSize = Math.min(n, BLOCK_SIZE);
            int gridSizeX = (int) Math.ceil((double) n / blockSize);
            cuLaunchKernel(function,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSize, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
            if (Use.DEBUG_SYNC) {
                JCudaDriver.cuCtxSynchronize();
                IsNan();
                IsNan(array);
            }
        }
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
            Pointer kernelParameters = Pointer.to(Pointer.to(data_gpu), Pointer.to(new short[]{Float.floatToFloat16(value)}), Pointer.to(new int[]{n}));
            int blockSize = Math.min(n, BLOCK_SIZE);
            int gridSizeX = (int) Math.ceil((double) n / blockSize);
            cuLaunchKernel(function,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSize, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );

            if (Use.DEBUG_SYNC) {
                JCudaDriver.cuCtxSynchronize();
                IsNan();
            }
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

            if (Use.DEBUG_SYNC) {
                JCudaDriver.cuCtxSynchronize();
                IsNan(input);
                IsNan();
            }
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

            if (Use.DEBUG_SYNC) {
                JCudaDriver.cuCtxSynchronize();
                IsNan(input);
                IsNan(error);
                IsNan();
            }
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
            int[] maxIndex = new int[1];

            int PType = cudaDataType.CUDA_R_16F;
            int SUCCESS = JCublas2.cublasIamaxEx(cublasHandle, size, data_gpu, PType, 1, Pointer.to(maxIndex));
            if (cublasStatus.CUBLAS_STATUS_SUCCESS != SUCCESS)
            {
                throw new ArithmeticException("Error!");
            }

            if (Use.DEBUG_SYNC) {
                JCudaDriver.cuCtxSynchronize();
                IsNan();
            }

            index = maxIndex[0];
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
            Pointer kernelParameters = Pointer.to(Pointer.to(data_gpu), Pointer.to(array.data_gpu), Pointer.to(new short[]{Float.floatToFloat16(decay)}), Pointer.to(new int[]{p}));
            int blockSize = Math.min(p, BLOCK_SIZE);
            int gridSizeX = (int) Math.ceil((double) p / blockSize);
            cuLaunchKernel(function,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSize, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
            if (Use.DEBUG_SYNC) {
                JCudaDriver.cuCtxSynchronize();
                IsNan(array);
                IsNan();
            }
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
            Pointer kernelParameters = Pointer.to(Pointer.to(data_gpu), Pointer.to(vector.data_gpu), Pointer.to(new short[]{Float.floatToFloat16(decay)}), Pointer.to(new int[]{p}));
            int blockSize = Math.min(p, BLOCK_SIZE);
            int gridSizeX = (int) Math.ceil((double) p / blockSize);
            cuLaunchKernel(function,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSize, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
            if (Use.DEBUG_SYNC) {
                JCudaDriver.cuCtxSynchronize();
                IsNan(vector);
                IsNan();
            }
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
            data[i] -= (float) (lr * nominator.data[i] / (Math.sqrt(denominator.data[i]) + 0.0000001f));
        }
    }

    public void subDivSqrtNorm(NNArray nominator, NNArray denominator, float lr, float normN, float normD) {
        if (Use.CPU) {
            float cur_lr = lr / (normN + 0.0000001f);
            for (int i = 0; i < size; i++) {
                data[i] -= (float) (cur_lr * (nominator.data[i]) / (Math.sqrt(denominator.data[i] / normD) + 0.0000001f));
            }
        }

        if (Use.GPU) {
            int p = size;

            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, helperModule, "subDivSqrtNorm");
            Pointer kernelParameters = Pointer.to(Pointer.to(nominator.data_gpu), Pointer.to(denominator.data_gpu), Pointer.to(new short[]{Float.floatToFloat16(lr)}), Pointer.to(new short[]{Float.floatToFloat16(normN)}), Pointer.to(new short[]{Float.floatToFloat16(normD)}), Pointer.to(data_gpu), Pointer.to(new int[]{p}));
            int blockSize = Math.min(p, BLOCK_SIZE);
            int gridSizeX = (int) Math.ceil((double) p / blockSize);
            cuLaunchKernel(function,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSize, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
            if (Use.DEBUG_SYNC) {
                JCudaDriver.cuCtxSynchronize();
                IsNan(nominator);
                IsNan(denominator);
                IsNan();
            }
        }
    }

    private float absSigmoid(float val) {
        return (float) (1.0f / (1.0f + Math.pow(Math.E, -Math.abs(val))));
    }

    public void subDivSqrtNormDiff(NNArray nominator, NNArray denominator, NNArray der, NNArray derPre, float lr, float normN, float normD) {
        float cur_lr = lr / (normN + 0.0000001f);
        for (int i = 0; i < size; i++) {
            data[i] -= (float) (cur_lr * absSigmoid(derPre.data[i] - der.data[i]) * (nominator.data[i])
                                / (Math.sqrt(denominator.data[i] / normD) + 0.0000001f));
        }
    }

    public void subDivSqrtNorm(NNArray nominator, NNArray denominator, NNArray phi, float lr, float normN, float normD) {
        float cur_lr = lr / (normN + 0.0000001f);
        for (int i = 0; i < size; i++) {
            data[i] -= (float) (cur_lr * phi.data[i] * (nominator.data[i])
                                / (Math.sqrt(denominator.data[i] / normD) + 0.0000001f));
        }
    }

    public void subDivPowNorm(NNArray nominator, NNArray denominator, float lr, float normN, float normD, float p) {
        float cur_lr = lr / (normN + 0.0000001f);
        for (int i = 0; i < size; i++) {
            data[i] -= (float) (cur_lr * (nominator.data[i]) / (Math.pow(denominator.data[i] / normD, p) + 0.0000001f));
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
            data[i] += (float) (cur_lr * (nominator.data[i]) / (Math.sqrt(denominator.data[i] / normD) + 0.0000001f));
        }
    }

    public void subDivSqrtNormQH(NNArray gradient, NNArray nominator, NNArray denominator, float lr, float normN, float normD, float v1, float v2) {
        float cur_lr = lr / (normN + 0.0000001f);
        float v_1 = 1f - v1;
        float v_2 = 1f - v2;
        for (int i = 0; i < size; i++) {
            data[i] -= (float) (cur_lr * (v_1 * gradient.data[i] + v1 * nominator.data[i])
                                / (Math.sqrt(v_2 * gradient.data[i] * gradient.data[i] + v2 * denominator.data[i] / normD) + 0.0000001f));
        }
    }

    public void subDivSqrtNormNesterov(NNArray nominator, NNArray denominator, NNArray grad, float lr, float beta1, float normN, float normD) {
        float bt = (1.0f - beta1) / (normN);
        for (int i = 0; i < size; i++) {
            data[i] -= (float) (lr * (beta1 * nominator.data[i] + bt * grad.data[i]) / (Math.sqrt(denominator.data[i] / normD) + 0.0000001f));
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

    private void dropout_GPU(NNArray A, double chanceDrop) {
        int n = A.size;
        int blockSize = Math.min(n, BLOCK_SIZE);
        int gridSizeX = (int) Math.ceil((double) n / blockSize);
        CUfunction function = new CUfunction();
        float[] randomArray = new float[n];
        if (Use.CPU) {
            for (int i = 0; i < n; i++) {
                randomArray[i] = (float) Math.random();
            }
        }

        short[] randomArray_short = new short[n];
        for (int i = 0; i < n; i++) {
            randomArray_short[i] = Float.floatToFloat16((float) Math.random());
        }

        NNArray randomArrayGPU = new NNArray(randomArray, randomArray_short);

        cuModuleGetFunction(function, helperModule, "dropout");
        Pointer kernelParameters = Pointer.to(Pointer.to(A.data_gpu), Pointer.to(randomArrayGPU.data_gpu), Pointer.to(new short[]{Float.floatToFloat16((float)chanceDrop)}), Pointer.to(new int[]{n}));
        cuLaunchKernel(function,
                gridSizeX, 1, 1,      // Grid dimension
                blockSize, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                kernelParameters, null // Kernel- and extra parameters
        );

        if (Use.DEBUG_SYNC) {
            JCudaDriver.cuCtxSynchronize();
            IsNan(A);
            IsNan(randomArrayGPU);
        }
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

            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, helperModule, "dropoutBack");
            Pointer kernelParameters = Pointer.to(Pointer.to(output.data_gpu), Pointer.to(error.data_gpu), Pointer.to(new short[]{Float.floatToFloat16((float)drop)}), Pointer.to(data_gpu), Pointer.to(new int[]{p}));
            int blockSize = Math.min(p, BLOCK_SIZE);
            int gridSizeX = (int) Math.ceil((double) p / blockSize);
            cuLaunchKernel(function,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSize, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );

            if (Use.DEBUG_SYNC) {
                JCudaDriver.cuCtxSynchronize();
                IsNan(output);
                IsNan(error);
                IsNan();
            }
        }
    }

    public void save(FileWriter writer) throws IOException {
        short[] hostData = null;
        if (Use.GPU) {
            hostData = GetAllHalfValues(data_gpu, size);
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
                    assert hostData != null;
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
            short[] hostdata = new short[array.size];
            for (int i = 0; i < row; i++) {
                int i_index = i * 1024;
                double[] arr = Arrays.stream(scanner.nextLine().split(" ")).mapToDouble(Short::parseShort).toArray();
                for (int j = 0; j < arr.length; j++, i_index++) {
                    hostdata[i_index] = (short) arr[j];
                }
            }
            cudaMemcpy(array.data_gpu, Pointer.to(hostdata), (long) Sizeof.SHORT * array.size, cudaMemcpyHostToDevice);
        }
        return array;
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

    public static short[] GetFirstSingleValueShortStatic(Pointer data_gpu, int n) {
        short[] data_h = new short[n];
        cudaMemcpy(Pointer.to(data_h), data_gpu, (long) n * Sizeof.SHORT, cudaMemcpyDeviceToHost);
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

    public float[] GetFirstSingleValueHalf(Pointer data_gpu, int n) {
        short[] data_h = new short[n];
        float[] data_f = new float[n];
        cudaMemcpy(Pointer.to(data_h), data_gpu, (long) n * Sizeof.SHORT, cudaMemcpyDeviceToHost);
        for (int i = 0; i < n; i++) {
            data_f[i] = Float.float16ToFloat(data_h[i]);
        }
        return data_f;
    }

    public short[] GetAllHalfValues(Pointer data_gpu, int n) {
        short[] data_h = new short[n];
        cudaMemcpy(Pointer.to(data_h), data_gpu, (long) n * Sizeof.SHORT, cudaMemcpyDeviceToHost);
        return data_h;
    }

    public int[] GetFirstSingleValueInt(Pointer data_gpu, int n) {
        int[] data_h = new int[n];
        JCublas2.cublasGetVector(n, Sizeof.INT, data_gpu, 1, Pointer.to(data_h), 1);
        return data_h;
    }

    public void free() {
        if (data_gpu != null) JCuda.cudaFree(data_gpu);
    }

    public boolean IsNan(NNArray data) {
        if (Use.DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();

        int p = data.size;

        int[] result = new int[1];
        Pointer result_gpu = new Pointer();
        cudaMalloc(result_gpu, Sizeof.INT);
        cudaMemset(result_gpu, 0, Sizeof.INT);

        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, helperModule, "fisnan");
        Pointer kernelParameters = Pointer.to(Pointer.to(data.data_gpu), Pointer.to(result_gpu), Pointer.to(new int[]{p}));
        int blockSize = Math.min(p, BLOCK_SIZE);
        int gridSizeX = (int) Math.ceil((double) p / blockSize);
        cuLaunchKernel(function,
                gridSizeX, 1, 1,      // Grid dimension
                blockSize, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                kernelParameters, null // Kernel- and extra parameters
        );

        if (Use.DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();

        JCublas2.cublasGetVector(result.length, Sizeof.INT, result_gpu, 1, Pointer.to(result), 1);

        if (result[0] > 0)
        {
            System.out.println("Error!!");
            return true;
        }

        cudaMemset(result_gpu, 0, Sizeof.INT);

        CUfunction function2 = new CUfunction();
        cuModuleGetFunction(function2, helperModule, "hisinf");
        Pointer kernelParameters2 = Pointer.to(Pointer.to(data.data_gpu), Pointer.to(result_gpu), Pointer.to(new int[]{p}));
        blockSize = Math.min(p, BLOCK_SIZE);
        gridSizeX = (int) Math.ceil((double) p / blockSize);
        cuLaunchKernel(function2,
                gridSizeX, 1, 1,      // Grid dimension
                blockSize, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                kernelParameters2, null // Kernel- and extra parameters
        );

        if (Use.DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();

        JCublas2.cublasGetVector(result.length, Sizeof.INT, result_gpu, 1, Pointer.to(result), 1);

        if (result[0] > 0)
        {
            System.out.println("Error!!");
            return true;
        }

        return false;
    }

    public boolean IsNan() {
        if (Use.DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
        int p = size;

        int[] result = new int[1];
        Pointer result_gpu = new Pointer();
        cudaMalloc(result_gpu, Sizeof.INT);
        cudaMemset(result_gpu, 0, Sizeof.INT);

        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, helperModule, "fisnan");
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
        if (result[0] > 0)
        {
            if (Use.DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
            System.out.println("Error!!");
            return true;
        }

        cudaMemset(result_gpu, 0, Sizeof.INT);

        if (Use.DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();

        CUfunction function2 = new CUfunction();
        cuModuleGetFunction(function2, helperModule, "hisinf");
        Pointer kernelParameters2 = Pointer.to(Pointer.to(data_gpu), Pointer.to(result_gpu), Pointer.to(new int[]{p}));
        blockSize = Math.min(p, BLOCK_SIZE);
        gridSizeX = (int) Math.ceil((double) p / blockSize);
        cuLaunchKernel(function2,
                gridSizeX, 1, 1,      // Grid dimension
                blockSize, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                kernelParameters2, null // Kernel- and extra parameters
        );

        JCublas2.cublasGetVector(result.length, Sizeof.INT, result_gpu, 1, Pointer.to(result), 1);
        if (result[0] > 0)
        {
            if (Use.DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
            System.out.println("Error!!");
            return true;
        }
        if (Use.DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();

        return false;
    }

    public static final String kernels =
                    "#include <cuda_fp16.h>\n" +
                    "#define TYPE half\n" +
                    "#define BLOCK_HEIGHT 1024\n" +
                    "#define BLOCK_WIDTH 64\n" +

                    "__device__ const int SharedMemorySize = 64 * 1024 / 2;\n" +
                    "__device__ const int BLOCK_DIM = 32;\n" +
                    "__device__ __constant__ half sh[13];\n" +

                    "extern \"C\"\n" +
                    "__global__ void fill(half* A, half alpha, int numElements)\n" +
                    "{\n" +
                    "    const int i = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (i < numElements) {\n" +
                    "        A[i] = alpha;\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void gelu(const half* __restrict__ A, half* C, int numElements)\n" +
                    "{\n" +
                    "    const int i = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (i < numElements) {\n" +
                    "        half a = A[i];\n" +
                    "        half t = tanh(sh[1] * a + sh[2] * (a * a * a));\n" +
                    "        C[i] = sh[3] * a * (sh[4] + t);\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void set(half* A, int i, half alpha)\n" +
                    "{\n" +
                    "    A[i] = alpha;\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void MatAdd(half2* A, const half2* __restrict__ B, int numElements)\n" +
                    "{\n" +
                    "    const int k = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (k < numElements) {\n" +
                    "       A[k] = __hadd2(A[k], B[k]);\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void imageVector(const half* __restrict__ A, half* C, int rows, int columns, int depth, int sizeKernel)\n" +
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
                    "__global__ void backImageVector(const half* __restrict__ A, half* C, int rows, int columns, int depth, int sizeKernel)\n" +
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
                    "__global__ void add3(const half* __restrict__ A, half* C, int rows, int columns)\n" +
                    "{\n" +
                    //"    extern __shared__ half shared[];\n" +
                    "    int h = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    int w = blockDim.y * blockIdx.y + threadIdx.y;\n" +
                    "    if (h < rows && w < columns) {\n" +
                    //"       if (w < SharedMemorySize) {\n" +
                    //"           shared[w] = A[w];\n" +
                    //"       }\n" +
                    //"       __syncthreads();\n" +
                    "       int index = h * blockDim.y * gridDim.y + w;\n" +
                    //"       w = blockDim.y * blockIdx.y + threadIdx.y;\n" +
                    //"       if (w < SharedMemorySize) {\n" +
                    //"           C[index] += shared[w];\n" +
                    //"       }\n" +
                    //"       else {\n" +
                    "          C[index] += A[w];\n" +
                    //"       }\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void dot_VectorAndMatrix(TYPE* C, const TYPE* __restrict__ B, const TYPE* __restrict__ A, int rows, int columns)\n" +
                    "{\n" +
                    "    int h = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (h < rows) {\n" +
                    "       TYPE s = sh[0];\n" +
                    "       for (int j = 0; j < columns; j++) {\n" +
                    "           s += B[j] * A[h * columns + j];\n" +
                    "       }\n" +
                    "       C[h] = s;\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void MatMulKernel(TYPE *out, TYPE *in, TYPE *a, const int matrixHeight, const int matrixWidth) {\n" +
                    "    // get variables for loop\n" +
                    "    // copy section of b into shared mem\n" +
                    "    // go through the threads vertically and sum them into a variable\n" +
                    "    // atomic add these variables to the corresponding c index\n" +

                    "    // looping is happening horizontally on the matrix\n" +
                    "    // BLOCK_WIDTH is again horizontal\n" +
                    "    // BLOCK_HEIGHT is going vertical\n" +
                    "    // n / BLOCK_WIDTH blocks horizontally\n" +
                    "    // m / BLOCK_HEIGHT block vertically\n" +

                    "    // get variables for loop\n" +
                    "    // variable for loop length: blockEltHeight\n" +
                    "    __shared__ int blockElt;\n" +
                    "    __shared__ int blockxInd;\n" +
                    "    __shared__ int blockyInd;\n" +
                    "    if (threadIdx.x == 0) {\n" +
                    "        if ((blockIdx.x + 1) * BLOCK_WIDTH <= matrixWidth)\n" +
                    "            blockElt = BLOCK_WIDTH;\n" +
                    "        else blockElt = matrixWidth % BLOCK_WIDTH;\n" +
                    "        blockxInd = blockIdx.x * BLOCK_WIDTH;\n" +
                    "        blockyInd = blockIdx.y * BLOCK_HEIGHT;\n" +
                    "    }\n" +

                    "    __syncthreads();\n" +

                    "    // copy section of b into shared mem\n" +
                    "    // use the first BLOCK_WIDTH of thread\n" +
                    "    __shared__ TYPE b[BLOCK_WIDTH];\n" +

                    "    if (threadIdx.x < blockElt)\n" +
                    "        b[threadIdx.x] = in[blockxInd + threadIdx.x];\n" +

                    "    __syncthreads();\n" +

                    "    // summing variable\n" +
                    "    TYPE cSum = (TYPE) sh[0];\n" +/////!!!!!!!!!!!!!!!!!!!!!
                    "    int threadyInd = blockyInd + threadIdx.x;\n" +

                    "    // make sure we are inside the matrix verticallly\n" +
                    "    if (threadyInd < matrixHeight) {\n" +

                    "        // go through the threads vertically and sum them into a variable\n" +
                    "        for (int i=0; i<blockElt; i++)\n" +
                    "            // A col index   : blockIdx.x * BLOCK_WIDTH + i : blockxInd + i\n" +
                    "            // A row index  : blockIdx.y * BLOCK_HEIGHT + threadIdx.x : blockyInd + threadIdx.x : threadyInd\n" +
                    "            // B index : b[i]\n" +

                    "            // cSum = B index * ( A col index * matrixHeight + A row index)\n" +
                    "            cSum += b[i] * a[(blockxInd + i) * (matrixHeight) + (threadyInd)];\n" +
                                    //printf("csum = %f\n", cSum);

                    "        // atomic add these variables to the corresponding c index\n" +
                    "        atomicAdd(out + threadyInd, cSum);\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void MatMulKernelT(TYPE *out, TYPE *in, TYPE *a, const int matrixHeight, const int matrixWidth) {\n" +
                        // get variables for loop
                        // copy section of b into shared mem
                        // go through the threads vertically and sum them into a variable
                        // atomic add these variables to the corresponding c index

                        // looping is happening vertically on the matrix
                        // BLOCK_WIDTH is going vertical
                        // BLOCK_HEIGHT is going horizontal
                        // m / BLOCK_WIDTH blocks vertically
                        // n / BLOCK_HEIGHT block horizontally

                        // get variables for loop
                        // variable for loop length: blockElt
                    "    __shared__ int blockElt;\n" +
                    "    __shared__ int blockxInd;\n" +
                    "    __shared__ int blockyInd;\n" +
                    "    if (threadIdx.x == 0) {\n" +
                    "        if ((blockIdx.y + 1) * BLOCK_WIDTH <= matrixHeight)\n" +
                    "            blockElt = BLOCK_WIDTH;\n" +
                    "        else blockElt = matrixHeight % BLOCK_WIDTH;\n" +
                    "        blockxInd = blockIdx.x * BLOCK_HEIGHT;\n" +
                    "        blockyInd = blockIdx.y * BLOCK_WIDTH;\n" +
                    "    }\n" +

                    "    __syncthreads();\n" +

                    "    // copy section of b into shared mem\n" +
                    "    // use the first BLOCK_WIDTH of thread\n" +
                    "    __shared__ TYPE b[BLOCK_WIDTH];\n" +

                    "    if (threadIdx.x < blockElt)\n" +
                    "        b[threadIdx.x] = in[blockyInd + threadIdx.x];\n" +

                    "    __syncthreads();\n" +

                    "    // summing variable\n" +
                    "    TYPE cSum = (TYPE) sh[0];\n" +
                    "    int threadxInd = blockxInd + threadIdx.x;\n" +

                    "    // make sure we are inside the array horizontally\n" +
                    "    if (threadxInd < matrixWidth) {\n" +

                    "        // go through the threads vertically and sum them into a variable\n" +
                    "        for (int i=0; i<blockElt; i++)\n" +
                    "            // A col index : blockIdx.x * BLOCK_HEIGHT + threadIdx.x : blockxInd + threadIdx.x : threadxInd\n" +
                    "            // A row index : blockIdx.y * BLOCK_WIDTH + i : blockyInd + i\n" +
                    "            // B index : b[i]\n" +

                    "            // cSum = B index * ( A col index * matrixHeight + A row index)\n" +
                    "            cSum += b[i] * a[(threadxInd) * (matrixHeight) + (blockyInd + i)];\n" +

                    "        // atomic add these variables to the corresponding c index\n" +
                    "        atomicAdd(out + threadxInd , cSum);\n" +
                            //printf("el[%d%d;%d] csum = %f tot = %f\n", blockIdx.x, blockIdx.y, threadIdx.x, cSum, *(out + blockIdx.x * BLOCK_HEIGHT + threadIdx.x));
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void dotT_VectorAndMatrix(const half* __restrict__ A, const half* __restrict__ B, half* C, int rows, int columns)\n" +
                    "{\n" +
                    "    int j = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (j < columns) {\n" +
                    "       half sum = sh[0];\n" +
                    "       for (int i = 0; i < rows; i++) {\n" +
                    "            int index = i * columns + j;\n" +
                    "            sum += A[i] * B[index];\n" +
                    "       }\n" +
                    "       C[j] = sum;\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void derivativeWeight(const half* __restrict__ input, const half* __restrict__ error, half* derWeight, int rows, int columns)\n" +
                    "{\n" +
                    "    const int i = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    const int j = blockDim.y * blockIdx.y + threadIdx.y;\n" +
                    "    if (i < rows && j < columns) {\n" +
                    "       const int index = i * columns + j;\n" +
                    "       half m = error[i] * input[j];\n" +
                    "       half v = derWeight[index] + m;\n" +
                    "       derWeight[index] = v;\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void findMean_part(const half* __restrict__ A, half* C, int width, int depth)\n" +
                    "{\n" +
                    "    int j = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (j < width) {\n" +
                    "       half s = sh[0];\n" +
                    "       int index = j * depth;\n" +
                    "       for (int k = 0; k < depth; k++, index++) {\n" +
                    "           s += A[index];\n" +
                    "       }\n" +
                    "       C[j] = s;\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void generateErrorNorm(const half* __restrict__ errorNL, const half* __restrict__ gamma, half* errorNorm, int width, int depth)\n" +
                    "{\n" +
                    "    const int j = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    const int k = blockDim.y * blockIdx.y + threadIdx.y;\n" +
                    "    const int index = j * depth + k;\n" +
                    "    if (j < width && k < depth) {\n" +
                    "       errorNorm[index] = errorNL[index] * gamma[k];\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void derVar_part_2(const half* __restrict__ error, const half* __restrict__ input, const half* __restrict__ mean, half* derVariance, int width, int depth)\n" +
                    "{\n" +
                    "    const int j = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (j < width) {\n" +
                    "        int index = j * depth;\n" +
                    "        half m = mean[j];\n" +
                    "        half s = sh[0];\n" +
                    "        for (int k = 0; k < depth; k++, index++) {\n" +
                    "           s += error[index] * (input[index] - m);\n" +
                    "        }\n" +
                    "        derVariance[j] = s;\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void derMean_part_2(const half* __restrict__ error, const half* __restrict__ input, const half* __restrict__ mean, half* derMean, half* dVar, int width, int depth)\n" +
                    "{\n" +
                    "    const int j = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (j < width) {\n" +
                    "        half DM = sh[0];\n" +
                    "        half DV = DM;\n" +
                    "        half m = mean[j];\n" +
                    "        for (int k = 0; k < depth; k++) {\n" +
                    "           int index = j * depth + k;\n" +
                    "           DM += error[index];\n" +
                    "           DV += input[index] - m;\n" +
                    "        }\n" +
                    "        derMean[j] = DM;\n" +
                    "        dVar[j] = DV;\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void derNorm_part_2(const half* __restrict__ errors, const half* __restrict__ dVar, const half* __restrict__ errorVar, const half* __restrict__ input, const half* __restrict__ mean, const half* __restrict__ errorMean, half* error, int width, int depth)\n" +
                    "{\n" +
                    "    const int j = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    const int k = blockDim.y * blockIdx.y + threadIdx.y;\n" +
                    "    const int index = j * depth + k;\n" +
                    "    if (j < width && k < depth) {\n" +
                    "        error[index] = errors[index] * dVar[j] + errorVar[j] * (input[index] - mean[j]) + errorMean[j];\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void derivativeWeight_2(const half* __restrict__ error, const half* __restrict__ output, const half* __restrict__ betta, const half* __restrict__ gamma, half* derBetta, half* derGamma, int width, int depth)\n" +
                    "{\n" +
                    "    const int i = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (i < depth) {\n" +
                    "       half dB = derBetta[i];\n" +
                    "       half dG = derGamma[i];\n" +
                    "       half g = gamma[i];\n" +
                    "       half b = betta[i];\n" +
                    "       for (int j = 0; j < width; j++) { \n" +
                    "           int ind = j * depth + i;\n" +
                    "           dB += error[ind];\n" +
                    "           dG += error[ind] * ((output[ind] - b) / g);\n" +
                    "       }\n" +
                    "       derBetta[i] = dB;\n" +
                    "       derGamma[i] = dG;\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void findVariance_part(const half* __restrict__ input, const half* __restrict__ mean, half* var, int width, int depth)\n" +
                    "{\n" +
                    "    const int j = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (j < width) {\n" +
                    "        half s = sh[0];\n" +
                    "        half m = mean[j];\n" +
                    "        int index = j * depth;\n" +
                    "        for (int k = 0; k < depth; k++, index++) {\n" +
                    "           half sub = input[index] - m;\n" +
                    "           s += sub * sub;\n" +
                    "        }\n" +
                    "        var[j] = s;\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void normalization_part_2(const half* __restrict__ input, const half* __restrict__ mean, const half* __restrict__ varSqrt, half* normOutput, const half* __restrict__ gamma, const half* __restrict__ betta, half* output, int width, int depth)\n" +
                    "{\n" +
                    "    int j = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    int k = blockDim.y * blockIdx.y + threadIdx.y;\n" +
                    "    if (j < width && k < depth) {\n" +
                    "        int index = j * depth + k;\n" +
                    "        half nO = (input[index] - mean[j]) / varSqrt[j];\n" +
                    "        output[index] = nO * gamma[k] + betta[k];\n" +
                    "        normOutput[index] = nO;\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void addMatrix(const half* __restrict__ matrix, half* data, int width, int depth)\n" +
                    "{\n" +
                    "    int k = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (k < depth) {\n" +
                    "       half d = data[k];\n" +
                    "       for (int i = 0; i < width; i++) { \n" +
                    "           int	index = i * depth + k;\n" +
                    "           d += matrix[index];\n" +
                    "       }\n" +
                    "       data[k] = d;\n" +
                    "    }\n" +
                    "  }\n" +

                    "extern \"C\"\n" +
                    "__global__ void reverse(half* A, int rows, int columns, int depth)\n" +
                    "{\n" +
                    "    const int i = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    const int j = blockDim.y * blockIdx.y + threadIdx.y;\n" +
                    "    const int k = blockDim.z * blockIdx.z + threadIdx.z;\n" +
                    "    const int index = i * blockDim.y * gridDim.y + j;\n" +
                    "    if (index < rows * columns  && (k < depth))\n" +
                    "    {\n" +
                    "       const int index = rows - 1 - i;\n" +
                    "       half valf = A[i * depth * columns + j * depth + k];\n" +
                    "       half vals = A[index  * depth * columns + j * depth + k];\n" +
                    "       A[i  * depth * columns + j * depth + k] = valf;\n" +
                    "       A[index  * depth * columns + j * depth + k] = vals;\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void set2(half* A, int i, int j, int k, int columns, int depth, half value)\n" +
                    "{\n" +
                    "    A[i * depth * columns + j * depth + k] = value;\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void transpose_naive(half* odata, const half* __restrict__ idata, int width, int height)\n" +
                    "{\n" +
                    "    unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;\n" +

                    "    if (xIndex < width && yIndex < height)\n" +
                    "    {\n" +
                    "        unsigned int index_in  = xIndex + width * yIndex;\n" +
                    "        unsigned int index_out = yIndex + height * xIndex;\n" +
                    "        odata[index_out] = idata[index_in];\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void sharedMem_transpose(half* R, half* M, int rows, int cols){\n" +
                        // fill data into shared memory
                    "    __shared__ half M_Shared[BLOCK_DIM][BLOCK_DIM];\n" +

                    "    int tile_size = BLOCK_DIM;\n" +
                    "    int idx = tile_size * blockIdx.x + threadIdx.x;\n" +
                    "    int idy = tile_size * blockIdx.y + threadIdx.y;\n" +
                    "    int index_in = idx * cols + idy;\n" +
                    "    int index_out = idy * rows + idx;\n" +

                    "    if (idx < rows && idy < cols) {\n" +
                    "        M_Shared[threadIdx.y][threadIdx.x] = M[index_in];\n" +
                    "    }\n" +
                    "    __syncthreads();\n" +

                    "    if(idx < rows && idy < cols){\n" +
                    "        R[index_out] = M_Shared[threadIdx.y][threadIdx.x];\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void matrixTransposeSolveBankConflicts(half* d_b, const half* __restrict__ d_a, int rows, int cols)\n" +
                    "{\n" +
                    "    __shared__ half mat[BLOCK_DIM][BLOCK_DIM + 1];\n" +

                    "    int bx = blockIdx.x * BLOCK_DIM;\n" +
                    "    int by = blockIdx.y * BLOCK_DIM;\n" +

                    "    int i = by + threadIdx.y; int j = bx + threadIdx.x;\n" +
                    "    int ti = bx + threadIdx.y; int tj = by + threadIdx.x;\n" +

                    "    if (i<rows && j<cols)\n" +
                    "       mat[threadIdx.y][threadIdx.x] = d_a[i*cols+j];\n" +

                    "    __syncthreads();\n" +
                    "    if (tj < cols && ti<rows)\n" +
                    "       d_b[ti*rows+tj]=mat[threadIdx.x][threadIdx.y];\n" +
                    "}\n" +
//r[j * rows + i] = m[i * cols + j];
                   /* "extern \"C\"\n" +//r[j * rows + i] = m[i * cols + j];
                    "__global__ void transpose(half *odata, half *idata, int width, int height)\n" +
                    "{\n" +
                    "    __shared__ half block[BLOCK_DIM][BLOCK_DIM+1];\n" +

                        // read the matrix tile into shared memory
                        // load one element per thread from device memory (idata) and store it
                        // in transposed order in block[][]
                    "    unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;\n" +
                    "    unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;\n" +
                    "    if((xIndex < width) && (yIndex < height))\n" +
                    "    {\n" +
                    "        unsigned int index_in = yIndex * width + xIndex;\n" +
                    "        block[threadIdx.y][threadIdx.x] = idata[index_in];\n" +
                    "    }\n" +


                        // synchronise to ensure all writes to block[][] have completed
                    "    __syncthreads();\n" +

                        // write the transposed matrix tile to global memory (odata) in linear order
                    "    xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;\n" +
                    "    yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;\n" +
                    "    if((xIndex < height) && (yIndex < width))\n" +
                    "    {\n" +
                    "        unsigned int index_out = yIndex * height + xIndex;\n" +
                    "        odata[index_out] = block[threadIdx.x][threadIdx.y];\n" +
                    "    }\n" +
                    "}\n" +*/

                    // shared memory CUDA transpose kernel with padding to avoid smem bank conflicts
                    "extern \"C\"\n" +
                    "__global__ void transposeV3(half* AT, const half*  __restrict__ A, int width, int height)" +
                    "{\n" +
                    "  const int idx = threadIdx.x + blockDim.x*blockIdx.x;\n" +
                    "  const int idy = threadIdx.y + blockDim.y*blockIdx.y;\n" +

                    // pad by 1 to avoid 32-width bank-conflicts
                    "  __shared__ half s_A[BLOCK_DIM][BLOCK_DIM+1];\n" +

                    // check this is a legal matrix entry
                    "  if(idx<width && idy<height){\n" +
                    "     s_A[threadIdx.y][threadIdx.x] = A[idx + idy * height];\n" +
                    "  }\n" +

                    // ensure all threads in thread-block finish
                    "  __syncthreads();\n" +

                    // find coordinates of thread in transposed block
                    "  const int idxT = threadIdx.x + blockDim.y*blockIdx.y;\n" +
                    "  const int idyT = threadIdx.y + blockDim.x*blockIdx.x;\n" +

                    // output
                    "        if(idxT < width && idyT < height){\n" +
                    "            AT[idxT + idyT * width] = s_A[threadIdx.x][threadIdx.y];\n" +
                    "        }\n" +
                    "    }\n" +


                    "extern \"C\"\n" +
                    "__global__ void matrixDiv(half* A, half B, int numElements)\n" +
                    "{\n" +
                    "    const int i = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (i < numElements) {\n" +
                    "        A[i] /= B;\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void addCopy(const half* __restrict__ matrix, half* data, int row, int col, int m_col, int start) \n" +
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
                    "__global__ void normalization_part_1(half* A, const half* __restrict__ var, int numElements)\n" +
                    "{\n" +
                    "    const int i = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (i < numElements) {\n" +
                    "       A[i] = hsqrt(var[i] + sh[5]);\n" +
                    "    }\n" +
                    "}\n" +

                    /*"extern \"C\"\n" +
                    "__global__ void NormalizationLayerForward2D(half*** P, const half* __restrict__ gamma, const half* __restrict__ betta, int width, int depth, int n)\n" +
                    "{\n" +
                    "    int x = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    int y = blockDim.y * blockIdx.y + threadIdx.y;\n" +
                    "    if (x < n && y < width) {\n" +
                    "       half mean = P[3][x][y];\n" +
                    "       for (int index = y * depth, int k = 0; k < depth; k++, index++) {\n" +
                    "           mean += P[2][x][index];\n" +
                    "       }\n" +
                    "       P[3][x][y] = mean;\n" +

                    "       half var = P[4][x][y];\n" +
                    "       half sub;\n" +
                    "       for (int index = y * depth, int k = 0; k < depth; k++, index++) {\n" +
                    "           sub = (float)(P[2][x][index] - mean);\n" +
                    "           var += sub * sub;\n" +
                    "       }\n" +
                    "       var = var / depth;\n" +
                    "       P[4][x][y] = var;\n" +

                    "       half varSqrt = (float) (sqrt(var + epsilon));\n" +
                    "       for (int index = y * depth, int k = 0; k < depth; k++, index++) {\n" +
                    "           float nO = ((float)(P[2][x][index] - mean)) / varSqrt;\n" +
                    "           P[0][x][index] = nO;\n" +
                    "           P[1][x][index] = nO * gamma[k] + betta[k];\n" +
                    "       }\n" +
                    "    }\n" +
                    "}\n" +*/

                    "extern \"C\"\n" +
                    "__global__ void DenseLayerForward(half*** P, half* weight, const half* __restrict__ threshold, int row, int column, int n)\n" +
                    "{\n" +
                    "    int x = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    int y = blockDim.y * blockIdx.y + threadIdx.y;\n" +
                    "    if (x < n && y < row) {\n" +
                    "       int index = y * column;\n" +
                    "       half sum = P[1][x][y];\n" +
                    "       for (int j = 0; j < column; j++, index++) {\n" +
                    "           P[1][x][y] += P[0][x][j] * weight[index];\n" +
                    "       }\n" +
                    "       P[1][x][y] = sum;\n" +
                    "       atomicAdd(&P[1][x][y], threshold[y]);\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void dot(half* const __restrict__ a, const half* __restrict__ b, half* result, int row1, int col1, int row2, int col2)\n" +
                    "{\n" +
                    "    int i = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    int j = blockDim.y * blockIdx.y + threadIdx.y;\n" +
                    "    if (i < row1 && j < col2) {\n" +
                    "       half sum = sh[0];\n" +
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
                    "__global__ void dropout(half* A, half* random, half chanceDrop, int numElements)\n" +
                    "{\n" +
                    "    const int idx = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (idx < numElements) {\n" +
                    "       half drop = sh[4] / (sh[4] - chanceDrop);\n" +
                    "       if (random[idx] > chanceDrop)\n" +
                    "       {\n" +
                    "           A[idx] = A[idx] * drop;\n" +
                    "       }\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void sub_gpu(const half2* __restrict__ first, const half2* __restrict__ second, half2* result, int numElements)\n" +
                    "{\n" +
                    "    const int idx = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (idx < numElements) {\n" +
                    "       result[idx] = __hsub2(first[idx], second[idx]);\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void mul(half* result, half val, int numElements)\n" +
                    "{\n" +
                    "    int idx = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (idx < numElements) {\n" +
                    "       result[idx] *= val;\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void clip(half* data, half min, half max, int numElements)\n" +
                    "{\n" +
                    "    int idx = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (idx < numElements) {\n" +
                    "        half a = data[idx];\n" +
                    "        if (a > max) {\n" +
                    "            data[idx] = max;\n" +
                    "        } else if (a < min) {\n" +
                    "            data[idx] = min;\n" +
                    "        }\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void indexMaxElement(half* data, half* max_value, int* result, int numElements)\n" +
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
                    "__global__ void derVar_part_1(const half* __restrict__ var, half* dVar, int numElements)\n" +
                    "{\n" +
                    "    int idx = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (idx < numElements) {\n" +
                    "       dVar[idx] = sh[6] * hexp2(sh[7] * hlog2(var[idx] + sh[5]));\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void derVar_part_3(const half* __restrict__ dVar, half* derVariance, int numElements)\n" +
                    "{\n" +
                    "    int idx = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (idx < numElements) {\n" +
                    "       derVariance[idx] *= dVar[idx];\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void derMean_part_1(const half* __restrict__ var, half* dMean, int numElements)\n" +
                    "{\n" +
                    "    int idx = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (idx < numElements) {\n" +
                    "       dMean[idx] = (sh[8] / hsqrt(var[idx] + sh[5]));\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void derMean_part_3(const half* __restrict__ dMean, const half* __restrict__ derVar, const half* __restrict__ dVar, int depth, half* derMean, int numElements)\n" +
                    "{\n" +
                    "    int idx = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (idx < numElements) {\n" +
                    "       half dM = derMean[idx];\n" +
                    "       dM *= dMean[idx];\n" +
                    "       dM += (sh[9] * derVar[idx] * dVar[idx]) / (__int2half_rn(depth));\n" +
                    "       derMean[idx] = dM;\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void derNorm_part_1(const half* __restrict__ var, half* dVar, int numElements)\n" +
                    "{\n" +
                    "    int idx = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (idx < numElements) {\n" +
                    "       dVar[idx] = sh[4] / hsqrt(var[idx] + sh[5]);\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void pow2(half* data, int numElements)\n" +
                    "{\n" +
                    "    int idx = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (idx < numElements) {\n" +
                    "       half d = data[idx];\n" +
                    "       data[idx] = d * d;\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void fisnan(const half* __restrict__ data, int* result, int numElements)\n" +
                    "{\n" +
                    "    int idx = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (idx < numElements) {\n" +
                    "       if (result[0] == 0) {\n" +
                    "           if (__hisnan(data[idx])) {\n" +
                    "               result[0] = idx;\n" +
                    "           }\n" +
                    "       }\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void hisinf(const half* __restrict__ data, int* result, int numElements)\n" +
                    "{\n" +
                    "    int idx = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (idx < numElements) {\n" +
                    "       if (result[0] == 0) {\n" +
                    "           if (__hisinf(data[idx])) {\n" +
                    "               result[0] = idx;\n" +
                    "           }\n" +
                    "           if (__hisinf(data[idx]) == -1) {\n" +
                    "               result[0] = idx;\n" +
                    "           }\n" +
                    "       }\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void momentum(half* data, const half* __restrict__ array, half decay, int numElements)\n" +
                    "{\n" +
                    "    int idx = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (idx < numElements) {\n" +
                    "       data[idx] = decay * data[idx] + array[idx] * (sh[4] - decay);\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void momentumPow2(half* data, const half* __restrict__ vector, half decay, int numElements)\n" +
                    "{\n" +
                    "    int idx = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (idx < numElements) {\n" +
                    "       data[idx] = decay * data[idx] + (sh[4] - decay) * vector[idx] * vector[idx];\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void subDivSqrtNorm(const half* __restrict__ nominator, const half* __restrict__ denominator, half lr, half normN, half normD, half* data, int numElements)\n" +
                    "{\n" +
                    "    int idx = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (idx < numElements) {\n" +
                    "       half sh5 = sh[5];\n" +
                    "       half cur_lr = lr / (normN +  sh5);\n" +
                    "       data[idx] -= (half)(cur_lr * (nominator[idx]) / (hsqrt(denominator[idx] / normD) +  sh5));\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void addBackCopy(const half* __restrict__ matrix, int m_column, int row, int column, int start, half* data)\n" +
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
                    "__global__ void dropoutBack(const half* __restrict__ output, const half* __restrict__ error, half drop, half* data, int numElements)\n" +
                    "{\n" +
                    "    const int idx = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (idx < numElements) {\n" +
                    "       if (output[idx] != sh[0]) {\n" +
                    "            data[idx] = error[idx] * drop;\n" +
                    "       };\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void mask(const half* __restrict__ A, half val, half newVal, half* C, int numElements)\n" +
                    "{\n" +
                    "    const int i = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (i < numElements) {\n" +
                    "       if (A[i] == val) {\n" +
                    "           C[i] = newVal;\n" +
                    "       }\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void fillUnderDiagonal(int column, half val, half* data, int numElements)\n" +
                    "{\n" +
                    "    int i = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (i < numElements) {\n" +
                    "        for (int j = 0; j < i + 1; j++) {\n" +
                    "            data[i * column + j] = val;\n" +
                    "        }\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void derGelu(const half* __restrict__ input, const half* __restrict__ error, half* data, int numElements)\n" +
                    "{\n" +
                    "    int idx = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (idx < numElements) {\n" +
                    "        half x = input[idx];\n" +
                    "        half val = tanh(sh[1] * x + sh[2] * (x * x * x));\n" +
                    "        data[idx] = error[idx] * sh[3] * (sh[4] + val + x * (sh[4] - val * val) * (sh[10] + sh[11] * x * x));\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void matrixMultiplicationKernel(const half* __restrict__ A, const half* __restrict__ B, half* C, int width, int P, int Q) {\n" +
                    "    int r = blockIdx.y * blockDim.y + threadIdx.y;\n" +
                    "    int c = blockIdx.x * blockDim.x + threadIdx.x;\n" +
                    "    const int index = r * blockDim.y * gridDim.y + c;\n" +
                    "    if (index < P * Q) {\n" +
                    "        half value = __float2half(0.0f);\n" +
                    "        for(int k = 0; k < width; k++) {\n" +
                    "            value += A[r * width + k] * B[k * Q + c];\n" +
                    "        }\n" +
                    "        C[r * Q + c] = value;\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void Softmax(const half* __restrict__ input, half* data, int column, int numElements)\n" +
                    "{\n" +
                    "    int k = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (k < numElements)\n" +
                    "    {\n" +
                    "       half sh0 = sh[0];\n" +
                    "       half sum = sh0;\n" +
                    "       int index = k * column;\n" +
                    "       half max = input[index];\n" +
                    "       for (int i = 1; i < column; i++, index++) {\n" +
                    "           half inx = input[index];\n" +
                    "           if (max < inx)\n" +
                    "               max = inx;\n" +
                    "       }\n" +
                    "       index = k * column;\n" +
                    "       for (int i = 0; i < column; i++, index++) {\n" +
                    "           half d = hexp(input[index] - max);\n" +
                    "           sum += d;\n" +
                    "           data[index] = d;\n" +
                    "       }\n" +
                    "       if (sum = sh0) {\n" +
                    "           sum = sh[5];\n" +
                    "       }\n" +
                    "       if (__hisinf(sum)) {\n" +
                    "           sum = sh[12];\n" +
                    "       }\n" +
                    "       if (__hisinf(sum) == -1) {\n" +
                    "           sum = -sh[12];\n" +
                    "       }\n" +
                    "       index = k * column;\n" +
                    "       for (int i = 0; i < column; i++, index++) {\n" +
                    "           data[index] /= sum;\n" +
                    "       }\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void derSoftmax(const half* __restrict__ output, const half* __restrict__ error, half* data, int row, int column)\n" +
                    "{\n" +
                    //"    extern __shared__ half shared[];\n" +
                    "    int k = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    int i = blockDim.y * blockIdx.y + threadIdx.y;\n" +
                    "    if (k < row && i < column)\n" +
                    "    {\n" +
                    //"       if (idx < SharedMemorySize) {\n" +
                    //"           shared[idx] = error[idx];\n" +
                    //"       }\n" +
                    //"       __syncthreads();\n" +
                    "       k = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "       i = blockDim.y * blockIdx.y + threadIdx.y;\n" +
                    "       half value = sh[0];\n" +
                    "       int index = k * column;\n" +
                    "       int indexI = index + i;\n" +
                    "       data[indexI] = sh[0];\n" +
                    "       half o = output[indexI];\n" +
                    "       half sum = sh[0];\n" +
                    "       for (int j = 0; j < column; j++) {\n" +
                    "           int indexJ = index + j;\n" +
                    "           if (i != j) {\n" +
                    "               value = o * -output[indexJ];\n" +
                    "           } else {\n" +
                    "               value = o * (sh[4] - o);\n" +
                    "           }\n" +
                    //"         if (indexJ < SharedMemorySize) {\n" +
                    //"             sum += shared[indexJ] * value;\n" +
                    //"         }\n" +
                    //"         else {\n" +
                    "             sum += error[indexJ] * value;\n" +
                    //"         }\n" +
                    "        }\n" +
                    "        data[indexI] = sum;\n" +
                    "    }\n" +
                    "}\n";


}
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
import test.speechtotext.PositionLoader;
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
    @Getter
    protected boolean TYPE = false;

    public NNArray(int size) {
        this.size = size;

        if (Use.CPU) {
            this.data = new float[size];
            this.sdata = new short[size];
        }

        if (Use.GPU) {
            this.data_gpu = new Pointer();
            cudaMalloc(this.data_gpu, (long) size * Sizeof.FLOAT);
            cudaMemset(this.data_gpu, 0, (long) size * Sizeof.FLOAT);

            allocatedPut();
        }
    }

    public NNArray(int size, boolean TYPE) {
        this.size = size;

        if (Use.CPU) {
            this.data = new float[size];
            this.sdata = new short[size];
        }

        if (Use.GPU) {
            this.data_gpu = new Pointer();
            this.TYPE = TYPE;

            if (!TYPE) {
                cudaMalloc(this.data_gpu, (long) size * Sizeof.FLOAT);
                cudaMemset(this.data_gpu, 0, (long) size * Sizeof.FLOAT);
            } else {
                cudaMalloc(this.data_gpu, (long) size * Sizeof.SHORT);
                cudaMemset(this.data_gpu, 0, (long) size * Sizeof.SHORT);
            }

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
            cudaMalloc(this.data_gpu, (long) Sizeof.FLOAT * this.size);
            cudaMemcpy(this.data_gpu, Pointer.to(_sdata), (long) Sizeof.FLOAT * this.size, cudaMemcpyHostToDevice);

            allocatedPut();
        }
    }

    public NNArray(float[] _data, short[] _sdata, boolean TYPE) {
        this.size = _data.length;

        if (Use.CPU) {
            this.data = _data;
            this.sdata = _sdata;
        }

        if (Use.GPU) {
            this.TYPE = TYPE;

            this.data_gpu = new Pointer();
            if (!TYPE) {
                cudaMalloc(this.data_gpu, (long) Sizeof.FLOAT * this.size);
                cudaMemcpy(this.data_gpu, Pointer.to(_sdata), (long) Sizeof.FLOAT * this.size, cudaMemcpyHostToDevice);
            }
            else
            {
                cudaMalloc(this.data_gpu, (long) Sizeof.SHORT * this.size);
                cudaMemcpy(this.data_gpu, Pointer.to(_sdata), (long) Sizeof.SHORT * this.size, cudaMemcpyHostToDevice);
            }

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
        sdata[i] = floatToBFloat16(value);
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
            Pointer kernelParameters = null;
            if (!isTYPE()) {
                cuModuleGetFunction(function, helperModule, "matrixDiv_float");
                kernelParameters = Pointer.to(Pointer.to(data_gpu), Pointer.to(new float[]{val}), Pointer.to(new int[]{n}));
            }
            else {
                cuModuleGetFunction(function, helperModule, "matrixDiv");
                kernelParameters = Pointer.to(Pointer.to(data_gpu), Pointer.to(new short[]{floatToBFloat16(val)}), Pointer.to(new int[]{n}));
            }

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
                if (!isTYPE()) {
                    IsNan_float();
                }
                else
                {
                    IsNan();
                }
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
            if (!isTYPE()) {
                cuModuleGetFunction(function, helperModule, "pow2");
            } else {
                cuModuleGetFunction(function, helperModule, "pow2_TYPE");
            }
            Pointer kernelParameters = Pointer.to(Pointer.to(data_gpu), Pointer.to(new int[]{n}));
            int blockSize = Math.min(n, BLOCK_SIZE);
            int gridSizeX = (int) Math.ceil((double) n / blockSize);
            cuLaunchKernel(function,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSize, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
            if (Use.DEBUG_SYNC) {
                if (!isTYPE()) {
                    JCudaDriver.cuCtxSynchronize();
                    IsNan_float();
                } else {
                    JCudaDriver.cuCtxSynchronize();
                    IsNan();
                }
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
            Pointer kernelParameters = null;
            if (!isTYPE()) {
                cuModuleGetFunction(function, helperModule, "clip");
                kernelParameters = Pointer.to(Pointer.to(data_gpu), Pointer.to(new float[]{min}), Pointer.to(new float[]{max}), Pointer.to(new int[]{n}));
            }
            else
            {
                cuModuleGetFunction(function, helperModule, "clip_TYPE");
                kernelParameters = Pointer.to(Pointer.to(data_gpu), Pointer.to(new short[]{floatToBFloat16(min)}), Pointer.to(new short[]{floatToBFloat16(max)}), Pointer.to(new int[]{n}));
            }
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
            Pointer kernelParameters = null;
            if (!isTYPE()) {
                cuModuleGetFunction(function, helperModule, "mul_float");
                kernelParameters = Pointer.to(Pointer.to(data_gpu), Pointer.to(new float[]{val}), Pointer.to(new int[]{n}));
            }
            else
            {
                cuModuleGetFunction(function, helperModule, "mul");
                kernelParameters = Pointer.to(Pointer.to(data_gpu), Pointer.to(new short[]{floatToBFloat16(val)}), Pointer.to(new int[]{n}));
            }
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
                IsNan_float();
            }
        }

        return this;
    }

    public NNArray mul_with_maxValue(float val, float maxValue) {
        if (Use.CPU) {
            for (int i = 0; i < size; i++) {
                float v = data[i] * val;
                if (v < maxValue) {
                    data[i] = v;
                } else {
                    data[i] = maxValue;
                }
            }
        }
        if (Use.GPU) {
            int n = size;
            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, helperModule, "mul_with_maxValue");
            Pointer kernelParameters = Pointer.to(Pointer.to(data_gpu), Pointer.to(new float[]{val}), Pointer.to(new float[]{maxValue}), Pointer.to(new int[]{n}));
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
            if (!TYPE) {
                cudaMemset(data_gpu, 0, (long) size * Sizeof.FLOAT);
            }
            else
            {
                cudaMemset(data_gpu, 0, (long) size * Sizeof.SHORT);
            }
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
            if (!array.TYPE) {
                cudaMemcpy(data_gpu, array.data_gpu, (long) array.size * Sizeof.FLOAT, cudaMemcpyDeviceToDevice);

                if (Use.DEBUG_SYNC) {
                    JCudaDriver.cuCtxSynchronize();
                    array.IsNan_float(array);
                    IsNan_float();
                }
            }
            else {
                cudaMemcpy(data_gpu, array.data_gpu, (long) array.size * Sizeof.SHORT, cudaMemcpyDeviceToDevice);

                if (Use.DEBUG_SYNC) {
                    JCudaDriver.cuCtxSynchronize();
                    array.IsNan(array);
                    IsNan();
                }
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
            int n = size;
            CUfunction function = new CUfunction();
            if (!isTYPE()) {
                cuModuleGetFunction(function, helperModule, "matAdd");
            }
            else
            {
                cuModuleGetFunction(function, helperModule, "matAdd_TYPE");
            }
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
                if (!isTYPE()) {
                    IsNan_float();
                    IsNan_float(array);
                }
                else
                {
                    IsNan();
                    IsNan(array);
                }
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
            Pointer kernelParameters = null;
            if (!TYPE) {
                cuModuleGetFunction(function, helperModule, "fill_float");
                kernelParameters = Pointer.to(Pointer.to(data_gpu), Pointer.to(new float[]{value}), Pointer.to(new int[]{n}));
            }
            else
            {
                cuModuleGetFunction(function, helperModule, "fill");
                kernelParameters = Pointer.to(Pointer.to(data_gpu), Pointer.to(new short[]{floatToBFloat16(value)}), Pointer.to(new int[]{n}));
            }
            int blockSize = Math.min(n, BLOCK_SIZE);
            int gridSizeX = (int) Math.ceil((double) n / blockSize);
            cuLaunchKernel(function,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSize, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );

            if (Use.DEBUG_SYNC) {
                if (!TYPE) {
                    JCudaDriver.cuCtxSynchronize();
                    IsNan_float();
                }
                else
                {
                    JCudaDriver.cuCtxSynchronize();
                    IsNan();
                }
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
            if (!input.TYPE) {
                cuModuleGetFunction(function, helperModule, "gelu");
            }
            else
            {
                cuModuleGetFunction(function, helperModule, "gelu_TYPE");
            }
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
                if (!input.TYPE) {
                    IsNan_float(input);
                    IsNan_float();
                }
                else
                {
                    IsNan(input);
                    IsNan();
                }
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
            if (!input.TYPE) {
                cuModuleGetFunction(function, helperModule, "derGelu");
            }
            else
            {
                cuModuleGetFunction(function, helperModule, "derGelu_TYPE");
            }
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
                if (!input.TYPE) {
                    IsNan_float(input);
                    IsNan_float(error);
                    IsNan_float();
                }
                else
                {
                    IsNan(input);
                    IsNan(error);
                    IsNan();
                }
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
        div((float) Math.sqrt(norm) + 0.00000001f);
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
            //for (int i = 1; i < size; i++)//!!!
            for (int i = 0; i < size; i++) {
                if (max < data[i]) {
                    index = i;
                    max = data[i];
                }
            }
        }
        if (Use.GPU) {
            int[] maxIndex = new int[1];

            Pointer m_gpu = new Pointer();
            cudaMalloc(m_gpu, Sizeof.FLOAT);
            cudaMemset(m_gpu, 0, Sizeof.FLOAT);

            Pointer mId_gpu = new Pointer();
            cudaMalloc(mId_gpu, Sizeof.INT);
            cudaMemset(mId_gpu, 0, Sizeof.INT);


            int n = size;
            CUfunction function = new CUfunction();
            if (!TYPE) {
                cuModuleGetFunction(function, helperModule, "reduceMaxIdxOptimizedShared");
            }
            else
            {
                cuModuleGetFunction(function, helperModule, "reduceMaxIdxOptimizedShared_TYPE");
            }

            Pointer kernelParameters = Pointer.to(Pointer.to(data_gpu), Pointer.to(new int[]{n}), Pointer.to(m_gpu), Pointer.to(mId_gpu));
            int blockSize = Math.min(n, BLOCK_SIZE);
            int gridSizeX = (int) Math.ceil((double) n / blockSize);
            cuLaunchKernel(function,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSize, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );

            cudaMemcpy(Pointer.to(maxIndex), mId_gpu, Sizeof.INT, cudaMemcpyDeviceToHost);

            if (Use.DEBUG_SYNC) {
                JCudaDriver.cuCtxSynchronize();
                if (!TYPE) {
                    IsNan_float();
                }
                else
                {
                    IsNan();
                }
            }

            index = maxIndex[0];
        }

        return index;
    }

    public void float2TYPEVector(NNArray v)
    {
        if (Use.CPU) {
            for (int i = 0; i < size; i++) {
                data[i] = v.data[i];
            }
        }
        if (Use.GPU) {
            if (v.TYPE)
            {
                throw new ExceptionInInitializerError("Error Float!");
            }

            int p = v.size;
            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, helperModule, "float2TYPEVector");
            Pointer kernelParameters = Pointer.to(Pointer.to(v.data_gpu), Pointer.to(data_gpu), Pointer.to(new int[]{p}));
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
                v.IsNan_float(v);
                IsNan();
            }
        }
    }

    public void TYPE2FloatVector(NNArray v)
    {
        if (Use.CPU) {
            if (size >= 0) System.arraycopy(v.data, 0, data, 0, size);
        }
        if (Use.GPU) {
            if (!v.TYPE)
            {
                throw new ExceptionInInitializerError("Error Float!");
            }

            int p = v.size;
            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, helperModule, "TYPE2FloatVector");
            Pointer kernelParameters = Pointer.to(Pointer.to(v.data_gpu), Pointer.to(data_gpu), Pointer.to(new int[]{p}));
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
                v.IsNan(v);
                IsNan_float();
            }
        }
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
                    value = output.data[i] * (0.01f - output.data[i]);
                }
                data[i] += error.getData()[j] * value;
            }
        }
    }

    public void momentum(NNArray array, final float decay) {
        final float rt = 1.0f - decay;

        if (Use.CPU) {
            for (int i = 0; i < size; i++) {
                data[i] = decay * data[i] + array.data[i] * rt;
            }
        }

        if (Use.GPU) {
            int p = size;

            Pointer kernelParameters = null;
            CUfunction function = new CUfunction();
            if (!array.TYPE) {
                cuModuleGetFunction(function, helperModule, "momentum_float");
                kernelParameters = Pointer.to(Pointer.to(data_gpu), Pointer.to(array.data_gpu), Pointer.to(new float[]{decay}), Pointer.to(new float[]{rt}), Pointer.to(new int[]{p}));
            }
            else
            {
                cuModuleGetFunction(function, helperModule, "momentum");
                kernelParameters = Pointer.to(Pointer.to(data_gpu), Pointer.to(array.data_gpu), Pointer.to(new short[]{floatToBFloat16(decay)}), Pointer.to(new short[]{floatToBFloat16(rt)}), Pointer.to(new int[]{p}));
            }
            int blockSize = Math.min(p, BLOCK_SIZE);
            int gridSizeX = (int) Math.ceil((double) p / blockSize);
            cuLaunchKernel(function,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSize, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
            if (!array.TYPE) {
                if (Use.DEBUG_SYNC) {
                    JCudaDriver.cuCtxSynchronize();
                    IsNan_float(array);
                    IsNan_float();
                }
            }
            else
            {
                if (Use.DEBUG_SYNC) {
                    JCudaDriver.cuCtxSynchronize();
                    IsNan(array);
                    IsNan();
                }
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
            data[i] = decay * data[i] + array.data[i] * rt * Math.max(1, e_array.data[i] / (Math.abs(array.data[i]) + 0.00000001f));
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
        final float dr = 1 - decay;

        if (Use.CPU) {
            for (int i = 0; i < size; i++) {
                data[i] = decay * data[i] + dr * vector.data[i] * vector.data[i];
            }
        }

        if (Use.GPU) {
            int p = size;

            CUfunction function = new CUfunction();
            Pointer kernelParameters = null;
            if (!vector.TYPE) {
                cuModuleGetFunction(function, helperModule, "momentumPow2_float");
                kernelParameters = Pointer.to(Pointer.to(data_gpu), Pointer.to(vector.data_gpu), Pointer.to(new float[]{decay}), Pointer.to(new float[]{dr}), Pointer.to(new int[]{p}));
            }
            else
            {
                cuModuleGetFunction(function, helperModule, "momentumPow2");
                kernelParameters = Pointer.to(Pointer.to(data_gpu), Pointer.to(vector.data_gpu), Pointer.to(new short[]{floatToBFloat16(decay)}), Pointer.to(new short[]{floatToBFloat16(dr)}), Pointer.to(new int[]{p}));
            }
            int blockSize = Math.min(p, BLOCK_SIZE);
            int gridSizeX = (int) Math.ceil((double) p / blockSize);
            cuLaunchKernel(function,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSize, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
            if (!vector.TYPE) {
                if (Use.DEBUG_SYNC) {
                    JCudaDriver.cuCtxSynchronize();
                    IsNan_float(vector);
                    IsNan_float();
                }
            }
            else
            {
                if (Use.DEBUG_SYNC) {
                    JCudaDriver.cuCtxSynchronize();
                    IsNan(vector);
                    IsNan();
                }
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
            Pointer kernelParameters = null;
            if (!nominator.TYPE) {
                cuModuleGetFunction(function, helperModule, "subDivSqrtNorm");
                kernelParameters = Pointer.to(Pointer.to(nominator.data_gpu), Pointer.to(denominator.data_gpu), Pointer.to(new float[]{lr}), Pointer.to(new float[]{normN}), Pointer.to(new float[]{normD}), Pointer.to(data_gpu), Pointer.to(new int[]{p}));
            }
            else
            {
                cuModuleGetFunction(function, helperModule, "subDivSqrtNorm_TYPE");
                kernelParameters = Pointer.to(Pointer.to(nominator.data_gpu), Pointer.to(denominator.data_gpu), Pointer.to(new short[]{floatToBFloat16(lr)}), Pointer.to(new short[]{floatToBFloat16(normN)}), Pointer.to(new short[]{floatToBFloat16(normD)}), Pointer.to(data_gpu), Pointer.to(new int[]{p}));
            }
            int blockSize = Math.min(p, BLOCK_SIZE);
            int gridSizeX = (int) Math.ceil((double) p / blockSize);
            cuLaunchKernel(function,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSize, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
            if (!nominator.TYPE) {
                if (Use.DEBUG_SYNC) {
                    JCudaDriver.cuCtxSynchronize();
                    IsNan_float(nominator);
                    IsNan_float(denominator);
                    IsNan_float();
                }
            }
            else
            {
                if (Use.DEBUG_SYNC) {
                    JCudaDriver.cuCtxSynchronize();
                    IsNan(nominator);
                    IsNan(denominator);
                    IsNan();
                }
            }
        }
    }

    private float absSigmoid(float val) {
        return (float) (1.0f / (1.0f + Math.pow(Math.E, -Math.abs(val))));
    }

    public void subDivSqrtNormDiff(NNArray nominator, NNArray denominator, NNArray der, NNArray derPre, float lr, float normN, float normD) {
        float cur_lr = lr / (normN + 0.00000001f);
        for (int i = 0; i < size; i++) {
            data[i] -= (float) (cur_lr * absSigmoid(derPre.data[i] - der.data[i]) * (nominator.data[i])
                    / (Math.sqrt(denominator.data[i] / normD) + 0.00000001f));
        }
    }

    public void subDivSqrtNorm(NNArray nominator, NNArray denominator, NNArray phi, float lr, float normN, float normD) {
        float cur_lr = lr / (normN + 0.00000001f);
        for (int i = 0; i < size; i++) {
            data[i] -= (float) (cur_lr * phi.data[i] * (nominator.data[i])
                    / (Math.sqrt(denominator.data[i] / normD) + 0.00000001f));
        }
    }

    public void subDivPowNorm(NNArray nominator, NNArray denominator, float lr, float normN, float normD, float p) {
        float cur_lr = lr / (normN + 0.00000001f);
        for (int i = 0; i < size; i++) {
            data[i] -= (float) (cur_lr * (nominator.data[i]) / (Math.pow(denominator.data[i] / normD, p) + 0.00000001f));
        }
    }

    public void subDivSqrtNormClip(NNArray nominator, NNArray denominator, float lr, float normN, float normD, float min, float max) {
        float cur_lr = 1 / (normN + 0.00000001f);
        for (int i = 0; i < size; i++) {
            data[i] -= clip((float) (lr / (Math.sqrt(denominator.data[i] / normD) + 0.00000001f)), min, max) * cur_lr * (nominator.data[i]);
        }
    }

    public float clip(float val, float min, float max) {
        if (val < min) {
            return min;
        }
        return Math.min(val, max);
    }

    public void deltaSubDivSqrtNorm(NNArray nominator, NNArray denominator, float lr, float normN, float normD) {
        float cur_lr = lr / (normN + 0.00000001f);
        for (int i = 0; i < size; i++) {
            data[i] += (float) (cur_lr * (nominator.data[i]) / (Math.sqrt(denominator.data[i] / normD) + 0.00000001f));
        }
    }

    public void subDivSqrtNormQH(NNArray gradient, NNArray nominator, NNArray denominator, float lr, float normN, float normD, float v1, float v2) {
        float cur_lr = lr / (normN + 0.00000001f);
        float v_1 = 1f - v1;
        float v_2 = 1f - v2;
        for (int i = 0; i < size; i++) {
            data[i] -= (float) (cur_lr * (v_1 * gradient.data[i] + v1 * nominator.data[i])
                    / (Math.sqrt(v_2 * gradient.data[i] * gradient.data[i] + v2 * denominator.data[i] / normD) + 0.00000001f));
        }
    }

    public void subDivSqrtNormNesterov(NNArray nominator, NNArray denominator, NNArray grad, float lr, float beta1, float normN, float normD) {
        float bt = (1.0f - beta1) / (normN);
        for (int i = 0; i < size; i++) {
            data[i] -= (float) (lr * (beta1 * nominator.data[i] + bt * grad.data[i]) / (Math.sqrt(denominator.data[i] / normD) + 0.00000001f));
        }
    }

    public void subDivNormNesterov(NNArray nominator, NNArray denominator, NNArray grad, float lr, float beta1, float normN) {
        float bt = (1.0f - beta1) / (normN);
        for (int i = 0; i < size; i++) {
            data[i] -= lr * (beta1 * nominator.data[i] + bt * grad.data[i]) / (denominator.data[i] + 0.00000001f);
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
            result.data[i] = (float) (data[i] * Math.sqrt(nominator.data[i] + 0.00000001f) / (Math.sqrt(denominator.data[i]) + 0.00000001f));
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
            randomArray_short[i] = floatToBFloat16((float) Math.random());
        }

        NNArray randomArrayGPU = new NNArray(randomArray, randomArray_short, TYPE);

        cuModuleGetFunction(function, helperModule, "dropout");
        Pointer kernelParameters = Pointer.to(Pointer.to(A.data_gpu), Pointer.to(randomArrayGPU.data_gpu), Pointer.to(new short[]{floatToBFloat16((float) chanceDrop)}), Pointer.to(new int[]{n}));
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
            Pointer kernelParameters = Pointer.to(Pointer.to(output.data_gpu), Pointer.to(error.data_gpu), Pointer.to(new short[]{floatToBFloat16((float) drop)}), Pointer.to(data_gpu), Pointer.to(new int[]{p}));
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
        float[] hostData = null;
        short[] hostData_TYPE = null;
        if (Use.GPU) {
            if (!TYPE) {
                hostData = GetFirstSingleValueFloat(data_gpu, size);
            }
            else
            {
                hostData_TYPE = GetAllTYPEValues(data_gpu, size);
            }
        }
        writer.write(size + "\n");
        writer.write(TYPE + "\n");
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
                    if (!TYPE) {
                        assert hostData != null;
                        writer.write(hostData[i_index] + " ");
                    }
                    else
                    {
                        assert hostData_TYPE != null;
                        writer.write(hostData_TYPE[i_index] + " ");
                    }
                }
            }
            writer.write("\n");
            writer.flush();
        }
        writer.flush();
    }

    public static NNArray read(Scanner scanner) {
        NNArray array = new NNArray(Integer.parseInt(scanner.nextLine()), Boolean.parseBoolean(scanner.nextLine()));
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
            if (!array.TYPE) {
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
            else
            {
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

    public String GetFirstSingleValue(PositionLoader loader, int n) {
        float[] data_h = new float[n];
        cudaMemcpy(Pointer.to(data_h), data_gpu, (long) n * Sizeof.FLOAT, cudaMemcpyDeviceToHost);
        return loader.decodeString_new(data_h);
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

    public float[] GetFirstSingleValueTYPE(Pointer data_gpu, int n) {
        short[] data_h = new short[n];
        float[] data_f = new float[n];
        cudaMemcpy(Pointer.to(data_h), data_gpu, (long) n * Sizeof.SHORT, cudaMemcpyDeviceToHost);
        for (int i = 0; i < n; i++) {
            data_f[i] = bFloat16ToFloat(data_h[i]);
        }
        return data_f;
    }

    public short[] GetAllTYPEValues(Pointer data_gpu, int n) {
        short[] data_h = new short[n];
        cudaMemcpy(Pointer.to(data_h), data_gpu, (long) n * Sizeof.SHORT, cudaMemcpyDeviceToHost);
        return data_h;
    }

    public int[] GetFirstSingleValueInt(Pointer data_gpu, int n) {
        int[] data_h = new int[n];
        JCublas2.cublasGetVector(n, Sizeof.INT, data_gpu, 1, Pointer.to(data_h), 1);
        return data_h;
    }

    public static float bFloat16ToFloat(short bits) {
        return Float.intBitsToFloat(Short.toUnsignedInt(bits) << 16);
    }

    public static short floatToBFloat16(float f) {
        int bits = Float.floatToIntBits(f);
        // This does not implement rounding correctly but is a good first approximation for this benchmark.
        return (short) ((bits + 0x7FFF) >>> 16);
    }

    public void free() {
        if (data_gpu != null) JCuda.cudaFree(data_gpu);
    }

    public boolean IsNan(NNArray data) {
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

        JCublas2.cublasGetVector(result.length, Sizeof.INT, result_gpu, 1, Pointer.to(result), 1);

        if (result[0] > 0) {
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

        JCublas2.cublasGetVector(result.length, Sizeof.INT, result_gpu, 1, Pointer.to(result), 1);

        if (result[0] > 0) {
            System.out.println("Error!!");
            return true;
        }

        return false;
    }

    public boolean IsNan_float(NNArray data) {
        int p = data.size;

        int[] result = new int[1];
        Pointer result_gpu = new Pointer();
        cudaMalloc(result_gpu, Sizeof.INT);
        cudaMemset(result_gpu, 0, Sizeof.INT);

        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, helperModule, "fisnan_float");
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

        if (result[0] > 0) {
            System.out.println("Error!!");
            return true;
        }

        cudaMemset(result_gpu, 0, Sizeof.INT);

        CUfunction function2 = new CUfunction();
        cuModuleGetFunction(function2, helperModule, "hisinf_float");
        Pointer kernelParameters2 = Pointer.to(Pointer.to(data.data_gpu), Pointer.to(result_gpu), Pointer.to(new int[]{p}));
        blockSize = Math.min(p, BLOCK_SIZE);
        gridSizeX = (int) Math.ceil((double) p / blockSize);
        cuLaunchKernel(function2,
                gridSizeX, 1, 1,      // Grid dimension
                blockSize, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                kernelParameters2, null // Kernel- and extra parameters
        );

        JCublas2.cublasGetVector(result.length, Sizeof.INT, result_gpu, 1, Pointer.to(result), 1);

        if (result[0] > 0) {
            System.out.println("Error!!");
            return true;
        }

        return false;
    }

    public boolean IsNan() {
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
        if (result[0] > 0) {
            System.out.println("Error!!");
            return true;
        }

        cudaMemset(result_gpu, 0, Sizeof.INT);

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
        if (result[0] > 0) {
            System.out.println("Error!!");
            return true;
        }

        return false;
    }

    public boolean IsNan_float() {
        int p = size;

        int[] result = new int[1];
        Pointer result_gpu = new Pointer();
        cudaMalloc(result_gpu, Sizeof.INT);
        cudaMemset(result_gpu, 0, Sizeof.INT);

        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, helperModule, "fisnan_float");
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
        if (result[0] > 0) {
            System.out.println("Error!!");
            return true;
        }

        cudaMemset(result_gpu, 0, Sizeof.INT);

        CUfunction function2 = new CUfunction();
        cuModuleGetFunction(function2, helperModule, "hisinf_float");
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
        if (result[0] > 0) {
            System.out.println("Error!!");
            return true;
        }

        return false;
    }

    public static final String kernels =
                    "#include <cuda_bf16.h>\n" +
                    "#include <float.h>\n" +
                    "#define TYPE __nv_bfloat16\n" +
                    "#define BLOCK_HEIGHT 1024\n" +
                    "#define BLOCK_WIDTH 64\n" +

                    "__device__ const int SharedMemorySize = 64 * 1024 / 2;\n" +
                    "__device__ const int BLOCK_DIM = 32;\n" +
                    "__device__ const int TILE_WIDTH = 32;\n" +
                    "__device__ __constant__ TYPE sh[17];\n" +

                    "__inline__ __device__ float InfinityCheck(float v)\n" +
                    "{\n" +
                    "    int r = __isinf(v);\n" +
                    "    if (r == 1) {\n" +
                    "        v = FLT_MAX;\n" +
                    "    }\n" +
                    "    else if (r == -1) {\n" +
                    "        v = -FLT_MAX;\n" +
                    "    }\n" +
                    "    return v;\n" +
                    "}\n" +

                    "__inline__ __device__ float InfinityCheck_TYPE(TYPE v)\n" +
                    "{\n" +
                    "    int r = __isinf(__bfloat162float(v));\n" +
                    "    if (r == 1) {\n" +
                    "        v = __float2bfloat16(FLT_MAX);\n" +
                    "    }\n" +
                    "    else if (r == -1) {\n" +
                    "        v = __float2bfloat16(-FLT_MAX);\n" +
                    "    }\n" +
                    "    return v;\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void fill(TYPE* A, TYPE alpha, int numElements)\n" +
                    "{\n" +
                    "    const int i = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (i < numElements) {\n" +
                    "        A[i] = alpha;\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void fill_float(float* A, float alpha, int numElements)\n" +
                    "{\n" +
                    "    const int i = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (i < numElements) {\n" +
                    "        A[i] = alpha;\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void float2TYPEVector(float* v, TYPE* data, int numElements)\n" +
                    "{\n" +
                    "    const int i = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (i < numElements) {\n" +
                    "        data[i] = __float2bfloat16(v[i]);\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void TYPE2FloatVector(TYPE* v, float* data, int numElements)\n" +
                    "{\n" +
                    "    const int i = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (i < numElements) {\n" +
                    "        data[i] = __bfloat162float(v[i]);\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void gelu(const float* __restrict__ A, float* C, int numElements)\n" +
                    "{\n" +
                    "    const int i = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (i < numElements) {\n" +
                    "        float a = A[i];\n" +
                    "        float t = tanhf(0.7978846f * a + 0.0356774f * (a * a * a));\n" +
                    "        C[i] = 0.5f * a * (1.0f + t);\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void gelu_TYPE(const TYPE* __restrict__ A, TYPE* C, int numElements)\n" +
                    "{\n" +
                    "    const int i = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (i < numElements) {\n" +
                    "        float a = __bfloat162float(A[i]);\n" +
                    "        float t = tanhf(0.7978846f * a + 0.0356774f * (a * a * a));\n" +
                    "        C[i] = __float2bfloat16(0.5f * a * (1.0f + t));\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void matAdd(float* A, const float* __restrict__ B, int numElements)\n" +
                    "{\n" +
                    "    const int k = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (k < numElements) {\n" +
                    "       A[k] = A[k] + B[k];\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void matAdd_TYPE(TYPE* A, const TYPE* __restrict__ B, int numElements)\n" +
                    "{\n" +
                    "    const int k = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (k < numElements) {\n" +
                    "       A[k] = A[k] + B[k];\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void matAdd_(float* A, const float* __restrict__ B, int numElements)\n" +
                    "{\n" +
                    "    const int k = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (k < numElements) {\n" +
                    "       A[k] = A[k] + B[k];\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void matAdd_TYPE_(TYPE* A, const TYPE* __restrict__ B, int numElements)\n" +
                    "{\n" +
                    "    const int k = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (k < numElements) {\n" +
                    "       A[k] = A[k] + B[k];\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void imageVector(const TYPE* __restrict__ A, TYPE* C, int rows, int columns, int depth, int sizeKernel)\n" +
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
                    "__global__ void backImageVector(const TYPE* __restrict__ A, TYPE* C, int rows, int columns, int depth, int sizeKernel)\n" +
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
                    "    int h = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    int w = blockDim.y * blockIdx.y + threadIdx.y;\n" +
                    "    if (h < rows && w < columns) {\n" +
                    "       int index = h * columns + w;\n" +
                    "       C[index] = C[index] + A[w];\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void add3_TYPE(const TYPE* __restrict__ A, TYPE* C, int rows, int columns)\n" +
                    "{\n" +
                    "    int h = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    int w = blockDim.y * blockIdx.y + threadIdx.y;\n" +
                    "    if (h < rows && w < columns) {\n" +
                    "       int index = h * columns + w;\n" +
                    "       C[index] = C[index] + A[w];\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void dotT_VectorAndMatrix(const float* __restrict__ A, const float* __restrict__ B, float* C, int rows, int columns)\n" +
                    "{\n" +
                    "    int j = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (j < columns) {\n" +
                    "       float sum = sh[0];\n" +
                    "       for (int i = 0; i < rows; i++) {\n" +
                    "            int index = i * columns + j;\n" +
                    "            sum = sum + A[i] * B[index];\n" +
                    "       }\n" +
                    "       C[j] = sum;\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void dotT_VectorAndMatrix_TYPE(const TYPE* __restrict__ A, const TYPE* __restrict__ B, TYPE* C, int rows, int columns)\n" +
                    "{\n" +
                    "    int j = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (j < columns) {\n" +
                    "       TYPE sum = sh[0];\n" +
                    "       for (int i = 0; i < rows; i++) {\n" +
                    "            int index = i * columns + j;\n" +
                    "            sum = sum + A[i] * B[index];\n" +
                    "       }\n" +
                    "       C[j] = sum;\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void derivativeWeight(const float* __restrict__ input, const float* __restrict__ error, float* derWeight, int rows, int columns)\n" +
                    "{\n" +
                    "    const int i = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    const int j = blockDim.y * blockIdx.y + threadIdx.y;\n" +
                    "    if (i < rows && j < columns) {\n" +
                    "       const int index = i * columns + j;\n" +
                    "       derWeight[index] = derWeight[index] + error[i] * input[j];\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void derivativeWeight_TYPE(const TYPE* __restrict__ input, const TYPE* __restrict__ error, TYPE* derWeight, int rows, int columns)\n" +
                    "{\n" +
                    "    const int i = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    const int j = blockDim.y * blockIdx.y + threadIdx.y;\n" +
                    "    if (i < rows && j < columns) {\n" +
                    "       const int index = i * columns + j;\n" +
                    "       derWeight[index] = derWeight[index] + error[i] * input[j];\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void addMatrix(const float* __restrict__ matrix, float* data, int width, int depth)\n" +
                    "{\n" +
                    "    int k = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (k < depth) {\n" +
                    "       float d = data[k];\n" +
                    "       for (int i = 0; i < width; i++) { \n" +
                    "           int	index = i * depth + k;\n" +
                    "           d = d + matrix[index];\n" +
                    "       }\n" +
                    "       data[k] = d;\n" +
                    "    }\n" +
                    "  }\n" +

                    "extern \"C\"\n" +
                    "__global__ void addMatrix_TYPE(const TYPE* __restrict__ matrix, TYPE* data, int width, int depth)\n" +
                    "{\n" +
                    "    int k = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (k < depth) {\n" +
                    "       TYPE d = data[k];\n" +
                    "       for (int i = 0; i < width; i++) { \n" +
                    "           int	index = i * depth + k;\n" +
                    "           d = d + matrix[index];\n" +
                    "       }\n" +
                    "       data[k] = d;\n" +
                    "    }\n" +
                    "  }\n" +

                    //MIT License
                    //Copyright (c) 2019 Apriorit Inc.
                    //////////////////////////////////
                    "__device__ void atomicMax(float* const address, const float value)\n" +
                    "{\n" +
                    "    if (*address >= value)\n" +
                    "    {\n" +
                    "        return;\n" +
                    "    }\n" +

                    "    int* const addressAsI = (int*)address;\n" +
                    "    int old = *addressAsI, assumed;\n" +

                    "    do\n" +
                    "    {\n" +
                    "        assumed = old;\n" +
                    "        if (__int_as_float(assumed) >= value)\n" +
                    "        {\n" +
                    "            break;\n" +
                    "        }\n" +

                    "        old = atomicCAS(addressAsI, assumed, __float_as_int(value));\n" +
                    "    } while (assumed != old);\n" +
                    "}\n" +

                    "__device__ void atomicMax(TYPE* const address, const TYPE value)\n" +
                    "{\n" +
                    "    if (*address >= value)\n" +
                    "    {\n" +
                    "        return;\n" +
                    "    }\n" +

                    "    int* const addressAsI = (int*)address;\n" +
                    "    int old = *addressAsI, assumed;\n" +

                    "    do\n" +
                    "    {\n" +
                    "        assumed = old;\n" +
                    "        if (__float2bfloat16(__int_as_float(assumed)) >= value)\n" +
                    "        {\n" +
                    "            break;\n" +
                    "        }\n" +

                    "        old = atomicCAS(addressAsI, assumed, __float_as_int(__bfloat162float(value)));\n" +
                    "    } while (assumed != old);\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void reduceMaxIdxOptimizedShared(const float* __restrict__ input, const int size, float* maxOut, int* maxIdxOut)\n" +
                    "{\n" +
                    "    __shared__ float sharedMax;\n" +
                    "    __shared__ int sharedMaxIdx;\n" +

                    "    if (0 == threadIdx.x)\n" +
                    "    {\n" +
                    "        sharedMax = 0.f;\n" +
                    "        sharedMaxIdx = 0;\n" +
                    "    }\n" +

                    "    __syncthreads();\n" +

                    "    float localMax = 0.f;\n" +
                    "    int localMaxIdx = 0;\n" +

                    "    for (int i = threadIdx.x; i < size; i += blockDim.x)\n" +
                    "    {\n" +
                    "        float val = input[i];\n" +

                    //"        if (localMax < abs(val))\n" +
                    "        if (localMax < val)\n" +
                    "        {\n" +
                    //"            localMax = abs(val);\n" +
                    "            localMax = val;\n" +
                    "            localMaxIdx = i;\n" +
                    "        }\n" +
                    "    }\n" +

                    "    atomicMax(&sharedMax, localMax);\n" +

                    "    __syncthreads();\n" +

                    "    if (sharedMax == localMax)\n" +
                    "    {\n" +
                    "        sharedMaxIdx = localMaxIdx;\n" +
                    "    }\n" +

                    "    __syncthreads();\n" +

                    "    if (0 == threadIdx.x)\n" +
                    "    {\n" +
                    "        *maxOut = sharedMax;\n" +
                    "        *maxIdxOut = sharedMaxIdx;\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void reduceMaxIdxOptimizedShared_TYPE(const TYPE* __restrict__ input, const int size, TYPE* maxOut, int* maxIdxOut)\n" +
                    "{\n" +
                    "    __shared__ TYPE sharedMax;\n" +
                    "    __shared__ int sharedMaxIdx;\n" +

                    "    if (0 == threadIdx.x)\n" +
                    "    {\n" +
                    "        sharedMax = sh[0];\n" +
                    "        sharedMaxIdx = 0;\n" +
                    "    }\n" +

                    "    __syncthreads();\n" +

                    "    TYPE localMax = sh[0];\n" +
                    "    int localMaxIdx = 0;\n" +

                    "    for (int i = threadIdx.x; i < size; i += blockDim.x)\n" +
                    "    {\n" +
                    "        TYPE val = input[i];\n" +

                    //"        if (localMax < abs(val))\n" +
                    "        if (localMax < val)\n" +
                    "        {\n" +
                    //"            localMax = abs(val);\n" +
                    "            localMax = val;\n" +
                    "            localMaxIdx = i;\n" +
                    "        }\n" +
                    "    }\n" +

                    "    atomicMax(&sharedMax, localMax);\n" +

                    "    __syncthreads();\n" +

                    "    if (sharedMax == localMax)\n" +
                    "    {\n" +
                    "        sharedMaxIdx = localMaxIdx;\n" +
                    "    }\n" +

                    "    __syncthreads();\n" +

                    "    if (0 == threadIdx.x)\n" +
                    "    {\n" +
                    "        *maxOut = sharedMax;\n" +
                    "        *maxIdxOut = sharedMaxIdx;\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void reverse(TYPE* A, int rows, int columns, int depth)\n" +
                    "{\n" +
                    "    const int i = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    const int j = blockDim.y * blockIdx.y + threadIdx.y;\n" +
                    "    const int k = blockDim.z * blockIdx.z + threadIdx.z;\n" +
                    "    const int index = i * blockDim.y * gridDim.y + j;\n" +
                    "    if (index < rows * columns  && (k < depth))\n" +
                    "    {\n" +
                    "       const int index = rows - 1 - i;\n" +
                    "       TYPE valf = A[i * depth * columns + j * depth + k];\n" +
                    "       TYPE vals = A[index  * depth * columns + j * depth + k];\n" +
                    "       A[i  * depth * columns + j * depth + k] = valf;\n" +
                    "       A[index  * depth * columns + j * depth + k] = vals;\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void sharedMem_transpose(float* R, float* M, int rows, int cols){\n" +
                    // fill data into shared memory
                    "    __shared__ float M_Shared[BLOCK_DIM][BLOCK_DIM];\n" +

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
                    "__global__ void sharedMem_transpose_TYPE(TYPE* R, TYPE* M, int rows, int cols){\n" +
                    // fill data into shared memory
                    "    __shared__ TYPE M_Shared[BLOCK_DIM][BLOCK_DIM];\n" +

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
                    "__global__ void matrixDiv(TYPE* A, TYPE B, int numElements)\n" +
                    "{\n" +
                    "    const int i = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (i < numElements) {\n" +
                    "        A[i] = A[i] / B;\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void matrixDiv_float(float* A, float B, int numElements)\n" +
                    "{\n" +
                    "    const int i = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (i < numElements) {\n" +
                    "        A[i] /= B;\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void addCopy(const float* __restrict__ matrix, float* data, int row, int col, int m_col, int start) \n" +
                    "{\n" +
                    "    const int x = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    const int y = blockDim.y * blockIdx.y + threadIdx.y;\n" +
                    "    if (x < row && y < m_col)\n" +
                    "    {\n" +
                    "        const int indexIn = x * col + start * m_col + y;\n" +
                    "        const int indexOut = x * m_col + y;\n" +
                    "        data[indexIn] = matrix[indexOut];\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void addCopy_TYPE(const TYPE* __restrict__ matrix, TYPE* data, int row, int col, int m_col, int start) \n" +
                    "{\n" +
                    "    const int x = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    const int y = blockDim.y * blockIdx.y + threadIdx.y;\n" +
                    "    if (x < row && y < m_col)\n" +
                    "    {\n" +
                    "        const int indexIn = x * col + start * m_col + y;\n" +
                    "        const int indexOut = x * m_col + y;\n" +
                    "        data[indexIn] = matrix[indexOut];\n" +
                    "    }\n" +
                    "}\n" +

                    ////////////////////////////////////////////////////////////
                    //среднее_mean = X.sum () / n
                    //std = ((( X-mean)** 2 ) .sum () / n).sqrt()
                    //z_scores = (X - среднее_mean) / std
                    ////////////////////////////////////////////////////////////
                    "extern \"C\"\n" +
                    "__global__ void NormalizationLayerForward2D(float*** P, const float* __restrict__ gamma, const float* __restrict__ betta, int width, int depth, int n)\n" +
                    "{\n" +
                    "    int x = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    int y = blockDim.y * blockIdx.y + threadIdx.y;\n" +
                    "    if (x < n && y < width) {\n" +
                    "        float* input = P[0][x];\n" +
                    "        float mean = P[1][x][y];\n" +
                    //       საშუალო მნიშვნელობის გამოთვლა
                    //       find mean
                    ////////////////////////////////////////////////////////////
                    "        int index = y * depth;\n" +
                    "        for (int k = 0; k < depth; k++, index++) {\n" +
                    "           mean = mean + input[index];\n" +
                    "        }\n" +
                    "        mean = mean / depth;\n" +
                    "        P[1][x][y] = mean;\n" +
                    ////////////////////////////////////////////////////////////
                    //       find variance
                    ////////////////////////////////////////////////////////////
                    "        float var = P[2][x][y];\n" +
                    "        float sub;\n" +
                    "        index = y * depth;\n" +
                    "        mean = P[1][x][y];\n" +
                    "        for (int k = 0; k < depth; k++, index++) {\n" +
                    "            sub = input[index] - mean;\n" +
                    "            var = var + sub * sub;\n" +
                    "        }\n" +
                    "        var = var / depth;\n" +
                    "        P[2][x][y] = var;\n" +
                    ////////////////////////////////////////////////////////////
                    //       Normalization
                    ////////////////////////////////////////////////////////////
                    "        float varSqrt = sqrtf(var + 0.001f);\n" +
                    "        float* output = P[3][x];\n" +
                    "        index = y * depth;\n" +
                    "        for (int k = 0; k < depth; k++, index++) {\n" +
                    "             output[index] = ((input[index] - mean) / varSqrt) * gamma[k] + betta[k];\n" +
                    "        }\n" +
                    ////////////////////////////////////////////////////////////
                    "    }\n" +
                    "}\n" +

                    /*"extern \"C\"\n" +
                    "__global__ void NormalizationLayerForward_TYPE_2D(TYPE*** P, const TYPE* __restrict__ gamma, const TYPE* __restrict__ betta, int width, int depth, int n)\n" +
                    "{\n" +
                    "    int x = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    int y = blockDim.y * blockIdx.y + threadIdx.y;\n" +
                    "    if (x < n && y < width) {\n" +
                    "        TYPE* input = P[0][x];\n" +
                    "        TYPE mean = P[1][x][y];\n" +
                    "        TYPE dep = __int2bfloat16_rn(depth);\n" +
                    //       საშუალო მნიშვნელობის გამოთვლა
                    //       find mean
                    ////////////////////////////////////////////////////////////
                    "        int index = y * depth;\n" +
                    "        for (int k = 0; k < depth; k++, index++) {\n" +
                    "           mean += input[index];\n" +
                    "        }\n" +
                    "        mean = mean / dep;\n" +
                    "        P[1][x][y] = mean;\n" +
                    ////////////////////////////////////////////////////////////
                    //       find variance
                    ////////////////////////////////////////////////////////////
                    "        TYPE var = P[2][x][y];\n" +
                    "        TYPE sub;\n" +
                    "        index = y * depth;\n" +
                    "        mean = P[1][x][y];\n" +
                    "        for (int k = 0; k < depth; k++, index++) {\n" +
                    "            sub = input[index] - mean;\n" +
                    "            var = var + sub * sub;\n" +
                    "        }\n" +
                    "        var = var / dep;\n" +
                    "        P[2][x][y] = var;\n" +
                    ////////////////////////////////////////////////////////////
                    //       Normalization
                    ////////////////////////////////////////////////////////////
                    "        TYPE varSqrt = hsqrt(var + sh[14]);\n" +
                    "        TYPE* output = P[3][x];\n" +
                    "        index = y * depth;\n" +
                    "        for (int k = 0; k < depth; k++, index++) {\n" +
                    "             output[index] = ((input[index] - mean) / varSqrt) * gamma[k] + betta[k];\n" +
                    "        }\n" +
                    ////////////////////////////////////////////////////////////
                    "    }\n" +
                    "}\n" +*/

                    "extern \"C\"\n" +
                    "__global__ void NormalizationLayerBackward2D(float*** P, const float* __restrict__ gamma, const float* __restrict__ betta, float* derGamma, float* derBetta, int outWidth, int outDepth, int width, int depth, int n)\n" +
                    "{\n" +
                    "    int x = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    int y = blockDim.y * blockIdx.y + threadIdx.y;\n" +
                    "    if (x < n && y < width) {\n" +
                    "        float* errorNL = P[0][x];\n" +
                    "        float var = P[1][x][y];\n" +
                    "        float* input = P[2][x];\n" +
                    "        float mean = P[3][x][y];\n" +
                    "        float* error = P[4][x];\n" +
                    "        float* output = P[5][x];\n" +
                    /////////////////////////////////////////////////////////////////
                    //       Der var
                    /////////////////////////////////////////////////////////////////
                    "        float dVar_m = -0.5f * powf(var + 0.001f, -1.5f);\n" +
                    "        int index = y * depth;\n" +
                    "        float derVariance = 0.0f;\n" +
                    "        for (int k = 0; k < depth; k++, index++) {\n" +
                    "            derVariance += errorNL[index] * gamma[k] * (input[index] - mean);\n" +
                    "        }\n" +
                    "        derVariance *= dVar_m;\n" +
                    /////////////////////////////////////////////////////////////////
                    //       Der mean
                    /////////////////////////////////////////////////////////////////
                    "        dVar_m = 0.0f;\n" +
                    "        float derMean = 0.0f;\n" +
                    "        float dMean = -1.0f / sqrtf(var + 0.001f);\n" +

                    "        index = y * depth;\n" +
                    "        for (int k = 0; k < depth; k++, index++) {\n" +
                    "            derMean += errorNL[index] * gamma[k];\n" +
                    "            dVar_m += input[index] - mean;\n" +
                    "        }\n" +

                    "        derMean *= dMean;\n" +
                    "        derMean += (-2.0f * derVariance * dVar_m) / depth;\n" +
                    //////////////////////////////////////////////////////////////////
                    //       Der norm
                    //////////////////////////////////////////////////////////////////
                    "        derMean /= depth;\n" +
                    "        derVariance *= 2.0f / (depth);\n" +

                    "        float _dVar = 1.0f / sqrtf(var + 0.001f);\n" +
                    "        index = y * depth;\n" +
                    "        for (int k = 0; k < depth; k++, index++) {\n" +
                    "            error[index] = errorNL[index] * gamma[k] * _dVar + derVariance * (input[index] - mean) + derMean;\n" +
                    "        }\n" +
                    //////////////////////////////////////////////////////////////////
                    //       Derivative weight
                    //////////////////////////////////////////////////////////////////
                    "        index = y * depth;\n" +
                    "        for (int k = 0; k < depth; k++, index++) {\n" +
                    "            atomicAdd(&derBetta[k], errorNL[index]);\n" +
                    "            atomicAdd(&derGamma[k], errorNL[index] * ((output[index] - betta[k]) / gamma[k]));\n" +
                    "        }\n" +
                    "    }\n" +
                    "}\n" +

                    /*"extern \"C\"\n" +
                    "__global__ void NormalizationLayerBackward_TYPE_2D(TYPE*** P, const TYPE* __restrict__ gamma, const TYPE* __restrict__ betta, TYPE* derGamma, TYPE* derBetta, int outWidth, int outDepth, int width, int depth, int n)\n" +
                    "{\n" +
                    "    int x = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    int y = blockDim.y * blockIdx.y + threadIdx.y;\n" +
                    "    if (x < n && y < width) {\n" +
                    "        TYPE sh5 = sh[14];\n" +
                    "        TYPE sh0 = sh[0];\n" +
                    "        TYPE dep = __int2TYPE_rn(depth);\n" +
                    "        TYPE* errorNL = P[0][x];\n" +
                    "        TYPE var = P[1][x][y];\n" +
                    "        TYPE* input = P[2][x];\n" +
                    "        TYPE mean = P[3][x][y];\n" +
                    "        TYPE* error = P[4][x];\n" +
                    "        TYPE* output = P[5][x];\n" +
                    /////////////////////////////////////////////////////////////////
                    //       Der var
                    /////////////////////////////////////////////////////////////////
                    //"        TYPE dVar_m = sh[6] * pow(var + sh5, sh[7]);\n" +
                    "        TYPE dVar_m = sh[6] * hexp2(sh[7] * hlog2(var + sh5));\n" +

                    "        int index = y * depth;\n" +
                    "        TYPE derVariance = sh0;\n" +
                    "        for (int k = 0; k < depth; k++, index++) {\n" +
                    "            derVariance += errorNL[index] * gamma[k] * (input[index] - mean);\n" +
                    "        }\n" +
                    "        derVariance *= dVar_m;\n" +
                    /////////////////////////////////////////////////////////////////
                    //       Der mean
                    /////////////////////////////////////////////////////////////////
                    "        dVar_m = sh0;\n" +
                    "        TYPE derMean = sh0;\n" +
                    "        TYPE dMean = sh[8] / hsqrt(var + sh5);\n" +

                    "        index = y * depth;\n" +
                    "        for (int k = 0; k < depth; k++, index++) {\n" +
                    "            derMean += errorNL[index] * gamma[k];\n" +
                    "            dVar_m += input[index] - mean;\n" +
                    "        }\n" +

                    "        derMean *= dMean;\n" +
                    "        derMean += (sh[9] * derVariance * dVar_m) / dep;\n" +
                    //////////////////////////////////////////////////////////////////
                    //       Der norm
                    //////////////////////////////////////////////////////////////////
                    "        derMean /= dep;\n" +
                    "        derVariance *= (-sh[9]) / dep;\n" +

                    "        TYPE _dVar = sh[4] / hsqrt(var + sh5);\n" +
                    "        index = y * depth;\n" +
                    "        for (int k = 0; k < depth; k++, index++) {\n" +
                    "            error[index] = errorNL[index] * gamma[k] * _dVar + derVariance * (input[index] - mean) + derMean;\n" +
                    "        }\n" +
                    //////////////////////////////////////////////////////////////////
                    //       Derivative weight
                    //////////////////////////////////////////////////////////////////
                    "        index = y * depth;\n" +
                    "        for (int k = 0; k < depth; k++, index++) {\n" +
                    "            atomicAdd(&derBetta[k], errorNL[index]);\n" +
                    "            atomicAdd(&derGamma[k], errorNL[index] * ((output[index] - betta[k]) / gamma[k]));\n" +
                    "        }\n" +
                    "    }\n" +
                    "}\n" +*/

                    "extern \"C\"\n" +
                    "__global__ void dropout(TYPE* A, TYPE* random, TYPE chanceDrop, int numElements)\n" +
                    "{\n" +
                    "    const int idx = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (idx < numElements) {\n" +
                    "       TYPE drop = sh[4] / (sh[4] - chanceDrop);\n" +
                    "       if (random[idx] > chanceDrop)\n" +
                    "       {\n" +
                    "           A[idx] = A[idx] * drop;\n" +
                    "       }\n" +
                    "    }\n" +
                    "}\n" +

                    /*"extern \"C\"\n" +
                    "__global__ void sub_gpu_TYPE2(const TYPE2* __restrict__ first, const TYPE2* __restrict__ second, TYPE2* result, int numElements)\n" +
                    "{\n" +
                    "    const int idx = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (idx < numElements) {\n" +
                    "       result[idx] = __hsub2(first[idx], second[idx]);\n" +
                    "    }\n" +
                    "}\n" +*/

                    "extern \"C\"\n" +
                    "__global__ void sub_gpu(const float* __restrict__ first, const float* __restrict__ second, float* result, int numElements)\n" +
                    "{\n" +
                    "    const int idx = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (idx < numElements) {\n" +
                    "       result[idx] = first[idx] - second[idx];\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void sub_gpu_TYPE(const TYPE* __restrict__ first, const TYPE* __restrict__ second, TYPE* result, int numElements)\n" +
                    "{\n" +
                    "    const int idx = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (idx < numElements) {\n" +
                    "       result[idx] = first[idx] - second[idx];\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void sub_bfloatAndFloat(const TYPE* __restrict__ first, const float* __restrict__ second, float* result, int numElements)\n" +
                    "{\n" +
                    "    const int idx = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (idx < numElements) {\n" +
                    "       result[idx] = __bfloat162float(first[idx]) - second[idx];\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void sub_floatAndbFloat(const float* __restrict__ first, const TYPE* __restrict__ second, float* result, int numElements)\n" +
                    "{\n" +
                    "    const int idx = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (idx < numElements) {\n" +
                    "       result[idx] = first[idx] - __bfloat162float(second[idx]);\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void mul(TYPE* result, TYPE val, int numElements)\n" +
                    "{\n" +
                    "    int idx = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (idx < numElements) {\n" +
                    "       result[idx] = result[idx] * val;\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void mul_float(float* result, float val, int numElements)\n" +
                    "{\n" +
                    "    int idx = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (idx < numElements) {\n" +
                    "       result[idx] *= val;\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void clip(float* data, float min, float max, int numElements)\n" +
                    "{\n" +
                    "    int idx = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (idx < numElements) {\n" +
                    "        float a = data[idx];\n" +
                    "        if (a > max) {\n" +
                    "            data[idx] = max;\n" +
                    "        } else if (a < min) {\n" +
                    "            data[idx] = min;\n" +
                    "        }\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void clip_TYPE(TYPE* data, TYPE min, TYPE max, int numElements)\n" +
                    "{\n" +
                    "    int idx = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (idx < numElements) {\n" +
                    "        TYPE a = data[idx];\n" +
                    "        if (a > max) {\n" +
                    "            data[idx] = max;\n" +
                    "        } else if (a < min) {\n" +
                    "            data[idx] = min;\n" +
                    "        }\n" +
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
                    "__global__ void pow2_TYPE(TYPE* data, int numElements)\n" +
                    "{\n" +
                    "    int idx = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (idx < numElements) {\n" +
                    "       data[idx] = data[idx] * data[idx];\n" +
                    "    }\n" +
                    "}\n" +

                    /*"extern \"C\"\n" +
                    "__global__ void subAbs_TYPE(TYPE* first, TYPE* second, TYPE* result, int numElements)\n" +
                    "{\n" +
                    "    int i = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (i < numElements) {\n" +
                    "       result[i] = __habs(first[i] - second[i]);\n" +
                    "    }\n" +
                    "}\n" +*/

                    "extern \"C\"\n" +
                    "__global__ void subAbs(float* first, float* second, float* result, int numElements)\n" +
                    "{\n" +
                    "    int i = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (i < numElements) {\n" +
                    "       result[i] = fabsf(first[i] - second[i]);\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void sum(float* data, float* result, int numElements)\n" +
                    "{\n" +
                    "    int i = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (i < numElements) {\n" +
                    "       atomicAdd(&result[0], data[i]);\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void derAbs(float* first, float* second, float* result, int numElements)\n" +
                    "{\n" +
                    "    int i = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (i < numElements) {\n" +
                    "       float diff = first[i] - second[i];\n" +
                    "       result[i] = diff / fabsf(diff + 0.00000001f);\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void fisnan(const TYPE* __restrict__ data, int* result, int numElements)\n" +
                    "{\n" +
                    "    int idx = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (idx < numElements) {\n" +
                    "       if (result[0] == 0) {\n" +
                    "           if (isnan(__bfloat162float(data[idx]))) {\n" +
                    "               result[0] = idx;\n" +
                    "           }\n" +
                    "       }\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void fisnan_float(const float* __restrict__ data, int* result, int numElements)\n" +
                    "{\n" +
                    "    int idx = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (idx < numElements) {\n" +
                    "       if (result[0] == 0) {\n" +
                    "           if (__isnan(data[idx])) {\n" +
                    "               result[0] = idx;\n" +
                    "           }\n" +
                    "       }\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void hisinf(const TYPE* __restrict__ data, int* result, int numElements)\n" +
                    "{\n" +
                    "    int idx = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (idx < numElements) {\n" +
                    "       if (result[0] == 0) {\n" +
                    "           if (isinf(__bfloat162float(data[idx]))) {\n" +
                    "               result[0] = idx;\n" +
                    "           }\n" +
                    "           if (isinf(__bfloat162float(data[idx])) == -1) {\n" +
                    "               result[0] = idx;\n" +
                    "           }\n" +
                    "       }\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void hisinf_float(const float* __restrict__ data, int* result, int numElements)\n" +
                    "{\n" +
                    "    int idx = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (idx < numElements) {\n" +
                    "       if (result[0] == 0) {\n" +
                    "           if (__isinf(data[idx])) {\n" +
                    "               result[0] = idx;\n" +
                    "           }\n" +
                    "           if (__isinf(data[idx]) == -1) {\n" +
                    "               result[0] = idx;\n" +
                    "           }\n" +
                    "       }\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void momentum(TYPE* data, const TYPE* __restrict__ array, TYPE decay, TYPE rt, int numElements)\n" +
                    "{\n" +
                    "    int idx = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (idx < numElements) {\n" +
                    "       data[idx] = decay * data[idx] + array[idx] * rt;\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void momentum_float(float* data, const float* __restrict__ array, float decay, float rt, int numElements)\n" +
                    "{\n" +
                    "    int idx = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (idx < numElements) {\n" +
                    "       data[idx] = decay * data[idx] + array[idx] * rt;\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void momentumPow2(TYPE* data, const TYPE* __restrict__ vector, TYPE decay, TYPE dr, int numElements)\n" +
                    "{\n" +
                    "    int idx = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (idx < numElements) {\n" +
                    "       data[idx] = decay * data[idx] + dr * vector[idx] * vector[idx];\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void momentumPow2_float(float* data, const float* __restrict__ vector, float decay, float dr, int numElements)\n" +
                    "{\n" +
                    "    int idx = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (idx < numElements) {\n" +
                    "       data[idx] = decay * data[idx] + dr * vector[idx] * vector[idx];\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void subDivSqrtNorm_TYPE(const TYPE* __restrict__ nominator, const TYPE* __restrict__ denominator, TYPE lr, TYPE normN, TYPE normD, TYPE* data, int numElements)\n" +
                    "{\n" +
                    "    int idx = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (idx < numElements) {\n" +
                    "       TYPE sh5 = sh[5];\n" +
                    "       TYPE cur_lr = lr / (normN +  sh5);\n" +
                    "       data[idx] = data[idx] - (cur_lr * (nominator[idx]) / (__float2bfloat16(sqrtf(__bfloat162float(denominator[idx] / normD))) + sh5));\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void subDivSqrtNorm(const float* __restrict__ nominator, const float* __restrict__ denominator, float lr, float normN, float normD, float* data, int numElements)\n" +
                    "{\n" +
                    "    int idx = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (idx < numElements) {\n" +
                    "       float cur_lr = lr / (normN + 0.0000001f);\n" +
                    "       data[idx] -= cur_lr * (nominator[idx]) / (sqrtf(denominator[idx] / normD) + 0.0000001f);\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void addBackCopy(const float* __restrict__ matrix, int m_column, int row, int column, int start, float* data)\n" +
                    "{\n" +
                    "    const int x = blockIdx.x * blockDim.x + threadIdx.x;\n" +
                    "    const int y = blockIdx.y * blockDim.y + threadIdx.y;\n" +
                    "    if (x < row && y < column) {\n" +
                    "       const int indexOut = x * m_column + start * column + y;\n" +
                    "       const int indexIn = x * column + y;\n" +
                    "       data[indexIn] = matrix[indexOut];\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void addBackCopy_TYPE(const TYPE* __restrict__ matrix, int m_column, int row, int column, int start, TYPE* data)\n" +
                    "{\n" +
                    "    const int x = blockIdx.x * blockDim.x + threadIdx.x;\n" +
                    "    const int y = blockIdx.y * blockDim.y + threadIdx.y;\n" +
                    "    if (x < row && y < column) {\n" +
                    "       const int indexOut = x * m_column + start * column + y;\n" +
                    "       const int indexIn = x * column + y;\n" +
                    "       data[indexIn] = matrix[indexOut];\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void dropoutBack(const TYPE* __restrict__ output, const TYPE* __restrict__ error, TYPE drop, TYPE* data, int numElements)\n" +
                    "{\n" +
                    "    const int idx = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (idx < numElements) {\n" +
                    "       if (output[idx] != sh[0]) {\n" +
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
                    "__global__ void mask_TYPE(const TYPE* __restrict__ A, TYPE val, TYPE newVal, TYPE* C, int numElements)\n" +
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
                    "__global__ void fillUnderDiagonal_TYPE(int column, TYPE val, TYPE* data, int numElements)\n" +
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
                    "        float val = tanhf(0.7978846f * x + 0.0356774f * x * x * x);\n" +
                    "        data[idx] = error[idx] * 0.5f * (1.0f + val + x * (1.0f - val * val) * (0.79788846f + 0.1070322f * x * x));\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void derGelu_TYPE(const TYPE* __restrict__ input, const TYPE* __restrict__ error, TYPE* data, int numElements)\n" +
                    "{\n" +
                    "    int idx = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (idx < numElements) {\n" +
                    "        float x = __bfloat162float(input[idx]);\n" +
                    "        float val = tanhf(0.7978846f * x + 0.0356774f * x * x * x);\n" +
                    "        data[idx] = __float2bfloat16(__bfloat162float(error[idx]) * 0.5f * (1.0f + val + x * (1.0f - val * val) * (0.79788846f + 0.1070322f* x * x)));\n" +
                    "    }\n" +
                    "}\n" +

                    "__device__ const int BLOCK_SIZE = 32;\n" +

                    "extern \"C\"\n" +
                    "__global__ void matvec_kernel(const float * __restrict__ dA, const float * __restrict__ dx, float * __restrict__ dy, const unsigned int nRows, const unsigned int nCols)\n" +
                    "{\n" +
                    "    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;\n" +
                    "    __shared__ float x_shared[BLOCK_SIZE];\n" +
                    "    float y_val = 0.0;\n" +
                    "    #pragma unroll\n" +
                    "    for (unsigned int m = 0; m < ((nCols + BLOCK_SIZE - 1)/ BLOCK_SIZE); ++m)\n" +
                    "    {\n" +
                    "        if ((m * BLOCK_SIZE + threadIdx.x) <  nCols) \n" +
                    "           x_shared[threadIdx.x] = dx[threadIdx.x + m * BLOCK_SIZE];\n" +
                    "        else\n" +
                    "            x_shared[threadIdx.x] = 0.f;\n" +
                    "        __syncthreads();\n" +
                    "    #pragma unroll\n" +
                    "        for (unsigned int e = 0; e < BLOCK_SIZE; ++e) {\n" +
                            // --- Column-major ordering - faster
                    //"        y_val += dA[tid + (e + BLOCK_SIZE * m) * nRows] * x_shared[e];\n" +
                            // --- Row-major ordering - slower
                    "        y_val = y_val + dA[tid * nCols + (e + BLOCK_SIZE * m)] * x_shared[e];\n" +
                    "    }\n" +
                    "        __syncthreads();\n" +
                    "    }\n" +
                    "    if (tid < nRows) dy[tid] = y_val;\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void matvec_kernel_TYPE(const TYPE* __restrict__ dA, const TYPE* __restrict__ dx, TYPE* __restrict__ dy, const unsigned int nRows, const unsigned int nCols)\n" +
                    "{\n" +
                    "    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;\n" +
                    "    __shared__ TYPE x_shared[BLOCK_SIZE];\n" +
                    "    TYPE y_val = 0.0;\n" +
                    "    #pragma unroll\n" +
                    "    for (unsigned int m = 0; m < ((nCols + BLOCK_SIZE - 1)/ BLOCK_SIZE); ++m)\n" +
                    "    {\n" +
                    "        if ((m * BLOCK_SIZE + threadIdx.x) <  nCols) \n" +
                    "           x_shared[threadIdx.x] = dx[threadIdx.x + m * BLOCK_SIZE];\n" +
                    "        else\n" +
                    "            x_shared[threadIdx.x] = 0.f;\n" +
                    "        __syncthreads();\n" +
                    "    #pragma unroll\n" +
                    "        for (unsigned int e = 0; e < BLOCK_SIZE; ++e) {\n" +
                    // --- Column-major ordering - faster
                    //"        y_val += dA[tid + (e + BLOCK_SIZE * m) * nRows] * x_shared[e];\n" +
                    // --- Row-major ordering - slower
                    "        y_val = y_val + dA[tid * nCols + (e + BLOCK_SIZE * m)] * x_shared[e];\n" +
                    "    }\n" +
                    "        __syncthreads();\n" +
                    "    }\n" +
                    "    if (tid < nRows) dy[tid] = y_val;\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void Softmax(const float* __restrict__ input, float* data, int column, int numElements)\n" +
                    "{\n" +
                    "    int k = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (k < numElements)\n" +
                    "    {\n" +
                    "       float sum = 0.0f;\n" +
                    "       int index = k * column;\n" +
                    "       float max = input[index];\n" +
                    "       for (int i = 1; i < column; i++, index++) {\n" +
                    "           float in = input[index];\n" +
                    "           if (max < in)\n" +
                    "               max = in;\n" +
                    "       }\n" +
                    "       index = k * column;\n" +
                    "       for (int i = 0; i < column; i++, index++) {\n" +
                    "           float d = expf(input[index] - max);\n" +
                    "           d = InfinityCheck(d);\n" +
                    "           data[index] = d;\n" +
                    "           sum = sum + d;\n" +
                    "       }\n" +
                    "       if (sum == 0.0f) {\n" +
                    "           sum = sum + sh[5];\n" +
                    "       }\n" +
                    "       sum = InfinityCheck(sum);\n" +
                    "       index = k * column;\n" +
                    "       for (int i = 0; i < column; i++, index++) {\n" +
                    "           data[index] /= sum;\n" +
                    "       }\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void Softmax_TYPE(const TYPE* __restrict__ input, TYPE* data, int column, int numElements)\n" +
                    "{\n" +
                    "    int k = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    if (k < numElements)\n" +
                    "    {\n" +
                    "       TYPE sum = sh[0];\n" +
                    "       int index = k * column;\n" +
                    "       TYPE max = input[index];\n" +
                    "       for (int i = 1; i < column; i++, index++) {\n" +
                    "           TYPE in = input[index];\n" +
                    "           if (max < in)\n" +
                    "               max = in;\n" +
                    "       }\n" +
                    "       index = k * column;\n" +
                    "       for (int i = 0; i < column; i++, index++) {\n" +
                    "           TYPE d = exp(input[index] - max);\n" +
                    "           d = InfinityCheck_TYPE(d);\n" +
                    "           data[index] = d;\n" +
                    "           sum = sum + d;\n" +
                    "       }\n" +
                    "       if (sum == sh[0]) {\n" +
                    "           sum = sum + sh[5];\n" +
                    "       }\n" +
                    "       sum = InfinityCheck_TYPE(sum);\n" +
                    "       index = k * column;\n" +
                    "       for (int i = 0; i < column; i++, index++) {\n" +
                    "           data[index] = data[index] / sum;\n" +
                    "       }\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void matrixMulti(float* A_d, float* B_d, float* C_d, int m, int k, int n)\n" +
                    "{\n" +
                    "    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];\n" +
                    "    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];\n" +
                    "    int col = blockIdx.x*blockDim.x + threadIdx.x;\n" +
                    "    int row = blockIdx.y*blockDim.y + threadIdx.y;\n" +
                    "    int tx = threadIdx.x;\n" +
                    "    int ty = threadIdx.y;\n" +
                    "    float sum = 0;\n" +

                    "    for(int t=0; t<(n-1)/TILE_WIDTH+1; t++)\n" +
                    "    {\n" +
                    "        if(row<m && t*TILE_WIDTH+tx<n)\n" +
                    "            ds_A[ty][tx] = A_d[row*n + t*TILE_WIDTH+tx];\n" +
                    "        else\n" +
                    "            ds_A[ty][tx] = 0.0;\n" +
                    "        if(t*TILE_WIDTH+ty<n && col<k)\n" +
                    "            ds_B[ty][tx] = B_d[(t*TILE_WIDTH+ty)*k + col];\n" +
                    "        else\n" +
                    "            ds_B[ty][tx] = 0.0;\n" +
                    "        __syncthreads();\n" +
                    "        for(int i=0; i<TILE_WIDTH; i++)\n" +
                    "            sum += ds_A[ty][i] * ds_B[i][tx];\n" +
                    "        __syncthreads();\n" +
                    "    }\n" +
                    "    if(row<m && col<k)\n" +
                    "        C_d[col+row*k] = sum;\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void matrixMulti_TYPE(TYPE* A_d, TYPE* B_d, TYPE* C_d, int m, int k, int n)\n" +
                    "{\n" +
                    "    __shared__ TYPE ds_A[TILE_WIDTH][TILE_WIDTH];\n" +
                    "    __shared__ TYPE ds_B[TILE_WIDTH][TILE_WIDTH];\n" +
                    "    int col = blockIdx.x*blockDim.x + threadIdx.x;\n" +
                    "    int row = blockIdx.y*blockDim.y + threadIdx.y;\n" +
                    "    int tx = threadIdx.x;\n" +
                    "    int ty = threadIdx.y;\n" +
                    "    TYPE sum = sh[0];\n" +

                    "    for(int t=0; t<(n-1)/TILE_WIDTH+1; t++)\n" +
                    "    {\n" +
                    "        if(row<m && t*TILE_WIDTH+tx<n)\n" +
                    "            ds_A[ty][tx] = A_d[row*n + t*TILE_WIDTH+tx];\n" +
                    "        else\n" +
                    "            ds_A[ty][tx] = 0.0;\n" +
                    "        if(t*TILE_WIDTH+ty<n && col<k)\n" +
                    "            ds_B[ty][tx] = B_d[(t*TILE_WIDTH+ty)*k + col];\n" +
                    "        else\n" +
                    "            ds_B[ty][tx] = 0.0;\n" +
                    "        __syncthreads();\n" +
                    "        for(int i=0; i<TILE_WIDTH; i++)\n" +
                    "            sum = sum + ds_A[ty][i] * ds_B[i][tx];\n" +
                    "        __syncthreads();\n" +
                    "    }\n" +
                    "    if(row<m && col<k)\n" +
                    "        C_d[col+row*k] = sum;\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void derSoftmax(const float* __restrict__ output, const float* __restrict__ error, float* data, int row, int column)\n" +
                    "{\n" +
                    "    int k = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    int i = blockDim.y * blockIdx.y + threadIdx.y;\n" +
                    "    if (k < row && i < column)\n" +
                    "    {\n" +
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
                    "           sum += error[indexJ] * value;\n" +
                    "        }\n" +
                    "        data[indexI] = sum;\n" +
                    "    }\n" +
                    "}\n" +

                    "extern \"C\"\n" +
                    "__global__ void derSoftmax_TYPE(const TYPE* __restrict__ output, const TYPE* __restrict__ error, TYPE* data, int row, int column)\n" +
                    "{\n" +
                    "    int k = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "    int i = blockDim.y * blockIdx.y + threadIdx.y;\n" +
                    "    if (k < row && i < column)\n" +
                    "    {\n" +
                    "       k = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                    "       i = blockDim.y * blockIdx.y + threadIdx.y;\n" +
                    "       TYPE value = sh[0];\n" +
                    "       int index = k * column;\n" +
                    "       int indexI = index + i;\n" +
                    "       data[indexI] = sh[0];\n" +
                    "       TYPE o = output[indexI];\n" +
                    "       TYPE sum = sh[0];\n" +
                    "       TYPE sh4 = sh[4];\n" +
                    "       for (int j = 0; j < column; j++) {\n" +
                    "           int indexJ = index + j;\n" +
                    "           if (i != j) {\n" +
                    "               value = o * -output[indexJ];\n" +
                    "           } else {\n" +
                    "               value = o * (sh4 - o);\n" +
                    "           }\n" +
                    "           sum = sum + error[indexJ] * value;\n" +
                    "        }\n" +
                    "        data[indexI] = sum;\n" +
                    "    }\n" +
                    "}\n";


}
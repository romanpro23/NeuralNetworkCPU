package utilities;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasAtomicsMode;
import jcuda.jcublas.cublasHandle;
import jcuda.jcublas.cublasPointerMode;
import nnarrays.NNArray;

import java.lang.ref.WeakReference;
import java.util.Collection;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedList;

import static jcuda.driver.JCudaDriver.*;
import static jcuda.jcublas.cublasMath.CUBLAS_TENSOR_OP_MATH;
import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;
import static nnarrays.NNArray.floatToBFloat16;

public class GPUInit {

    public static LinkedHashMap<String, WeakReference<Object>> allocated;
    public static LinkedHashMap<String, Use> allocatedUse;
    public static jcuda.jcublas.cublasHandle cublasHandle;
    public static CUmodule helperModule;

    public static void startup() {
        JCudaHelper.init();
        cublasHandle = new cublasHandle();
        JCublas2.cublasCreate(cublasHandle);
        //JCublas2.cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH);
        helperModule = JCudaHelper.compile("la_helper_funs", NNArray.kernels);
        allocated = new LinkedHashMap<String, WeakReference<Object>>();
        allocatedUse = new LinkedHashMap<String, Use>();
        //JCublas2.cublasSetAtomicsMode(cublasHandle, cublasAtomicsMode.CUBLAS_ATOMICS_ALLOWED);
        //JCublas2.cublasSetPointerMode(cublasHandle, cublasPointerMode.CUBLAS_POINTER_MODE_HOST);
        Use.GPU = true;

        short[] sh = new short[]{
                floatToBFloat16(0.0f),//0
                floatToBFloat16(0.7978846f),//1
                floatToBFloat16(0.0356774f),//2
                floatToBFloat16(0.5f),//3
                floatToBFloat16(1.0f),//4
                floatToBFloat16(0.00000001f),//5
                floatToBFloat16(-0.5f),//6
                floatToBFloat16(-1.5f),//7
                floatToBFloat16(-1.0f),//8
                floatToBFloat16(-2.0f),//9
                floatToBFloat16(0.79788846f),//10
                floatToBFloat16(0.1070322f), //11
                floatToBFloat16(65504.0f), //12
                floatToBFloat16(65504.0f), //13
                floatToBFloat16(100.0f), //14
                floatToBFloat16(0.005f), //15
                floatToBFloat16(0.5f), //16
        };

        CUdeviceptr devPtr = new CUdeviceptr();
        int result = JCudaDriver.cuModuleGetGlobal(devPtr, new long[1], helperModule, "sh");
        int error = cuMemcpyHtoD(devPtr, Pointer.to(sh), (long) sh.length * Sizeof.SHORT);
    }

    public static void shutdown() {
        freeAll();
        JCublas2.cublasDestroy(cublasHandle);
    }

    public static void freeAll() {
        /*while (!allocated.isEmpty()) {
            NNArray mat = allocated.poll();
            mat.free();
        }*/
    }

    public static void freeAllBut(NNArray... args) {
        Collection<NNArray> keep = new HashSet<NNArray>();
        for (NNArray mat : args) keep.add(mat);
        freeAllBut(keep);
    }

    public static void freeAllBut(Collection<NNArray> keep) {
        /*while (!allocated.isEmpty()) {
            NNArray mat = allocated.poll();
            if (!keep.contains(mat)) {
                mat.free();
            }
        }*/
    }
}
package utilities;

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

public class GPUInit {

    public static LinkedHashMap<String, WeakReference<Object>> allocated;
    public static LinkedHashMap<String, Use> allocatedUse;
    public static jcuda.jcublas.cublasHandle cublasHandle;
    public static CUmodule helperModule;

    public static void startup() {
        JCudaHelper.init();
        cublasHandle = new cublasHandle();
        JCublas2.cublasCreate(cublasHandle);
        helperModule = JCudaHelper.compile("la_helper_funs", NNArray.kernels);
        allocated = new LinkedHashMap<String, WeakReference<Object>>();
        allocatedUse = new LinkedHashMap<String, Use>();
        //JCublas2.cublasSetAtomicsMode(cublasHandle, cublasAtomicsMode.CUBLAS_ATOMICS_ALLOWED);
        //JCublas2.cublasSetPointerMode(cublasHandle, cublasPointerMode.CUBLAS_POINTER_MODE_HOST);
        Use.GPU = true;
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
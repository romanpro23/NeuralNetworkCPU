package utilities;


import jcuda.Pointer;

public class Use {
    public static boolean GPU = true;
    public static boolean CPU = true;
    public int HashCode;
    public Pointer data_gpu;
    public static boolean DEBUG_SYNC = true;

    private static boolean mGPU = false;

    public static void GPU_Sleep() {
        if (Use.GPU) {
            Use.GPU = false;
            mGPU = true;

            //Use.CPU = false;//!!
        }
    }

    public static void GPU_WakeUp() {
        if (mGPU) {
            Use.GPU = true;
            mGPU = false;

            //Use.CPU = true;//!!
        }
    }
}
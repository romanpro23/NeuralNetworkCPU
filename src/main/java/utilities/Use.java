package utilities;


import jcuda.Pointer;

public class Use {
    public static boolean GPU = true;
    public static boolean CPU = false;
    public int HashCode;
    public Pointer data_gpu;
    public static boolean DEBUG_SYNC = false;

    private static boolean mGPU = false;

    public static final int WordCount = 180;//80;//34;//180;
    public static final int sizeVocabulary = 50;//71466 + 8;//50

    public static final boolean WorkingWithCharacter = true;

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
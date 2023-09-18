package utilities;


import jcuda.Pointer;

public class Use {
    public static boolean GPU = true;
    public static boolean CPU = false;
    public int HashCode;
    public Pointer data_gpu;
    public static boolean DEBUG_SYNC = false;
}
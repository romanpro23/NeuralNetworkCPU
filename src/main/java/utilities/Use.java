package utilities;


import jcuda.Pointer;

public class Use {
    public static boolean GPU = false;
    public int HashCode;
    public Pointer data_gpu;
    public Pointer rowsIndex_gpu;
    public Pointer columnsIndex_gpu;
}
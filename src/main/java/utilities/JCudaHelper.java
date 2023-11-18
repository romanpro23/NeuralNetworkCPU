package utilities;

import jcuda.Pointer;
import jcuda.driver.*;
import java.io.*;

import static jcuda.driver.JCudaDriver.*;
import static jcuda.runtime.JCuda.cudaDeviceSetCacheConfig;
import static jcuda.runtime.JCuda.cudaFuncGetAttributes;
import static jcuda.runtime.cudaFuncCache.cudaFuncCachePreferL1;
import static jcuda.runtime.cudaFuncCache.cudaFuncCachePreferShared;
import static utilities.GPUInit.helperModule;

public class JCudaHelper {
    public static CUcontext CONTEXT;
    private static CUdevice device;

    public static void init() {
        JCudaDriver.setExceptionsEnabled(true);

        JCudaDriver.cuInit(0);
        device = new CUdevice();
        JCudaDriver.cuDeviceGet(device, 0);

        CONTEXT = new CUcontext();
        JCudaDriver.cuCtxCreate(CONTEXT, 0, device);

        cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
    }

    public static void destroy() {
        JCudaDriver.cuCtxDestroy(CONTEXT);
    }

    private static String cuNameToPtx(final String cuFileName) {
        final int extensionPoint = cuFileName.lastIndexOf('.');
        if (extensionPoint == -1) {
            throw new RuntimeException("Wrong extension " + cuFileName);
        }
        return cuFileName.substring(0, extensionPoint + 1) + "ptx";
    }

    //@SuppressWarnings("StringBufferReplaceableByString")
    public static CUmodule compile(String kernelName, String kernelSrc) {
        String ptxFileName = kernelName+".ptx";
        File ptxFile = new File(ptxFileName);

        //long start = System.nanoTime();
        File cuFile = new File(kernelName+".cu");
        BufferedWriter out = null;
        try {
            out = new BufferedWriter(new FileWriter(cuFile));
            out.append(kernelSrc);
            out.flush();
            out.close();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        String modelString = System.getProperty("sun.arch.data.model");
        int[] major = new int[1];
        int[] minor = new int[1];
        JCudaDriver.cuDeviceComputeCapability(major, minor, device);

        final String command = new StringBuilder()
                .append("nvcc ")
                .append("-use_fast_math ").append("-arch=sm_"+major[0]+""+minor[0]+" ").append(' ')
                .append("-m ").append(modelString).append(' ')
                .append("-ptx ").append(cuFile.getAbsolutePath()).append(' ')
                .append("-o ").append(ptxFile.getAbsolutePath())
                .toString()
                ;
        execNvcc(command);

        CUmodule module = new CUmodule();
        cuModuleLoad(module, ptxFileName);

        return module;
    }

    public static void execNvcc(final String command) {
        int exitCode;
        String stdErr = "";
        String stdOut = "";

        try {
            final Process process = Runtime.getRuntime().exec(command);

            stdErr = streamToString(process.getErrorStream());
            stdOut = streamToString(process.getInputStream());

            exitCode = process.waitFor();
        }
        catch (IOException | InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException(
                    "Interrupted while waiting for nvcc output.\nSTDOUT:\n" + stdOut + "\nSTDERR\n" + stdErr, e
            );
        }

        if (exitCode != 0) {
            throw new RuntimeException("Invocation '" + command + "' failed.\n" + stdOut + "\n" + stdErr);
        }
    }

    private static String streamToString(final InputStream inputStream) throws IOException {
        final StringBuilder builder = new StringBuilder();
        try (final LineNumberReader reader = new LineNumberReader(new InputStreamReader(inputStream))) {
            final char[] buffer = new char[8192];

            int read;
            while ((read = reader.read(buffer)) != -1) {
                builder.append(buffer, 0, read);
            }
        }
        return builder.toString();
    }
}
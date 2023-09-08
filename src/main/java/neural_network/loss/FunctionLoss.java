package neural_network.loss;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUfunction;
import jcuda.driver.JCudaDriver;
import jcuda.jcublas.JCublas2;
import jcuda.runtime.JCuda;
import lombok.NoArgsConstructor;
import nnarrays.NNArray;
import nnarrays.NNArrays;
import utilities.Use;

import static java.lang.Math.log;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;
import static nnarrays.NNArray.BLOCK_SIZE;
import static utilities.GPUInit.helperModule;

public interface FunctionLoss {
    float findAccuracy(NNArray[] outputs, NNArray[] idealOutputs);

    default float findAccuracy(NNArray outputs, NNArray idealOutputs){
        return findAccuracy(new NNArray[]{outputs}, new NNArray[]{idealOutputs});
    }

    NNArray[] findDerivative(NNArray[] outputs, NNArray[] idealOutputs);

    default NNArray[] findDerivative(NNArray outputs, NNArray idealOutputs){
        return findDerivative(new NNArray[]{outputs}, new NNArray[]{idealOutputs});
    }

    class MSE implements FunctionLoss {
        private final float n;

        public MSE() {
            this(2);
        }

        public MSE(double n) {
            this.n = (float) (2.0 / n);
        }

        @Override
        public float findAccuracy(NNArray[] outputs, NNArray[] idealOutputs) {
            float accuracy = 0;

            if (Use.CPU) {
                for (int i = 0; i < outputs.length; i++) {
                    accuracy += NNArrays.sum(NNArrays.sub(idealOutputs[i], outputs[i]).pow2()) / outputs[i].size();
                }
            }

            if (Use.GPU) {
                Pointer accuracy_gpu = new Pointer();
                float[] init = new float[1];
                cudaMalloc(accuracy_gpu, (long) 1 * Sizeof.FLOAT);
                cudaMemcpy(accuracy_gpu, Pointer.to(init), (long) 1 * Sizeof.FLOAT, cudaMemcpyHostToDevice);

                for (int r = 0; r < outputs.length; r++) {
                    Pointer sum_gpu = idealOutputs[r].sum_gpu(NNArrays.sub(idealOutputs[r], outputs[r]).pow2());

                    CUfunction function = new CUfunction();
                    cuModuleGetFunction(function, helperModule, "divide_add");
                    Pointer kernelParameters = Pointer.to(Pointer.to(accuracy_gpu), Pointer.to(sum_gpu), Pointer.to(Pointer.to(new int[]{outputs[r].size()})));
                    cuLaunchKernel(function,
                            1, 1, 1,      // Grid dimension
                            1, 1, 1,      // Block dimension
                            0, null,               // Shared memory size and stream
                            kernelParameters, null // Kernel- and extra parameters
                    );

                    JCuda.cudaFree(sum_gpu);
                }

                float[] accuracyArray = new float[1];
                JCublas2.cublasGetVector(1, Sizeof.FLOAT, Pointer.to(accuracy_gpu), 1, Pointer.to(accuracyArray), 1);
                accuracy = accuracyArray[0];
            }

            return accuracy;
        }

        @Override
        public NNArray[] findDerivative(NNArray[] outputs, NNArray[] idealOutputs) {
            NNArray[] error = new NNArray[outputs.length];

            for (int i = 0; i < error.length; i++) {
                error[i] = NNArrays.sub(outputs[i], idealOutputs[i]);
                error[i].mul(n / error[i].size());
            }

            return error;
        }
    }

    class Capsule implements FunctionLoss {
        @Override
        public float findAccuracy(NNArray[] outputs, NNArray[] idealOutputs) {
            float accuracy = 0;

            for (int i = 0; i < outputs.length; i++) {
                accuracy += NNArrays.sum(NNArrays.capsLoss(idealOutputs[i], outputs[i]));
            }

            return accuracy;
        }

        @Override
        public NNArray[] findDerivative(NNArray[] outputs, NNArray[] idealOutputs) {
            NNArray[] error = new NNArray[outputs.length];

            for (int i = 0; i < error.length; i++) {
                error[i] = NNArrays.derCapsLoss(idealOutputs[i],outputs[i]);
            }

            return error;
        }
    }

    class MAE implements FunctionLoss {

        @Override
        public float findAccuracy(NNArray[] outputs, NNArray[] idealOutputs) {
            float accuracy = 0;

            for (int i = 0; i < outputs.length; i++) {
                accuracy += NNArrays.sum(NNArrays.subAbs(idealOutputs[i], outputs[i])) / outputs[i].size();
            }

            return accuracy;
        }

        @Override
        public NNArray[] findDerivative(NNArray[] outputs, NNArray[] idealOutputs) {
            NNArray[] error = new NNArray[outputs.length];

            for (int i = 0; i < error.length; i++) {
                error[i] = NNArrays.derAbs(idealOutputs[i], outputs[i]);
                error[i].mul(-1.0f / error[i].size());
            }

            return error;
        }
    }

    class BinaryCrossEntropy implements FunctionLoss {
        @Override
        public float findAccuracy(NNArray[] outputs, NNArray[] idealOutputs) {
            float accuracy = 0;
            for (int i = 0; i < outputs.length; i++) {
                accuracy -= NNArrays.sum(NNArrays.binaryCrossEntropy(idealOutputs[i], outputs[i])) / outputs[i].size();
            }

            return accuracy;
        }

        @Override
        public NNArray[] findDerivative(NNArray[] outputs, NNArray[] idealOutputs) {
            NNArray[] error = new NNArray[outputs.length];

            for (int i = 0; i < error.length; i++) {
                error[i] = NNArrays.derBinaryCrossEntropy(outputs[i], idealOutputs[i]);
                error[i].div(error[i].size());
            }

            return error;
        }
    }

    class CrossEntropy implements FunctionLoss {
        @Override
        public float findAccuracy(NNArray[] outputs, NNArray[] idealOutputs) {
            float accuracy = 0;
            for (int i = 0; i < outputs.length; i++) {
                accuracy -= NNArrays.sum(NNArrays.crossEntropy(idealOutputs[i], outputs[i])) / outputs[i].size();
            }

            return accuracy;
        }

        @Override
        public NNArray[] findDerivative(NNArray[] outputs, NNArray[] idealOutputs) {
            NNArray[] error = new NNArray[outputs.length];

            for (int i = 0; i < error.length; i++) {
                error[i] = NNArrays.derCrossEntropy(outputs[i], idealOutputs[i]);
                error[i].div(error[i].size());
            }

            return error;
        }
    }

    class CategoricalCrossEntropy implements FunctionLoss {
        @Override
        public float findAccuracy(NNArray[] outputs, NNArray[] idealOutputs) {
            float accuracy = 0;
            for (int i = 0; i < outputs.length; i++) {
                accuracy -= NNArrays.sum(NNArrays.crossEntropy(idealOutputs[i], outputs[i]));
            }

            return accuracy;
        }

        @Override
        public NNArray[] findDerivative(NNArray[] outputs, NNArray[] idealOutputs) {
            NNArray[] error = new NNArray[outputs.length];

            for (int i = 0; i < error.length; i++) {
                error[i] = NNArrays.derCrossEntropy(outputs[i], idealOutputs[i]);
            }

            return error;
        }
    }

    class Poisson implements FunctionLoss {
        @Override
        public float findAccuracy(NNArray[] outputs, NNArray[] idealOutputs) {
            float accuracy = 0;
            for (int i = 0; i < outputs.length; i++) {
                accuracy += NNArrays.sum(NNArrays.poisson(idealOutputs[i], outputs[i])) / outputs[i].size();
            }

            return accuracy;
        }

        @Override
        public NNArray[] findDerivative(NNArray[] outputs, NNArray[] idealOutputs) {
            NNArray[] error = new NNArray[outputs.length];

            for (int i = 0; i < error.length; i++) {
                error[i] = NNArrays.derPoisson(outputs[i], idealOutputs[i]);
                error[i].div(error[i].size());
            }

            return error;
        }
    }

    class KLDivergence implements FunctionLoss {
        @Override
        public float findAccuracy(NNArray[] outputs, NNArray[] idealOutputs) {
            float accuracy = 0;
            for (int i = 0; i < outputs.length; i++) {
                accuracy += NNArrays.sum(NNArrays.klDivergence(idealOutputs[i], outputs[i])) / outputs[i].size();
            }

            return accuracy;
        }

        @Override
        public NNArray[] findDerivative(NNArray[] outputs, NNArray[] idealOutputs) {
            NNArray[] error = new NNArray[outputs.length];

            for (int i = 0; i < error.length; i++) {
                error[i] = NNArrays.derCrossEntropy(outputs[i], idealOutputs[i]);
                error[i].div(error[i].size());
            }

            return error;
        }
    }

    class Hinge implements FunctionLoss {
        @Override
        public float findAccuracy(NNArray[] outputs, NNArray[] idealOutputs) {
            float accuracy = 0;
            for (int i = 0; i < outputs.length; i++) {
                accuracy += NNArrays.sum(NNArrays.hinge(idealOutputs[i], outputs[i])) / outputs[i].size();
            }

            return accuracy;
        }

        @Override
        public NNArray[] findDerivative(NNArray[] outputs, NNArray[] idealOutputs) {
            NNArray[] error = new NNArray[outputs.length];

            for (int i = 0; i < error.length; i++) {
                error[i] = NNArrays.derHinge(idealOutputs[i], outputs[i]);
                error[i].div(error[i].size());
            }

            return error;
        }
    }

    class LogCosh implements FunctionLoss {
        @Override
        public float findAccuracy(NNArray[] outputs, NNArray[] idealOutputs) {
            float accuracy = 0;
            for (int i = 0; i < outputs.length; i++) {
                accuracy += NNArrays.sum(NNArrays.logCosh(idealOutputs[i], outputs[i])) / outputs[i].size();
            }

            return accuracy;
        }

        @Override
        public NNArray[] findDerivative(NNArray[] outputs, NNArray[] idealOutputs) {
            NNArray[] error = new NNArray[outputs.length];

            for (int i = 0; i < error.length; i++) {
                error[i] = NNArrays.derLogCosh(idealOutputs[i], outputs[i]);
                error[i].div(error[i].size());
            }

            return error;
        }
    }
}
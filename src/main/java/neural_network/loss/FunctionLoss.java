package neural_network.loss;

import lombok.NoArgsConstructor;
import nnarrays.NNArray;
import nnarrays.NNArrays;

import static java.lang.Math.log;

public interface FunctionLoss {
    float findAccuracy(NNArray[] outputs, NNArray[] idealOutputs);

    NNArray[] findDerivative(NNArray[] outputs, NNArray[] idealOutputs);

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

            for (int i = 0; i < outputs.length; i++) {
                accuracy += NNArrays.sum(NNArrays.sub(idealOutputs[i], outputs[i]).pow2()) / outputs[i].size();
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
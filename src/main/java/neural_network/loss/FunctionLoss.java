package neural_network.loss;

import lombok.NoArgsConstructor;
import nnarrays.NNArray;
import nnarrays.NNArrays;

import static java.lang.Math.log;

public interface FunctionLoss {
    float findAccuracy(NNArray[] outputs, NNArray[] idealOutputs);

    NNArray[] findDerivative(NNArray[] outputs, NNArray[] idealOutputs);

    @NoArgsConstructor
    class Quadratic implements FunctionLoss {
        private float n = 2.0f;
        private float nDiv = 1;

        public Quadratic(double n) {
            this.n = (float) n;
            nDiv = (float) (2.0 / n);
        }

        @Override
        public float findAccuracy(NNArray[] outputs, NNArray[] idealOutputs) {
            float accuracy = 0;

            for (int i = 0; i < outputs.length; i++) {
                accuracy += NNArrays.sum(NNArrays.sub(idealOutputs[i], outputs[i]).pow2());
            }

            return accuracy;
        }

        @Override
        public NNArray[] findDerivative(NNArray[] outputs, NNArray[] idealOutputs) {
            NNArray[] error = new NNArray[outputs.length];

            for (int i = 0; i < error.length; i++) {
                error[i] = NNArrays.sub(outputs[i], idealOutputs[i]);
                if (nDiv != 1) {
                    error[i].mul(nDiv);
                }
            }

            return error;
        }
    }

    class BinaryCrossEntropy implements FunctionLoss {
        @Override
        public float findAccuracy(NNArray[] outputs, NNArray[] idealOutputs) {
            float accuracy = 0;
            for (int i = 0; i < outputs.length; i++) {
                for (int j = 0; j < outputs[i].size(); j++) {
                    accuracy -= idealOutputs[i].get(j) * log(outputs[i].get(j) + 0.00000001f) +
                            (1 - idealOutputs[i].get(j)) * log(1.0000001f - outputs[i].get(j));
                }
            }

            return accuracy;
        }

        @Override
        public NNArray[] findDerivative(NNArray[] outputs, NNArray[] idealOutputs) {
            NNArray[] error = new NNArray[outputs.length];

            for (int i = 0; i < error.length; i++) {
                error[i] = NNArrays.derBinaryCrossEntropy(outputs[i], idealOutputs[i]);
            }

            return error;
        }
    }

    class CrossEntropy implements FunctionLoss {
        @Override
        public float findAccuracy(NNArray[] outputs, NNArray[] idealOutputs) {
            float accuracy = 0;
            for (int i = 0; i < outputs.length; i++) {
                for (int j = 0; j < outputs[i].size(); j++) {
                    accuracy -= idealOutputs[i].get(j) * log(outputs[i].get(j) + 0.00000001f);
                }
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
}
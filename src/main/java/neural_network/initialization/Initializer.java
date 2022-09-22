package neural_network.initialization;

import lombok.NoArgsConstructor;
import nnarrays.NNArray;
import nnarrays.NNMatrix;
import nnarrays.NNTensor;

import java.util.Random;

public interface Initializer {
    void initialize(NNMatrix weight);
//    void initialize(NNTensor weight);

    @NoArgsConstructor
    class RandomNormal implements Initializer {
        private final Random random = new Random();
        private float range = 1;

        public RandomNormal(double range) {
            this.range = (float) range;
        }

        @Override
        public void initialize(NNMatrix weight) {
            for (int i = 0; i < weight.size(); i++) {
                weight.set(i, (float) (random.nextGaussian() * range));
            }
        }
    }

    @NoArgsConstructor
    class RandomUniform implements Initializer {
        private float range = 1;

        public RandomUniform(double range) {
            this.range = (float) range;
        }

        @Override
        public void initialize(NNMatrix weight) {
            for (int i = 0; i < weight.size(); i++) {
                weight.set(i, (float) ((Math.random() - 0.5) * range));
            }
        }
    }

    class XavierUniform implements Initializer {
//        @Override
//        public void initialize(NNArray weight) {
//            if (weight instanceof NNMatrix) {
//            } else if (weight instanceof NNTensor4D) {
//                value = (float) (Math.sqrt(6.0 / (((NNTensor4D) weight).getRow() * ((NNTensor4D) weight).getColumn()
//                        * (((NNTensor4D) weight).getDepth() + ((NNTensor4D) weight).getLength()))));
//            } else if (weight instanceof NNTensor) {
//                value = (float) (Math.sqrt(6.0 / (((NNTensor) weight).getColumn()
//                        * (((NNTensor) weight).getDepth() + ((NNTensor) weight).getRow()))));
//            }
//
//        }

        @Override
        public void initialize(NNMatrix weight) {
            float value = (float) (Math.sqrt(6.0 / (weight.getRow() + weight.getColumn())));

            for (int i = 0; i < weight.size(); i++) {
                weight.set(i, (float) ((Math.random() - 0.5) * value));
            }
        }
    }

    class XavierNormal implements Initializer {
        private final Random random = new Random();
//
//        @Override
//        public void initialize(NNArray weight) {
//            float value = 1;
//            if (weight instanceof NNMatrix) {
//            } else if (weight instanceof NNTensor4D) {
//                value = (float) (Math.sqrt(2.0 / (((NNTensor4D) weight).getRow() * ((NNTensor4D) weight).getColumn()
//                        * (((NNTensor4D) weight).getDepth() + ((NNTensor4D) weight).getLength()))));
//            } else if (weight instanceof NNTensor) {
//                value = (float) (Math.sqrt(2.0 / (((NNTensor) weight).getColumn()
//                        * (((NNTensor) weight).getDepth() + ((NNTensor) weight).getRow()))));
//            }
//            for (int i = 0; i < weight.getSize(); i++) {
//                weight.fill(i, (float) (random.nextGaussian() * value));
//            }
//        }

        @Override
        public void initialize(NNMatrix weight) {
            float value = (float) (Math.sqrt(2.0 / (weight.getRow() + weight.getColumn())));

            for (int i = 0; i < weight.size(); i++) {
                weight.set(i, (float) (random.nextGaussian() * value));
            }
        }
    }

    class HeUniform implements Initializer {
//        @Override
//        public void initialize(NNArray weight) {
//            float value = 1;
//            if (weight instanceof NNMatrix) {
//                value = (float) (Math.sqrt(6.0 / ((NNMatrix) weight).getRow()));
//            } else if (weight instanceof NNTensor4D) {
//                value = (float) (Math.sqrt(6.0 / (((NNTensor4D) weight).getRow() * ((NNTensor4D) weight).getColumn()
//                        * ((NNTensor4D) weight).getDepth())));
//            } else if (weight instanceof NNTensor) {
//                value = (float) (Math.sqrt(6.0 / ((NNTensor) weight).getColumn()
//                        * ((NNTensor) weight).getRow()));
//            }
//            for (int i = 0; i < weight.getSize(); i++) {
//                weight.fill(i, (float) ((Math.random() - 0.5) * value));
//            }
//        }

        @Override
        public void initialize(NNMatrix weight) {
            float value = (float) (Math.sqrt(6.0 / weight.getColumn()));

            for (int i = 0; i < weight.size(); i++) {
                weight.set(i, (float) ((Math.random() - 0.5) * value));
            }
        }
    }

    class HeNormal implements Initializer {
        private final Random random = new Random();

//        @Override
//        public void initialize(NNArray weight) {
//            float value = 1;
//            if (weight instanceof NNMatrix) {
//                value = (float) (Math.sqrt(2.0 / ((NNMatrix) weight).getRow()));
//            } else if (weight instanceof NNTensor4D) {
//                value = (float) (Math.sqrt(2.0 / (((NNTensor4D) weight).getRow() * ((NNTensor4D) weight).getColumn()
//                        * ((NNTensor4D) weight).getDepth())));
//            } else if (weight instanceof NNTensor) {
//                value = (float) (Math.sqrt(2.0 / ((NNTensor) weight).getColumn()
//                        * ((NNTensor) weight).getRow()));
//            }
//            for (int i = 0; i < weight.getSize(); i++) {
//                weight.fill(i, (float) (random.nextGaussian() * value));
//            }
//        }

        @Override
        public void initialize(NNMatrix weight) {
            float value = (float) (Math.sqrt(6.0 / weight.getColumn()));

            for (int i = 0; i < weight.size(); i++) {
                weight.set(i, (float) (random.nextGaussian() * value));
            }
        }
    }

    class LeCunUniform implements Initializer {
//        @Override
//        public void initialize(NNArray weight) {
//            float value = 1;
//            if (weight instanceof NNMatrix) {
//                value = (float) (Math.sqrt(3.0 / ((NNMatrix) weight).getColumn()));
//            } else if (weight instanceof NNTensor4D) {
//                value = (float) (Math.sqrt(3.0 / (((NNTensor4D) weight).getRow() * ((NNTensor4D) weight).getColumn()
//                        * ((NNTensor4D) weight).getLength())));
//            } else if (weight instanceof NNTensor) {
//                value = (float) (Math.sqrt(3.0 / ((NNTensor) weight).getColumn()
//                        * ((NNTensor) weight).getDepth()));
//            }
//            for (int i = 0; i < weight.getSize(); i++) {
//                weight.fill(i, (float) ((Math.random() - 0.5) * value));
//            }
//        }

        @Override
        public void initialize(NNMatrix weight) {
            float value = (float) (Math.sqrt(3.0 / weight.getRow()));

            for (int i = 0; i < weight.size(); i++) {
                weight.set(i, (float) ((Math.random() - 0.5) * value));
            }
        }
    }

    class LeCunNormal implements Initializer {
        private final Random random = new Random();
//
//        @Override
//        public void initialize(NNArray weight) {
//            float value = 1;
//            if (weight instanceof NNMatrix) {
//                value = (float) (1.0 / Math.sqrt(((NNMatrix) weight).getColumn()));
//            } else if (weight instanceof NNTensor4D) {
//                value = (float) (1.0 / Math.sqrt((((NNTensor4D) weight).getRow() * ((NNTensor4D) weight).getColumn()
//                        * ((NNTensor4D) weight).getDepth())));
//            } else if (weight instanceof NNTensor) {
//                value = (float) (1.0 / Math.sqrt(((NNTensor) weight).getColumn()
//                        * ((NNTensor) weight).getDepth()));
//            }
//
//            for (int i = 0; i < weight.getSize(); i++) {
//                weight.fill(i, (float) (random.nextGaussian() * value));
//            }
//        }

        @Override
        public void initialize(NNMatrix weight) {
            float value = (float) (1 / Math.sqrt(weight.getRow()));

            for (int i = 0; i < weight.size(); i++) {
                weight.set(i, (float) (random.nextGaussian() * value));
            }
        }
    }
}

package neural_network.initialization;

import jcuda.Pointer;
import jcuda.Sizeof;
import nnarrays.*;
import utilities.Use;

import java.util.Random;

import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;
import static nnarrays.NNArray.bFloat16ToFloat;
import static nnarrays.NNArray.floatToBFloat16;

public abstract class Initializer {
    protected float range;
    private final Random random = new Random();

    public abstract void initialize(NNVector weight);

    public abstract void initialize(NNMatrix weight);

    public abstract void initialize(NNTensor weightu);

    public abstract void initialize(NNTensor4D weight);

    protected void initializeNormal(NNArray weight) {
        if ((Use.CPU) && (!Use.GPU)) {
            for (int i = 0; i < weight.size(); i++) {
                weight.set(i, (float) (random.nextGaussian() * range));
            }
        }

        if (Use.GPU) {
            if (!weight.isTYPE()) {
                float[] temp = new float[weight.size()];
                for (int i = 0; i < weight.size(); i++) {
                    temp[i] = (float) (random.nextGaussian() * range);
                    if (Use.CPU) {
                        weight.set(i, temp[i]);
                    }
                }
                cudaMemcpy(weight.getData_gpu(), Pointer.to(temp), (long) Sizeof.FLOAT * weight.size(), cudaMemcpyHostToDevice);
            } else {
                short[] temp = new short[weight.size()];
                for (int i = 0; i < weight.size(); i++) {
                    float val = (float) (random.nextGaussian() * range);
                    temp[i] = floatToBFloat16(val);
                    if (Use.CPU) {
                        weight.set(i, val);
                    }
                }

                cudaMemcpy(weight.getData_gpu(), Pointer.to(temp), (long) Sizeof.SHORT * weight.size(), cudaMemcpyHostToDevice);
            }
        }
    }

    protected void initializeUniform(NNArray weight) {
        if (Use.CPU) {
            for (int i = 0; i < weight.size(); i++) {
                weight.set(i, (float) (random.nextGaussian() * range));
            }
        }

        if (Use.GPU) {
            if (!weight.isTYPE()) {
                float[] temp = new float[weight.size()];
                for (int i = 0; i < weight.size(); i++) {
                    temp[i] = (float) (random.nextGaussian() * range);
                    if (Use.CPU) {
                        weight.set(i, temp[i]);
                    }
                }
                cudaMemcpy(weight.getData_gpu(), Pointer.to(temp), (long) Sizeof.FLOAT * weight.size(), cudaMemcpyHostToDevice);
            } else {
                short[] temp = new short[weight.size()];
                for (int i = 0; i < weight.size(); i++) {
                    float val = (float) (random.nextGaussian() * range);
                    temp[i] = floatToBFloat16(val);
                    if (Use.CPU) {
                        weight.set(i, val);
                    }
                }

                cudaMemcpy(weight.getData_gpu(), Pointer.to(temp), (long) Sizeof.SHORT * weight.size(), cudaMemcpyHostToDevice);
            }
        }
    }

    public static class RandomNormal extends Initializer {
        public RandomNormal() {
            this(1f);
        }

        public RandomNormal(double range) {
            this.range = (float) range;
        }

        @Override
        public void initialize(NNVector weight) {
            initializeNormal(weight);
        }

        @Override
        public void initialize(NNMatrix weight) {
            initializeNormal(weight);
        }

        @Override
        public void initialize(NNTensor weight) {
            initializeNormal(weight);
        }

        @Override
        public void initialize(NNTensor4D weight) {
            initializeNormal(weight);
        }
    }

    public static class RandomUniform extends Initializer {
        public RandomUniform() {
            this(1.0);
        }

        public RandomUniform(double range) {
            this.range = (float) range;
        }

        @Override
        public void initialize(NNVector weight) {
            initializeUniform(weight);
        }

        @Override
        public void initialize(NNMatrix weight) {
            initializeUniform(weight);
        }

        @Override
        public void initialize(NNTensor weight) {
            initializeUniform(weight);
        }

        @Override
        public void initialize(NNTensor4D weight) {
            initializeUniform(weight);
        }
    }

    public static class XavierUniform extends Initializer {
        @Override
        public void initialize(NNVector weight) {
            range = (float) (Math.sqrt(6.0 / (weight.size())));
            initializeUniform(weight);
        }

        @Override
        public void initialize(NNMatrix weight) {
            range = (float) (Math.sqrt(6.0 / (weight.getRow() + weight.getColumn())));
            initializeUniform(weight);
        }

        @Override
        public void initialize(NNTensor weight) {
            range = (float) (Math.sqrt(6.0 / ((weight.getRows() + weight.getDepth()) * weight.getColumns())));
            initializeUniform(weight);
        }

        @Override
        public void initialize(NNTensor4D weight) {
            range = (float) (Math.sqrt(6.0 / ((weight.getDepth() + weight.getColumn()) * weight.getLength() * weight.row())));
            initializeUniform(weight);
        }
    }

    public static class XavierNormal extends Initializer {
        @Override
        public void initialize(NNVector weight) {
            range = (float) (Math.sqrt(2.0 / (weight.size())));
            initializeNormal(weight);
        }

        @Override
        public void initialize(NNMatrix weight) {
            range = (float) (Math.sqrt(2.0 / (weight.getRow() + weight.getColumn())));
            initializeNormal(weight);
        }

        @Override
        public void initialize(NNTensor weight) {
            range = (float) (Math.sqrt(2.0 / ((weight.getRows() + weight.getDepth()) * weight.getColumns())));
            initializeNormal(weight);
        }

        @Override
        public void initialize(NNTensor4D weight) {
            range = (float) (Math.sqrt(2.0 / ((weight.getDepth() + weight.getColumn()) * weight.getLength() * weight.row())));
            initializeNormal(weight);
        }
    }

    public static class HeUniform extends Initializer {
        @Override
        public void initialize(NNVector weight) {
            range = (float) (Math.sqrt(6.0 / weight.size()));

            initializeUniform(weight);
        }

        @Override
        public void initialize(NNMatrix weight) {
            range = (float) (Math.sqrt(6.0 / weight.getColumn()));

            initializeUniform(weight);
        }

        @Override
        public void initialize(NNTensor weight) {
            range = (float) (Math.sqrt(6.0 / (weight.getDepth() * weight.getColumns())));

            initializeUniform(weight);
        }

        @Override
        public void initialize(NNTensor4D weight) {
            range = (float) (Math.sqrt(6.0 / (weight.getLength() * weight.column() * weight.row())));

            initializeUniform(weight);
        }
    }

    public static class HeNormal extends Initializer {
        @Override
        public void initialize(NNVector weight) {
            range = (float) (Math.sqrt(2.0 / weight.size()));

            initializeNormal(weight);
        }

        @Override
        public void initialize(NNMatrix weight) {
            range = (float) (Math.sqrt(2.0 / weight.getColumn()));

            initializeNormal(weight);
        }

        @Override
        public void initialize(NNTensor weight) {
            range = (float) (Math.sqrt(2.0 / (weight.getColumns() * weight.getDepth())));

            initializeNormal(weight);
        }

        @Override
        public void initialize(NNTensor4D weight) {
            range = (float) (Math.sqrt(2.0 / (weight.getLength() * weight.column() * weight.row())));
            initializeNormal(weight);
        }
    }

    public static class LeCunUniform extends Initializer {
        @Override
        public void initialize(NNVector weight) {
            range = (float) (Math.sqrt(3.0 / weight.size()));

            initializeUniform(weight);
        }

        @Override
        public void initialize(NNMatrix weight) {
            range = (float) (Math.sqrt(3.0 / weight.getRow()));

            initializeUniform(weight);
        }

        @Override
        public void initialize(NNTensor weight) {
            range = (float) (Math.sqrt(3.0 / (weight.getRows() * weight.getColumns())));

            initializeUniform(weight);
        }

        @Override
        public void initialize(NNTensor4D weight) {
            range = (float) (Math.sqrt(3.0 / (weight.getDepth() * weight.length() * weight.getRow())));

            initializeUniform(weight);
        }
    }

    public static class LeCunNormal extends Initializer {
        @Override
        public void initialize(NNVector weight) {
            range = (float) (1 / Math.sqrt(weight.size()));

            initializeNormal(weight);
        }

        @Override
        public void initialize(NNMatrix weight) {
            range = (float) (1 / Math.sqrt(weight.getRow()));

            initializeNormal(weight);
        }

        @Override
        public void initialize(NNTensor weight) {
            range = (float) (1 / Math.sqrt(weight.getRows() * weight.getColumns()));

            initializeNormal(weight);
        }

        @Override
        public void initialize(NNTensor4D weight) {
            range = (float) (1 / Math.sqrt(weight.getDepth() * weight.row() * weight.length()));

            initializeNormal(weight);
        }
    }
}

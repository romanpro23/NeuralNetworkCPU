package neural_network.initialization;

import nnarrays.*;

import java.util.Random;

public abstract class Initializer {
    protected float range;
    private final Random random = new Random();

    public abstract void initialize(NNVector weight);

    public abstract void initialize(NNMatrix weight);

    public abstract void initialize(NNTensor weightu);

    public abstract void initialize(NNTensor4D weight);

    protected void initializeNormal(NNArray weight) {
        for (int i = 0; i < weight.size(); i++) {
            weight.set(i, (float) (random.nextGaussian() * range));
        }
    }

    protected void initializeUniform(NNArray weight) {
        for (int i = 0; i < weight.size(); i++) {
            weight.set(i, (float) ((Math.random() - 0.5) * range));
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

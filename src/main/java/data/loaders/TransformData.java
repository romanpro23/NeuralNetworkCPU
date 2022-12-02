package data.loaders;

public abstract class TransformData {
    public abstract float transform(int input);
    public abstract float transformR(int input);
    public abstract float transformG(int input);
    public abstract float transformB(int input);

    protected boolean addNoise = false;

    public TransformData addNoise(){
        this.addNoise = true;

        return this;
    }

    public static class VGG extends TransformData{

        @Override
        public float transform(int input) {
            return input;
        }

        @Override
        public float transformR(int input) {
            return input - 123.64f;
        }

        @Override
        public float transformG(int input) {
            return input - 116.779f;
        }

        @Override
        public float transformB(int input) {
            return input - 103.939f;
        }
    }

    public static class Sigmoid extends TransformData {
        private float threshold;
        private float th;

        public Sigmoid(double threshold) {
            this.threshold = (float) threshold;
            this.th = (float) (1 - threshold);
        }

        public Sigmoid() {
            threshold = 0;
            th = 1;
        }

        @Override
        public float transform(int input) {
            float data;
            if (input < 0) {
                data = (1 + input / 255.0f) * th + threshold;
            } else {
                data = input / 255.0f * th + threshold;
            }

            if (addNoise){
                data = (float) Math.max(0, Math.min(1, data + (Math.random() - 0.5f) * 0.2));
            }

            return data;
        }

        @Override
        public float transformR(int input) {
            return transform(input);
        }

        @Override
        public float transformG(int input) {
            return transform(input);
        }

        @Override
        public float transformB(int input) {
            return transform(input);
        }
    }

    public static class Tanh extends TransformData {

        @Override
        public float transform(int input) {
            float data;
            if (input < 0) {
                data = (1 + input * 2 / 255.0f);
            } else {
                data = (input *2 / 255.0f - 1);
            }
            if (addNoise){
                data = (float) Math.max(-1, Math.min(1, data + (Math.random() - 0.5f) * 0.4));
            }

            return data;
        }

        @Override
        public float transformR(int input) {
            return transform(input);
        }

        @Override
        public float transformG(int input) {
            return transform(input);
        }

        @Override
        public float transformB(int input) {
            return transform(input);
        }
    }
}

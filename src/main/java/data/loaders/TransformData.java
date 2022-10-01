package data.loaders;

public interface TransformData {
    float transform(int input);

    class Sigmoid implements TransformData {
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
            if (input < 0) {
                return (1 + input / 255.0f) * th + threshold;
            } else {
                return input / 255.0f * th + threshold;
            }
        }
    }

    class Tanh implements TransformData {

        @Override
        public float transform(int input) {
            if (input < 0) {
                return (1 + input * 2 / 255.0f);
            } else {
                return (input *2 / 255.0f - 1);
            }
        }
    }
}

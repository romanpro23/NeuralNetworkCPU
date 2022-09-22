package trainer;

import nnarrays.NNArray;
import nnarrays.NNVector;

public interface DataMetric {
    int quality(NNArray[] ideal, NNArray[] output);

    class Top1 implements DataMetric {
        @Override
        public int quality(NNArray[] ideal, NNArray[] output) {
            int counter = 0;
            for (int i = 0; i < ideal.length; i++) {
                if (ideal[i].indexMaxElement() == output[i].indexMaxElement()) {
                    counter++;
                }
            }
            return counter;
        }
    }

    class Binary implements DataMetric {
        private float threshold;

        public Binary(float threshold) {
            this.threshold = threshold;
        }

        public Binary() {
            this(0.5f);
        }

        @Override
        public int quality(NNArray[] ideal, NNArray[] output) {
            int counter = 0;
            for (int i = 0; i < ideal.length; i++) {
                if (ideal[i].get(0) == 1 && output[i].get(0) >= threshold) {
                    counter++;
                }
                if (ideal[i].get(0) == 0 && output[i].get(0) < threshold) {
                    counter++;
                }
            }
            return counter;
        }
    }
}

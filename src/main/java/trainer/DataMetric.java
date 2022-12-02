package trainer;

import nnarrays.NNArray;
import nnarrays.NNVector;

import java.util.Arrays;

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

    class Top5 implements DataMetric {
        @Override
        public int quality(NNArray[] ideal, NNArray[] output) {
            int counter = 0;
            for (int i = 0; i < ideal.length; i++) {
                int[] index = ideal[i].indexMaxElement(5);
                int real = output[i].indexMaxElement();
                for (int j = 0; j < 5; j++) {
                    if (real == index[j]) {
                        counter++;
                    }
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
                if (ideal[i].get(0) == 1.0f && output[i].get(0) >= threshold) {
                    counter++;
                    continue;
                }
                if (ideal[i].get(0) == 0 && output[i].get(0) < threshold) {
                    counter++;
                }
            }
            return counter;
        }
    }
}

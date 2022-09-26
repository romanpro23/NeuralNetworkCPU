package nnarrays;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.SneakyThrows;

import static java.lang.Math.pow;

@NoArgsConstructor
public class NNArray {
    @Getter
    protected float data[];
    protected int size;
    @Getter
    protected int countAxes;

    public NNArray(int size) {
        this.size = size;
        this.data = new float[size];
    }

    public int[] getSize(){
        return new int[]{size};
    }

    public NNArray(float[] data) {
        this.size = data.length;
        this.data = data;
    }

    public int size() {
        return size;
    }

    public void set(int i, float value) {
        data[i] = value;
    }

    public float get(int i) {
        return data[i];
    }

    public void div(float val) {
        for (int i = 0; i < size; i++) {
            data[i] /= val;
        }
    }

    public NNArray pow2() {
        for (int i = 0; i < size; i++) {
            data[i] *= data[i];
        }

        return this;
    }

    public void clip(float val) {
        clip(-val, val);
    }

    public void clip(float min, float max) {
        float a;
        for (int i = 0; i < size; i++) {
            a = data[i];
            if (a > max) {
                data[i] = max;
            } else if (a < min) {
                data[i] = min;
            }
        }
    }

    public void sqrt() {
        for (int i = 0; i < size; i++) {
            data[i] = (float) Math.sqrt(data[i] + 0.00000001f);
        }
    }

    public NNArray mul(float val) {
        for (int i = 0; i < size; i++) {
            data[i] *= val;
        }

        return this;
    }

    public NNArray mul(NNArray array) {
        for (int i = 0; i < size; i++) {
            data[i] *= array.data[i];
        }

        return this;
    }

    public void clear() {
        for (int i = 0; i < size; i++) {
            data[i] = 0;
        }
    }

    public void sub(float val) {
        add(-val);
    }

    @SneakyThrows
    public void sub(NNArray array) {
        if (size != array.size) {
            throw new Exception("Array has difference size");
        }

        for (int i = 0; i < size; i++) {
            data[i] -= array.data[i];
        }
    }

    @SneakyThrows
    public void add(NNArray array) {
        if (size != array.size) {
            throw new Exception("Array has difference size");
        }

        for (int i = 0; i < size; i++) {
            data[i] += array.data[i];
        }
    }

    public void add(float val) {
        for (int i = 0; i < size; i++) {
            data[i] += val;
        }
    }

    public void subSign(float val) {
        float a;
        for (int i = 0; i < size; i++) {
            a = data[i];
            if (a > 0) {
                data[i] -= val;
            } else if (a < 0) {
                data[i] += val;
            }
        }
    }

    public void fill(float value) {
        for (int i = 0; i < size; i++) {
            data[i] = value;
        }
    }

    public void relu(NNArray input) {
        for (int i = 0; i < size; i++) {
            data[i] = Math.max(0, input.data[i]);
        }
    }

    public void silu(NNArray input) {
        for (int i = 0; i < size; i++) {
            data[i] = (float) (data[i] / (1 + pow(Math.E, -data[i])));
        }
    }

    public void derRelu(NNArray input, NNArray error) {
        for (int i = 0; i < size; i++) {
            if (input.data[i] > 0) {
                data[i] = error.data[i];
            }
        }
    }

    public void derSilu(NNArray input, NNArray error) {
        for (int i = 0; i < size; i++) {
            data[i] = (float) (error.data[i] * ((1 + pow(Math.E, -input.data[i]) + input.data[i] * pow(Math.E, -input.data[i]))
                    / Math.pow(1 + pow(Math.E, -input.data[i]), 2)));
        }
    }

    public void derSigmoid(NNArray output, NNArray error) {
        for (int i = 0; i < size; i++) {
            data[i] = output.data[i] * (1 - output.data[i]) * error.data[i];
        }
    }

    public void derTanh(NNArray output, NNArray error) {
        for (int i = 0; i < size; i++) {
            data[i] = (1 - output.data[i] * output.data[i]) * error.data[i];
        }
    }

    public void derLeakyRelu(NNArray input, NNArray error, float param) {
        for (int i = 0; i < size; i++) {
            if (input.data[i] > 0) {
                data[i] = error.data[i];
            } else {
                data[i] = param * error.data[i];
            }
        }
    }

    public void derElu(NNArray input, NNArray error, float param) {
        for (int i = 0; i < size; i++) {
            if (input.data[i] > 0) {
                data[i] = error.data[i];
            } else {
                data[i] = (float) (param * Math.pow(Math.E, input.data[i]) * error.data[i]);
            }
        }
    }

    public void sigmoid(NNArray input) {
        for (int i = 0; i < size; i++) {
            data[i] = (float) (1.0 / (1 + Math.pow(Math.E, -input.data[i])));
        }
    }

    public void tanh(NNArray input) {
        for (int i = 0; i < size; i++) {
            data[i] = (float) Math.tanh(input.data[i]);
        }
    }

    public void linear(NNArray input) {
        System.arraycopy(input.data, 0, data, 0, size);
    }

    public void elu(NNArray input, float param) {
        for (int i = 0; i < size; i++) {
            if (input.data[i] > 0) {
                data[i] = input.data[i];
            } else {
                data[i] = (float) ((Math.pow(Math.E, input.data[i]) - 1) * param);
            }
        }
    }

    public void softplus(NNArray input) {
        for (int i = 0; i < size; i++) {
            data[i] = (float) Math.log(Math.pow(Math.E, input.data[i]) + 1);
        }
    }

    public void hardSigmoid(NNArray input) {
        for (int i = 0; i < size; i++) {
            data[i] = Math.max(0, Math.min(1, input.data[i] * 0.2f + 0.5f));
        }
    }

    public void derHardSigmoid(NNArray output, NNArray error) {
        for (int i = 0; i < size; i++) {
            if (output.data[i] >= 0 && output.data[i] <= 1) {
                data[i] = 0.2f * error.data[i];
            }
        }
    }

    public void leakyRelu(NNArray input, float param) {
        for (int i = 0; i < size; i++) {
            if (input.data[i] > 0) {
                data[i] = input.data[i];
            } else {
                data[i] = input.data[i] * param;
            }
        }
    }

    public void gaussian(NNArray input) {
        for (int i = 0; i < size; i++) {
            data[i] = (float) (Math.pow(Math.E, -input.data[i] * input.data[i]));
        }
    }

    public void derGaussian(NNArray input, NNArray error) {
        for (int i = 0; i < size; i++) {
            data[i] = (float) (-2 * input.data[i] * Math.pow(Math.E, -input.data[i] * input.data[i]) * error.data[i]);
        }
    }

    public float max() {
        float max = data[0];
        for (int i = 1; i < size; i++) {
            if (data[i] > max) {
                max = data[i];
            }
        }
        return max;
    }

    public int indexMaxElement(){
        float max = data[0];
        int index = 0;
        for (int i = 1; i < size; i++) {
            if(max < data[i]){
                index = i;
                max = data[i];
            }
        }

        return index;
    }

    public void softmax(NNArray input) {
        float sum = 0;
        float max = input.max();

        for (int i = 0; i < size; i++) {
            data[i] = (float) (Math.pow(Math.E, input.data[i] - max));
            sum += data[i];
        }
        sum += 0.00000001f;

        for (int i = 0; i < size; i++) {
            data[i] /= sum;
        }
    }

    public void derSoftmax(NNArray output, NNArray error) {
        float value;
        for (int i = 0; i < size; i++) {
            data[i] = 0;
            for (int j = 0; j < size; j++) {
                if (i != j) {
                    value = output.data[i] * -output.data[j];
                } else {
                    value = output.data[i] * (1 - output.data[i]);
                }
                data[i] += error.getData()[j] * value;
            }
        }
    }

    public void momentum(NNArray array, final float decay) {
        final float rt = 1.0f - decay;
        for (int i = 0; i < size; i++) {
            data[i] = decay * data[i] + array.data[i] * rt;
        }
    }

    public void subAndMul(NNArray vector, float val) {
        for (int i = 0; i < size; i++) {
            data[i] -= val * vector.data[i];
        }
    }

    public void momentumPow2(NNArray vector, final float decay) {
        final float dr = 1 - decay;
        for (int i = 0; i < size; i++) {
            data[i] = decay * data[i] + dr * vector.data[i] * vector.data[i];
        }
    }

    public void subDivSqrt(NNArray nominator, NNArray denominator, float lr) {
        for (int i = 0; i < size; i++) {
            data[i] -= lr * nominator.data[i] / (Math.sqrt(denominator.data[i]) + 0.0000001f);
        }
    }

    public void subDivSqrtNorm(NNArray nominator, NNArray denominator, float lr, float normN, float normD) {
        float cur_lr = lr / (normN + 0.0000001f);
        for (int i = 0; i < size; i++) {
            data[i] -= cur_lr * (nominator.data[i] ) / (Math.sqrt(denominator.data[i] / normD) + 0.0000001f);
        }
    }

    public void addPow2(NNArray vector) {
        for (int i = 0; i < size; i++) {
            data[i] += vector.data[i] * vector.data[i];
        }
    }

    public void momentumN(NNArray array, final float decay, final float lr) {
        for (int i = 0; i < size; i++) {
            data[i] = decay * data[i] - array.data[i] * lr;
        }
    }

    public void addMomentumN(NNArray derivative, NNArray decay, final float decayR, final float lr) {
        for (int i = 0; i < size; i++) {
            data[i] += decayR * decay.data[i] - derivative.data[i] * lr;
        }
    }

    public NNArray divSqrt(NNArray nominator, NNArray denominator) {
        NNArray result = new NNArray(nominator.size);
        for (int i = 0; i < size; i++) {
            result.data[i] = (float) (data[i] * Math.sqrt(nominator.data[i] + 0.0000001f) / (Math.sqrt(denominator.data[i]) + 0.0000001f));
        }
        return result;
    }


    public void dropout(NNArray input, double chanceDrop) {
        float drop = (float) (1.0f / (1.0f - chanceDrop));
        for (int i = 0; i < size; i++) {
            if (Math.random() > chanceDrop) {
                data[i] = input.data[i] * drop;
            }
        }
    }

    public void dropoutBack(NNArray output, NNArray error,  double chanceDrop) {
        float drop = (float) (1.0f / (1.0f - chanceDrop));
        for (int i = 0; i < size; i++) {
            if (output.data[i] != 0) {
                data[i] = error.data[i] * drop;
            }
        }
    }
}

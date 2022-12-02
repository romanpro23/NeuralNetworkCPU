package neural_network.activation;

import nnarrays.NNArray;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;

public interface FunctionActivation {
    void activation(NNArray input, NNArray output);

    void derivativeActivation(NNArray input, NNArray output, NNArray error, NNArray delta);

    void save(FileWriter writer) throws IOException;

    static FunctionActivation read(Scanner scanner){
        FunctionActivation functionActivation;

        String activ = scanner.nextLine();
        functionActivation = switch (activ) {
            case "ReLU" -> new ReLU();
            case "ReLUMax" -> new ReLUMax(Double.parseDouble(scanner.nextLine()));
            case "Linear" -> new Linear();
            case "SiLU" -> new SiLU();
            case "LeakyReLU" -> new LeakyReLU(Double.parseDouble(scanner.nextLine()));
            case "SineReLU" -> new SineReLU(Double.parseDouble(scanner.nextLine()));
            case "ELU" -> new ELU(Double.parseDouble(scanner.nextLine()));
            case "Sigmoid" -> new Sigmoid();
            case "Gaussian" -> new Gaussian();
            case "HardSigmoid" -> new HardSigmoid();
            case "Softmax" -> new Softmax();
            case "Softplus" -> new Softplus();
            case "Tanh" -> new Tanh();
            default -> null;
        };

        return functionActivation;
    }

    class ReLU implements FunctionActivation {
        @Override
        public void activation(NNArray input, NNArray output) {
            output.relu(input);
        }

        @Override
        public void derivativeActivation(NNArray input, NNArray output, NNArray error, NNArray delta) {
            delta.derRelu(input, error);
        }

        @Override
        public void save(FileWriter writer) throws IOException {
            writer.write("ReLU\n");
        }
    }

    class ReLUMax implements FunctionActivation {
        private final float max;

        public ReLUMax() {
            this(5);
        }

        public ReLUMax(double max) {
            this.max = (float) max;
        }

        @Override
        public void activation(NNArray input, NNArray output) {
            output.relu_max(input, max);
        }

        @Override
        public void derivativeActivation(NNArray input, NNArray output, NNArray error, NNArray delta) {
            delta.derReluMax(input, error, max);
        }

        @Override
        public void save(FileWriter writer) throws IOException {
            writer.write("ReLUMax\n");
            writer.write(max + "\n");
        }
    }

    class SineReLU implements FunctionActivation {
        private final float epsilon;

        public SineReLU(double epsilon) {
            this.epsilon = (float) epsilon;
        }

        public SineReLU() {
            this.epsilon = 0.025f;
        }

        @Override
        public void activation(NNArray input, NNArray output) {
            output.sineRelu(input, epsilon);
        }

        @Override
        public void derivativeActivation(NNArray input, NNArray output, NNArray error, NNArray delta) {
            delta.derSineRelu(input, error, epsilon);
        }

        @Override
        public void save(FileWriter writer) throws IOException {
            writer.write("SineReLU\n");
            writer.write(epsilon + "\n");
        }
    }

    class Linear implements FunctionActivation {
        @Override
        public void activation(NNArray input, NNArray output) {
            output.linear(input);
        }

        @Override
        public void derivativeActivation(NNArray input, NNArray output, NNArray error, NNArray delta) {
            delta.linear(error);
        }

        @Override
        public void save(FileWriter writer) throws IOException {
            writer.write("Linear\n");
        }
    }

    class Softplus implements FunctionActivation {
        @Override
        public void activation(NNArray input, NNArray output) {
            output.softplus(input);
        }

        @Override
        public void derivativeActivation(NNArray input, NNArray output, NNArray error, NNArray delta) {
            delta.sigmoid(input);
            delta.mul(error);
        }

        @Override
        public void save(FileWriter writer) throws IOException {
            writer.write("Softplus\n");
        }
    }

    class SiLU implements FunctionActivation {
        @Override
        public void activation(NNArray input, NNArray output) {
            output.silu(input);
        }

        @Override
        public void derivativeActivation(NNArray input, NNArray output, NNArray error, NNArray delta) {
            delta.derSilu(input, error);
        }

        @Override
        public void save(FileWriter writer) throws IOException {
            writer.write("SiLU\n");
        }
    }

    class Sigmoid implements FunctionActivation {
        @Override
        public void activation(NNArray input, NNArray output) {
            output.sigmoid(input);
        }

        @Override
        public void derivativeActivation(NNArray input, NNArray output, NNArray error, NNArray delta) {
            delta.derSigmoid(output, error);
        }

        @Override
        public void save(FileWriter writer) throws IOException {
            writer.write("Sigmoid\n");
        }
    }

    class Gaussian implements FunctionActivation {
        @Override
        public void activation(NNArray input, NNArray output) {
            output.gaussian(input);
        }

        @Override
        public void derivativeActivation(NNArray input, NNArray output, NNArray error, NNArray delta) {
            delta.derGaussian(input, error);
        }

        @Override
        public void save(FileWriter writer) throws IOException {
            writer.write("Gaussian\n");
        }
    }

    class HardSigmoid implements FunctionActivation {
        @Override
        public void activation(NNArray input, NNArray output) {
            output.hardSigmoid(input);
        }

        @Override
        public void derivativeActivation(NNArray input, NNArray output, NNArray error, NNArray delta) {
            delta.derHardSigmoid(output, error);
        }

        @Override
        public void save(FileWriter writer) throws IOException {
            writer.write("HardSigmoid\n");
        }
    }

    class Softmax implements FunctionActivation {
        @Override
        public void activation(NNArray input, NNArray output) {
            output.softmax(input);
        }

        @Override
        public void derivativeActivation(NNArray input, NNArray output, NNArray error, NNArray delta) {
            delta.derSoftmax(output, error);
        }

        @Override
        public void save(FileWriter writer) throws IOException {
            writer.write("Softmax\n");
        }
    }

    class Tanh implements FunctionActivation {
        @Override
        public void activation(NNArray input, NNArray output) {
            output.tanh(input);
        }

        @Override
        public void derivativeActivation(NNArray input, NNArray output, NNArray error, NNArray delta) {
            delta.derTanh(output, error);
        }

        @Override
        public void save(FileWriter writer) throws IOException {
            writer.write("Tanh\n");
        }
    }

    class LeakyReLU implements FunctionActivation {
        private final float param;

        public LeakyReLU(double param) {
            this.param = (float) param;
        }

        public LeakyReLU() {
            this.param = 0.01f;
        }

        @Override
        public void activation(NNArray input, NNArray output) {
            output.leakyRelu(input, param);
        }

        @Override
        public void derivativeActivation(NNArray input, NNArray output, NNArray error, NNArray delta) {
            delta.derLeakyRelu(input, error, param);
        }

        @Override
        public void save(FileWriter writer) throws IOException {
            writer.write("LeakyReLU\n");
            writer.write(param + "\n");
        }
    }

    class ELU implements FunctionActivation {
        private final float param;

        public ELU(double param) {
            this.param = (float) param;
        }

        public ELU() {
            this.param = 0.01f;
        }

        @Override
        public void activation(NNArray input, NNArray output) {
            output.elu(input, param);
        }

        @Override
        public void derivativeActivation(NNArray input, NNArray output, NNArray error, NNArray delta) {
            delta.derElu(input, error, param);
        }

        @Override
        public void save(FileWriter writer) throws IOException {
            writer.write("ELU\n");
            writer.write(param + "\n");
        }
    }
}

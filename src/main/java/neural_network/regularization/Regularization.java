package neural_network.regularization;

import lombok.AllArgsConstructor;
import lombok.Getter;
import nnarrays.NNArray;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;

public interface Regularization {
    void regularization(NNArray array);

    void write(FileWriter fileWriter) throws IOException;

    public static Regularization read(Scanner scanner) {
        Regularization regularization;

        String reg = scanner.nextLine();
        regularization = switch (reg) {
            case "L1" -> new L1(Double.parseDouble(scanner.nextLine()));
            case "L2" -> new L2(Double.parseDouble(scanner.nextLine()));
            case "ElasticNet" -> new ElasticNet(
                    Double.parseDouble(scanner.nextLine()),
                    Double.parseDouble(scanner.nextLine())
            );
            default -> null;
        };

        return regularization;
    }

    @AllArgsConstructor
    class L1 implements Regularization {
        @Getter
        private final float lambda;

        public L1(double lambda) {
            this.lambda = (float) lambda;
        }

        @Override
        public void regularization(NNArray array) {
            array.subSign(lambda);
        }

        @Override
        public void write(FileWriter fileWriter) throws IOException {
            fileWriter.write("L1");
            fileWriter.write(lambda + "\n");
        }
    }

    class L2 implements Regularization {
        @Getter
        private final float lambda;
        private final float subLambda;

        public L2(double lambda) {
            this((float) lambda);
        }

        public L2(float lambda) {
            this.lambda = lambda;
            subLambda = 1.0f - lambda;
        }

        @Override
        public void regularization(NNArray array) {
            array.mul(subLambda);
        }

        @Override
        public void write(FileWriter fileWriter) throws IOException {
            fileWriter.write("L2");
            fileWriter.write(lambda + "\n");
        }
    }

    class ElasticNet implements Regularization {
        @Getter
        private final float lambda1;
        private final float subLambda;
        @Getter
        private final float lambda2;

        public ElasticNet(double lambda) {
            this(lambda, lambda);
        }

        public ElasticNet(double lambda1, double lambda2) {
            this.lambda1 = (float) lambda1;
            this.lambda2 = (float) lambda2;
            subLambda = (float) (1.0 - lambda1);
        }

        @Override
        public void regularization(NNArray array) {
            array.mul(subLambda);
            array.subSign(lambda2);
        }

        @Override
        public void write(FileWriter fileWriter) throws IOException {
            fileWriter.write("ElasticNet");
            fileWriter.write(lambda1 + "\n");
            fileWriter.write(lambda2 + "\n");
        }
    }
}

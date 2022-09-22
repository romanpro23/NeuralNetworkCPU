package trainer;

import data.mnist.DataLoader1D;
import data.mnist.NNData1D;
import neural_network.network.NeuralNetwork;

public class DataTrainer {
    private DataLoader1D loader;

    private final int sizeTrainEpoch;
    private final int sizeTestEpoch;

    public DataTrainer(int sizeTrainEpoch, int sizeTestEpoch, DataLoader1D loader) {
        this.loader = loader;
        this.sizeTrainEpoch = sizeTrainEpoch;
        this.sizeTestEpoch = sizeTestEpoch;
    }

    public float train(NeuralNetwork network, int sizeBatch, int countEpoch, DataMetric dataMetric) {
        int counter = 0;
        for (int i = 0; i < countEpoch; i++) {
            counter = 0;
            double accuracy = 0;
            for (int j = 0; j < (int) Math.ceil(sizeTrainEpoch * 1.0 / sizeBatch); j++) {
                NNData1D data = loader.getNextTrainData(Math.min(sizeBatch, sizeTrainEpoch - j * sizeBatch));
                accuracy += network.train(data.getInput(), data.getOutput());
                counter += dataMetric.quality(data.getOutput(), network.getOutputs());
            }
            System.out.println("\t\t\t" + (i + 1) + " ЕПОХА ");
            System.out.println("Результат навчального датасету: ");
            System.out.println("Відсоток правильних відповідей: " + String.format("%.2f", counter * 1.0 / sizeTrainEpoch * 100) + " %");
            System.out.println("Точність на навчальном датасеті: " + String.format("%.5f", accuracy / sizeTrainEpoch));

            counter = 0;
            accuracy = 0;
            for (int j = 0; j < (int) Math.ceil(sizeTestEpoch * 1.0 / sizeBatch); j++) {
                NNData1D data = loader.getNextTestData(Math.min(sizeBatch, sizeTestEpoch - j * sizeBatch));
                counter += dataMetric.quality(data.getOutput(), network.query(data.getInput()));
                accuracy += network.accuracy(data.getOutput());
            }
            System.out.println("Результат тренувального датасету: ");
            System.out.println("Відсоток правильних відповідей: " + String.format("%.2f", counter * 1.0 / sizeTestEpoch * 100) + " %");
            System.out.println("Точність на тренувальном датасеті: " + String.format("%.5f", accuracy / sizeTestEpoch));
            System.out.println();
        }
        return counter * 1.0f / sizeTestEpoch * 100;
    }
}

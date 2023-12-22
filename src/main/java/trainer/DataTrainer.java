package trainer;

import data.loaders.DataLoader;
import data.network_train.NNData;
import neural_network.network.NeuralNetwork;
import nnarrays.NNArray;
import nnarrays.NNMatrix;
import utilities.Use;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static neural_network.layers.NeuralLayer.CallGarbageCollector;

public class DataTrainer {
    private DataLoader loader;

    private final int sizeTrainEpoch;
    private final int sizeTestEpoch;
    private int sizeBatch;

    public DataTrainer(int sizeTrainEpoch, int sizeTestEpoch, DataLoader loader) {
        this.loader = loader;
        this.sizeTrainEpoch = sizeTrainEpoch;
        this.sizeTestEpoch = sizeTestEpoch;
        this.sizeBatch = 64;
    }

    public float train(NeuralNetwork network, int sizeBatch, int countEpoch, DataMetric dataMetric) {
        return train(network, sizeBatch, countEpoch, 1, dataMetric);
    }

    public float train(NeuralNetwork network, int sizeBatch, int countEpoch, int countUpdate, DataMetric dataMetric)  {
        this.sizeBatch = sizeBatch;
        int counter = 0;
        int lambda = 1;
        for (int i = 0; i < countEpoch; i++) {
            counter = 0;
            double accuracy = 0;
            int max = sizeTrainEpoch / sizeBatch;
            int cu = 0;
            int cunt = 0;
            int index = 0;
            System.out.print(" [");
            int ddd = 1 * countEpoch;
            int dd = sizeTrainEpoch / ddd;
            for (int j = 0; j < max; j++) {
                NNData data = loader.getNextTrainData(Math.min(sizeBatch, sizeTrainEpoch - j * sizeBatch));
                network.train(data.getInput(), data.getOutput(), false, lambda);
                cu++;
                cunt++;
                if(cu == countUpdate){
                    network.update();
                    cu = 0;
                }

                if(cunt == ddd){
                    CallGarbageCollector();
                    accuracy += lambda * network.accuracy(data.getOutput());
                    counter += dataMetric.quality(data.getOutput(), network.getOutputs());
                    cunt = 0;
                    index++;
                    System.out.print("\r" + (j * sizeBatch) + " ");
                }
                //if(j % Math.max(1, (max / 26)) == 0) {
                //    System.out.print("=");
                //}

                /*try {
                    TimeUnit.MILLISECONDS.sleep(100);
                }
                catch(InterruptedException ex)
                {

                }*/
            }
            System.out.println("]");
            System.out.println("\t\t\t" + (i + 1) + " ერა ");
            System.out.println("ტრენინგის მონაცემთა ნაკრების შედეგი: ");
            System.out.println("სწორი პასუხების პროცენტი (მხოლოდ კლასიფიკაციისთვის)" + String.format("%.2f", counter * 1.0 / dd * 100) + " %");
            System.out.println("სიზუსტე ტრენინგის მონაცემთა ბაზაში: " + String.format("%.5f", accuracy / dd));

            //score(network, dataMetric);
        }
        return counter * 1.0f / sizeTrainEpoch * 100;
    }

    public float score(NeuralNetwork network, DataMetric dataMetric) {
        int counter = 0;
        double accuracy = 0;
        for (int j = 0; j < (int) Math.ceil(sizeTestEpoch * 1.0 / sizeBatch); j++) {
            NNData data = loader.getNextTestData(Math.min(sizeBatch, sizeTestEpoch - j * sizeBatch));
            counter += dataMetric.quality(data.getOutput(), network.query(data.getInput()));
            accuracy += network.accuracy(data.getOutput());
        }
        System.out.println("Результат тренувального датасету: ");
        System.out.println("Відсоток правильних відповідей: " + String.format("%.2f", counter * 1.0 / sizeTestEpoch * 100) + " %");
        System.out.println("Точність на тренувальном датасеті: " + String.format("%.5f", accuracy / sizeTestEpoch));
        System.out.println();
        return counter * 1.0f / sizeTestEpoch * 100;
    }
}

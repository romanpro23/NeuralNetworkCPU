package neural_network.network.GAN;

import data.gan.GANGeneratorData;
import data.network_train.NNData;
import lombok.Getter;
import lombok.SneakyThrows;
import neural_network.initialization.Initializer;
import neural_network.network.NeuralNetwork;
import nnarrays.*;

public class GAN {
    @Getter
    private NeuralNetwork generator;
    private NeuralNetwork discriminator;

    private Initializer initializer;

    public GAN(NeuralNetwork generator, NeuralNetwork discriminator) {
        this.generator = generator;
        this.discriminator = discriminator;

        initializer = new Initializer.RandomNormal();
    }

    public NNArray[] query(NNArray[] input) {
        return generator.query(input);
    }

    @SneakyThrows
    private NNArray[] randomDataGenerator(int sizeBatch) {
        int[] size = generator.getInputSize();
        if (size.length == 1) {
            NNVector[] random = new NNVector[sizeBatch];
            for (int i = 0; i < random.length; i++) {
                random[i] = new NNVector(size[0]);
                initializer.initialize(random[i]);
            }
            return random;
        } else if (size.length == 2) {
            NNMatrix[] random = new NNMatrix[sizeBatch];
            for (int i = 0; i < random.length; i++) {
                random[i] = new NNMatrix(size[0], size[1]);
                initializer.initialize(random[i]);
            }
            return random;
        } else if (size.length == 3) {
            NNTensor[] random = new NNTensor[sizeBatch];
            for (int i = 0; i < random.length; i++) {
                random[i] = new NNTensor(size[0], size[1], size[2]);
                initializer.initialize(random[i]);
            }
            return random;
        }
        throw new Exception("Error dimension generator!");
    }

    public float train(NNArray[] input) {
        float accuracy = 0;
        //generate input data for discriminator
        NNArray[] random = randomDataGenerator(input.length);
        NNArray[] fake = generator.query(random);
        NNData data = GANGeneratorData.generateData(input, fake);

        //train discriminator
        accuracy += discriminator.train(data.getInput(), data.getOutput());

        //generate data for generator
        random = randomDataGenerator(input.length);
        NNVector realLabel = new NNVector(new float[]{1});
        NNVector[] realLabels = new NNVector[random.length];
        for (int i = 0; i < random.length; i++) {
            realLabels[i] = realLabel;
        }

        //train generator
        generator.queryTrain(random);
        discriminator.setTrainable(false);
        discriminator.forwardBackpropagation(generator.getOutputs(), realLabels);
        generator.train(discriminator.getError());
        discriminator.setTrainable(true);

        //accuracy generator
        for (int i = 0; i < discriminator.getError().length; i++) {
            accuracy += Math.abs(NNArrays.sum(discriminator.getError()[i]));
        }
        return accuracy;
    }
}

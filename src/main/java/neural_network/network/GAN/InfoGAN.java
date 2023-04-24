package neural_network.network.GAN;

import lombok.Getter;
import lombok.SneakyThrows;
import neural_network.initialization.Initializer;
import neural_network.network.NeuralNetwork;
import nnarrays.NNArray;
import nnarrays.NNMatrix;
import nnarrays.NNTensor;
import nnarrays.NNVector;

public class InfoGAN {
    @Getter
    protected final ConditionalGenerator generator;
    protected final ClassifierDiscriminator discriminator;

    private Initializer initializer;

    public InfoGAN(ConditionalGenerator generator, ClassifierDiscriminator discriminator) {
        this.generator = generator;
        this.discriminator = discriminator;

        initializer = new Initializer.RandomNormal();
    }

    public NNArray[] query(NNArray[] noise, NNArray[] labels) {
        return generator.query(noise, labels);
    }

    @SneakyThrows
    protected NNArray[] randomDataGenerator(int sizeBatch) {
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

    protected NNArray[] getRealLabel(int size) {
        NNArray[] result = getFakeLabel(size);
        for (int i = 0; i < size; i++) {
            result[i].fill(1);
        }

        return result;
    }

    @SneakyThrows
    protected NNArray[] getFakeLabel(int size) {
        int[] sizeD = discriminator.getOutputSize();
        if (sizeD.length == 1) {
            NNVector[] result = new NNVector[size];
            for (int i = 0; i < size; i++) {
                result[i] = new NNVector(sizeD[0]);
            }

            return result;
        } else if (sizeD.length == 2) {
            NNMatrix[] result = new NNMatrix[size];
            for (int i = 0; i < size; i++) {
                result[i] = new NNMatrix(sizeD[0], sizeD[1]);
            }
            return result;
        } else if (sizeD.length == 3) {
            NNTensor[] result = new NNTensor[size];
            for (int i = 0; i < size; i++) {
                result[i] = new NNTensor(sizeD[0], sizeD[1], sizeD[2]);
            }
            return result;
        }
        throw new Exception("Error dimension generator!");
    }

    public final float train(NNArray[] input, NNArray[] labels) {
        return train(input, labels, 1);
    }

    public final float train(NNArray[] input, NNArray[] labels, float lambda) {
        //generate input data for discriminator
        NNArray[] random = randomDataGenerator(input.length);

        //trainA discriminator
        float accuracyD = discriminator.train(generator.queryTrain(random, labels), labels, getFakeLabel(input.length), false, lambda);
        accuracyD += discriminator.train(input, labels, getRealLabel(input.length), false, lambda);
        discriminator.update();

        //generate data for generator
        random = randomDataGenerator(input.length);

        //trainA generator
        discriminator.setTrainable(false);
        generator.queryTrain(random, labels);
        //accuracy generator
        float accuracyG = discriminator.train(generator.getOutputs(), labels, getRealLabel(input.length), false, lambda);
        generator.train(discriminator.getError());
        discriminator.setTrainable(true);

        return accuracyD + accuracyG;
    }

    public InfoGAN setInitializer(Initializer initializer) {
        this.initializer = initializer;

        return this;
    }
}

package neural_network.network.GAN;

import data.gan.GANGeneratorData;
import data.network_train.NNData;
import lombok.SneakyThrows;
import neural_network.initialization.Initializer;
import neural_network.network.NeuralNetwork;
import nnarrays.*;

public class Pix2PixGAN {
    private NeuralNetwork generator;
    private NeuralNetwork discriminator;

    public Pix2PixGAN(NeuralNetwork generator, NeuralNetwork discriminator) {
        this.generator = generator;
        this.discriminator = discriminator;
    }

    public NNArray[] query(NNArray[] input) {
        return generator.query(input);
    }

    public float train(NNArray[] input, NNArray[] output) {
        return train(input, output, 1);
    }

    public float train(NNArray[] input, NNArray[] output, float lambda) {
        float accuracy = 0;
        //generate input data for discriminator
        NNArray[] fake = generator.query(input);

        NNArray[] realInput = NNArrays.concat(input, output);
        NNArray[] fakeInput = NNArrays.concat(input, fake);

        NNData data = GANGeneratorData.generateData(realInput, fakeInput);

        //train discriminator
        accuracy += discriminator.train(data.getInput(), data.getOutput());

        //generate data for generator
        NNVector[] realLabels = new NNVector[input.length];
        for (int i = 0; i < input.length; i++) {
            realLabels[i] = new NNVector(new float[]{1});
        }

        //train generator
        generator.queryTrain(input);
        fakeInput = NNArrays.concat(generator.getOutputs(), input);
        discriminator.setTrainable(false);
        //accuracy generator
        accuracy += discriminator.forwardBackpropagation(fakeInput, realLabels);
        NNArray[] errorGenerator = NNArrays.subArray(discriminator.getError(), generator.getOutputs());
        NNArray[] errorRecognition = generator.findDerivative(output);
        for (int i = 0; i < errorRecognition.length; i++) {
            errorRecognition[i].mul(lambda);
            errorGenerator[i].add(errorRecognition[i]);
        }
        generator.train(errorGenerator);
        discriminator.setTrainable(true);

        return accuracy;
    }
}

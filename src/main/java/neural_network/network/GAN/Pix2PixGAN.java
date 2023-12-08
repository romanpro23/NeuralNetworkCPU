package neural_network.network.GAN;

import data.ImageCreator;
import data.gan.GANGeneratorData;
import data.network_train.NNData;
import lombok.SneakyThrows;
import neural_network.initialization.Initializer;
import neural_network.network.NeuralNetwork;
import nnarrays.*;

public class Pix2PixGAN extends GAN {

    public Pix2PixGAN(NeuralNetwork generator, NeuralNetwork discriminator) {
        super(generator, discriminator);
    }

    public float train(NNArray[] input, NNArray[] output) {
        return train(input, output, 1);
    }

    public float train(NNArray[] input, NNArray[] output, float lambda) {
        //generate input data for discriminator
        NNArray[] fake = generator.queryTrain(input);

        NNArray[] realInput = NNArrays.concat(output, input);
        NNArray[] fakeInput = NNArrays.concat(fake, input);

        //trainA discriminator
        float accuracyD = 0;
        float accuracyG = 0;
        /*accuracyD += discriminator.train(realInput, getRealLabel(input.length), false);
        accuracyD += discriminator.train(fakeInput, getFakeLabel(input.length), false);

        //trainA generator
        discriminator.setTrainable(false);
        //accuracy generator
        float accuracyG = discriminator.train(fakeInput, getRealLabel(input.length), false);
        NNArray[] errorGenerator = NNArrays.subArray(discriminator.getError(), generator.getOutputs());
        NNArray[] errorRecognition = generator.findDerivative(output, lambda);
        accuracyG += generator.accuracy(output);
        for (int i = 0; i < errorRecognition.length; i++) {
            errorGenerator[i].add(errorRecognition[i]);
        }
        generator.train(errorGenerator);

        discriminator.setTrainable(true);
        discriminator.update();*/

        return accuracyD + accuracyG;
    }
}

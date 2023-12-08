package neural_network.network.GAN;

import data.ImageCreator;
import neural_network.network.NeuralNetwork;
import nnarrays.NNArray;
import nnarrays.NNArrays;
import nnarrays.NNTensor;

public class SRGAN extends GAN {

    public SRGAN(NeuralNetwork generator, NeuralNetwork discriminator) {
        super(generator, discriminator);
    }

    public float train(NNArray[] input, NNArray[] output) {
        return train(input, output, 1);
    }

    public float train(NNArray[] input, NNArray[] output, float lambda) {
        //generate input data for discriminator
        NNArray[] fake = generator.queryTrain(input);

        //trainA discriminator
        float accuracyD = 0;
        float accuracyG = 0;
        /*accuracyD += discriminator.train(output, getRealLabel(input.length), false);
        accuracyD += discriminator.train(fake, getFakeLabel(input.length), false);

        //trainA generator
        discriminator.setTrainable(false);
        //accuracy generator
        float accuracyG = discriminator.train(fake, getRealLabel(input.length), false, lambda);
        NNArray[] errorRecognition = generator.findDerivative(output);
        accuracyG += generator.accuracy(output);
        for (int i = 0; i < errorRecognition.length; i++) {
            errorRecognition[i].add(discriminator.getError()[i]);
        }
        generator.train(errorRecognition);

        discriminator.setTrainable(true);
        discriminator.update();*/

        return accuracyD + accuracyG;
    }
}

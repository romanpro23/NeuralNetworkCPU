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
        accuracyD += discriminator.train(realInput, getRealLabel(input.length), false);
        accuracyD += discriminator.train(fakeInput, getFakeLabel(input.length), false);

        //trainA generator
        discriminator.setTrainable(false);
        //accuracy generator
        float accuracyG = discriminator.train(fakeInput, getRealLabel(input.length), false);
        NNArray[] errorGenerator = NNArrays.subArray(discriminator.getError(), generator.getOutputs());
        NNArray[] errorRecognition = generator.findDerivative(output);
        accuracyG += generator.accuracy(output);
        for (int i = 0; i < errorRecognition.length; i++) {
            if(lambda != 1) {
                errorRecognition[i].mul(lambda);
            }
            errorGenerator[i].add(errorRecognition[i]);
        }
        generator.train(errorGenerator);
        errorGenerator[0].clip(-1, 1);
        ImageCreator.drawColorImage((NNTensor) errorGenerator[0], 64, 64, "_error", "D:/NetworkTest/Decolorize/Pix2Pix", true);
        ImageCreator.drawColorImage((NNTensor) output[0], 64, 64, "_output", "D:/NetworkTest/Decolorize/Pix2Pix", true);
        ImageCreator.drawColorImage((NNTensor) fake[0], 64, 64, "_test", "D:/NetworkTest/Decolorize/Pix2Pix", true);
        ImageCreator.drawColorImage((NNTensor) input[0], 64, 64, "_input", "D:/NetworkTest/Decolorize/Pix2Pix", true);

        discriminator.setTrainable(true);
        discriminator.update();

        return accuracyD + accuracyG;
    }
}

package neural_network.network.autoencoders;

import data.gan.GANGeneratorData;
import data.network_train.NNData;
import lombok.SneakyThrows;
import neural_network.initialization.Initializer;
import neural_network.network.NeuralNetwork;
import neural_network.optimizers.Optimizer;
import nnarrays.NNArray;
import nnarrays.NNVector;

public class AdversarialAutoencoder {
    private final NeuralNetwork decoder;
    private final NeuralNetwork discriminator;
    private final NeuralNetwork encoder;

    private Optimizer optimizerDecode, optimizerDistribution;

    public AdversarialAutoencoder(NeuralNetwork encoder, NeuralNetwork decoder, NeuralNetwork discriminator) {
        this.decoder = decoder;
        this.encoder = encoder;
        this.discriminator = discriminator;

        optimizerDecode = null;
        optimizerDistribution = null;
    }

    public AdversarialAutoencoder setOptimizersEncoder(Optimizer optimizerDecode, Optimizer optimizerDistribution) {
        this.optimizerDistribution = optimizerDistribution;
        encoder.initialize(optimizerDistribution);
        this.optimizerDecode = optimizerDecode;
        encoder.initialize(optimizerDecode);

        return this;
    }

    public NNArray[] query(NNArray[] input) {
        return decoder.query(encoder.query(input));
    }

    public NNArray[] queryDecoder(NNArray[] input) {
        return decoder.query(input);
    }

    public float train(NNArray[] input, NNVector[] distribution) {
        return train(input, input, distribution);
    }

    @SneakyThrows
    public float train(NNArray[] input) {
        NNVector[] distribution = new NNVector[input.length];
        Initializer initializer = new Initializer.RandomNormal();
        for (int i = 0; i < distribution.length; i++) {
            int[] size = encoder.getLayers().get(encoder.getLayers().size() - 1).size();
            if (size.length > 1) {
                throw new Exception("Layer must be flat");
            }
            distribution[i] = new NNVector(size[0]);
            initializer.initialize(distribution[i]);
        }

        return train(input, distribution);
    }

    public float train(NNArray[] input, NNArray[] output, NNVector[] distribution) {
        float accuracy = 0;
        accuracy += trainDecoder(input, output);
        accuracy += trainDiscriminator(input, distribution);

        return accuracy;
    }

    public float trainDecoder(NNArray[] input, NNArray[] output) {
        encoder.queryTrain(input);
        float accuracy = decoder.train(encoder.getOutputs(), output);

        encoder.setOptimizer(optimizerDecode);
        encoder.train(decoder.getError());
        return accuracy;
    }

    public float trainDiscriminator(NNArray[] input, NNVector[] distribution) {
        //generate input data for discriminator
        encoder.queryTrain(input);
        NNArray[] fake = encoder.getOutputs();
        NNData data = GANGeneratorData.generateData(distribution, fake);

        //trainA discriminator
        float accuracy = discriminator.train(data.getInput(), data.getOutput());

        //generate data for generator
        NNVector[] label = new NNVector[input.length];
        for (int i = 0; i < label.length; i++) {
            label[i] = new NNVector(new float[]{1});
        }

        //trainA generator
        discriminator.setTrainable(false);
        discriminator.forwardBackpropagation(encoder.getOutputs(), label);

        encoder.setOptimizer(optimizerDistribution);
        encoder.train(discriminator.getError());
        discriminator.setTrainable(true);

        return accuracy;
    }
}

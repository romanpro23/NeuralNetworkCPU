package neural_network.network.autoencoders;

import data.gan.GANGeneratorData;
import data.network_train.NNData;
import lombok.SneakyThrows;
import neural_network.initialization.Initializer;
import neural_network.network.NeuralNetwork;
import neural_network.optimizers.Optimizer;
import nnarrays.NNArray;
import nnarrays.NNVector;

public class SSAdversarialAutoencoder {
    private final NeuralNetwork decoder;
    private final NeuralNetwork styleDiscriminator;
    private final NeuralNetwork labelDiscriminator;
    private final NeuralNetwork encoder;

    private Optimizer optimizerDecode, optimizerDistribution, optimizerLabel;

    public SSAdversarialAutoencoder(NeuralNetwork encoder, NeuralNetwork decoder, NeuralNetwork styleDiscriminator, NeuralNetwork labelDiscriminator) {
        this.decoder = decoder;
        this.encoder = encoder;
        this.styleDiscriminator = styleDiscriminator;
        this.labelDiscriminator = labelDiscriminator;

        optimizerDecode = null;
        optimizerDistribution = null;
    }

    public SSAdversarialAutoencoder setOptimizersEncoder(Optimizer optimizerDecode, Optimizer optimizerDistribution) {
        this.optimizerDistribution = optimizerDistribution;
        this.optimizerDecode = optimizerDecode;
        this.optimizerLabel = optimizerDistribution;

        return this;
    }

    public SSAdversarialAutoencoder setOptimizersEncoder(Optimizer optimizerDecode, Optimizer optimizerDistribution, Optimizer optimizerLabel) {
        this.optimizerDistribution = optimizerDistribution;
        this.optimizerDecode = optimizerDecode;
        this.optimizerLabel = optimizerLabel;

        return this;
    }

    public NNArray[] query(NNArray[] input) {
        return decoder.query(encoder.query(input));
    }

    public NNArray[] queryDecoder(NNArray[] input) {
        return decoder.query(input);
    }

    public float train(NNArray[] input, NNArray[] label, NNVector[] distribution) {
        return train(input, input, label, distribution);
    }

    @SneakyThrows
    public float train(NNArray[] input, NNArray[] label) {
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

        return train(input, label, distribution);
    }

    public float train(NNArray[] input, NNArray[] output,  NNArray[] label, NNVector[] distribution) {
        float accuracy = 0;
        accuracy += trainDecoder(input, output);
        accuracy += trainStyleDiscriminator(input, distribution);
        accuracy += trainLabelDiscriminator(input, distribution);

        return accuracy;
    }

    public float trainDecoder(NNArray[] input, NNArray[] output) {
        encoder.queryTrain(input);
        float accuracy = decoder.train(encoder.getOutputs(), output);
        if (optimizerDecode != null) {
            encoder.setOptimizer(optimizerDecode);
        }
        encoder.train(decoder.getError());
        return accuracy;
    }

    public float trainStyleDiscriminator(NNArray[] input, NNVector[] distribution) {
        //generate input data for styleDiscriminator
        encoder.queryTrain(input);
        NNArray[] fake = encoder.getOutputs();
        NNData data = GANGeneratorData.generateData(distribution, fake);

        //train styleDiscriminator
        float accuracy = styleDiscriminator.train(data.getInput(), data.getOutput());

        //generate data for generator
        NNVector[] label = new NNVector[input.length];
        for (int i = 0; i < label.length; i++) {
            label[i] = new NNVector(new float[]{1});
        }

        //train generator
        styleDiscriminator.setTrainable(false);
        styleDiscriminator.forwardBackpropagation(encoder.getOutputs(), label);
        if (optimizerDistribution != null) {
            encoder.setOptimizer(optimizerDistribution);
        }
        encoder.train(styleDiscriminator.getError());
        styleDiscriminator.setTrainable(true);

        return accuracy;
    }

    public float trainLabelDiscriminator(NNArray[] input, NNVector[] labels) {
        //generate input data for labelDiscriminator
        NNArray[] fake = encoder.getOutputs();
        NNData data = GANGeneratorData.generateData(labels, fake);

        //train labelDiscriminator
        float accuracy = labelDiscriminator.train(data.getInput(), data.getOutput());

        //generate data for generator
        NNVector[] label = new NNVector[input.length];
        for (int i = 0; i < label.length; i++) {
            label[i] = new NNVector(new float[]{1});
        }

        //train generator
        labelDiscriminator.setTrainable(false);
        labelDiscriminator.forwardBackpropagation(encoder.getOutputs(), label);
        if (optimizerLabel != null) {
            encoder.setOptimizer(optimizerLabel);
        }
        encoder.train(labelDiscriminator.getError());
        labelDiscriminator.setTrainable(true);

        return accuracy;
    }
}

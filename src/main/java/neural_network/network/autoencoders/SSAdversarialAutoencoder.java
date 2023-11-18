package neural_network.network.autoencoders;

import data.gan.GANGeneratorData;
import data.network_train.NNData;
import lombok.SneakyThrows;
import neural_network.initialization.Initializer;
import neural_network.layers.LayersBlock;
import neural_network.network.NeuralNetwork;
import neural_network.optimizers.Optimizer;
import nnarrays.NNArray;
import nnarrays.NNArrays;
import nnarrays.NNVector;

public class SSAdversarialAutoencoder {
    private final NeuralNetwork decoder;
    private NeuralNetwork styleDiscriminator;
    private NeuralNetwork labelDiscriminator;
    private final NeuralNetwork encoder;

    private final LayersBlock classificationBlock, styleBlock;

    private Optimizer optimizerDecode, optimizerDistribution, optimizerLabel;

    public SSAdversarialAutoencoder(NeuralNetwork encoder, NeuralNetwork decoder, LayersBlock classificationBlock, LayersBlock styleBlock) {
        this.decoder = decoder;
        this.encoder = encoder;
        this.classificationBlock = classificationBlock;
        this.styleBlock = styleBlock;

        classificationBlock.initialize(encoder.getOutputSize());
        styleBlock.initialize(encoder.getOutputSize());

        optimizerDecode = null;
        optimizerDistribution = null;
    }

    public SSAdversarialAutoencoder setStyleDiscriminator(NeuralNetwork styleDiscriminator) {
        this.styleDiscriminator = styleDiscriminator;
        return this;
    }

    public SSAdversarialAutoencoder setLabelDiscriminator(NeuralNetwork labelDiscriminator) {
        this.labelDiscriminator = labelDiscriminator;
        return this;
    }

    public SSAdversarialAutoencoder setDiscriminators(NeuralNetwork labelDiscriminator, NeuralNetwork styleDiscriminator) {
        this.styleDiscriminator = styleDiscriminator;
        this.labelDiscriminator = labelDiscriminator;
        return this;
    }

    public SSAdversarialAutoencoder setOptimizersEncoder(Optimizer optimizerDecode,
                                                         Optimizer optimizerDistribution,
                                                         Optimizer optimizerLabel) {
        this.optimizerDistribution = optimizerDistribution;
        encoder.initialize(optimizerDistribution);
        styleBlock.initialize(optimizerDistribution);

        this.optimizerDecode = optimizerDecode;
        encoder.initialize(optimizerDecode);
        classificationBlock.initialize(optimizerDecode);
        styleBlock.initialize(optimizerDecode);

        this.optimizerLabel = optimizerLabel;
        encoder.initialize(optimizerLabel);
        classificationBlock.initialize(optimizerLabel);

        return this;
    }

    public NNArray[] query(NNArray[] input) {
        encoder.query(input);
        classificationBlock.generateOutput(encoder.getOutputs());
        styleBlock.generateOutput(encoder.getOutputs());

        NNArray[] inputDecoder = NNArrays.concat(classificationBlock.getOutput(), styleBlock.getOutput());
        return decoder.query(inputDecoder);
    }

    public NNArray[] queryDecoder(NNArray[] label, NNArray[] noise) {
        return decoder.query(NNArrays.concat(label, noise));
    }

    public float train(NNArray[] input, NNVector[] label, NNVector[] distribution) {
        return train(input, input, label, distribution);
    }

    public float train(NNArray[] input) {
        return train(input, null);
    }

    @SneakyThrows
    public float train(NNArray[] input, NNArray[] label) {
        NNVector[] distribution = new NNVector[input.length];
        Initializer initializer = new Initializer.RandomNormal();
        for (int i = 0; i < distribution.length; i++) {
            int[] size = styleBlock.size();
            if (size.length > 1) {
                throw new Exception("Layer must be flat");
            }
            distribution[i] = new NNVector(size[0]);
            initializer.initialize(distribution[i]);
        }

        return train(input, input, label, distribution);
    }

    public float train(NNArray[] input, NNArray[] output, NNArray[] label, NNVector[] distribution) {
        float accuracy = 0;

        accuracy += trainDecoder(input, output);
        accuracy += trainLabelDiscriminator(input, label);
        accuracy += trainStyleDiscriminator(input, distribution);

        return accuracy;
    }

    public float train(NNArray[] input, NNArray[] output, NNVector[] distribution) {
        return train(input, output, null, distribution);
    }

    private float trainDecoder(NNArray[] input, NNArray[] output) {
        //forward propagation
        encoder.queryTrain(input);
        classificationBlock.generateTrainOutput(encoder.getOutputs());
        styleBlock.generateTrainOutput(encoder.getOutputs());

        NNArray[] inputDecoder = NNArrays.concat(classificationBlock.getOutput(), styleBlock.getOutput());

        //trainA decoder
        float accuracy = decoder.train(inputDecoder, output);

        //find error for style and classification blocks
        NNArray[] errorClassificationBlock = NNArrays.subArray(decoder.getError(), classificationBlock.getOutput());
        NNArray[] errorStyleBlock = NNArrays.subArray(decoder.getError(), styleBlock.getOutput(), classificationBlock.getOutput()[0].size());

        classificationBlock.generateError(errorClassificationBlock);
        styleBlock.generateError(errorStyleBlock);
        if (optimizerDecode != null) {
            encoder.setOptimizer(optimizerDecode);
        }

        //generate error for encoder
        NNArray[] errorEncoder = new NNVector[input.length];
        for (int i = 0; i < errorEncoder.length; i++) {
            errorEncoder[i] = new NNVector(encoder.getOutputSize()[0]);
            errorEncoder[i].add(classificationBlock.getError()[i]);
            errorEncoder[i].add(styleBlock.getError()[i]);
        }

        //trainA encoder
        encoder.setOptimizer(optimizerDecode);
        encoder.train(errorEncoder);

        return accuracy;
    }

    private float trainStyleDiscriminator(NNArray[] input, NNVector[] distribution) {
        //generate input data for styleDiscriminator
        encoder.queryTrain(input);
        styleBlock.generateTrainOutput(encoder.getOutputs());
        NNData data = GANGeneratorData.generateData(distribution, styleBlock.getOutput());

        //trainA styleDiscriminator
        float accuracy = styleDiscriminator.train(data.getInput(), data.getOutput());

        //generate data for generator
        NNVector[] label = new NNVector[input.length];
        for (int i = 0; i < label.length; i++) {
            label[i] = new NNVector(new float[]{1}, new short[]{1});
        }

        //trainA generator
        styleDiscriminator.setTrainable(false);
        styleDiscriminator.forwardBackpropagation(styleBlock.getOutput(), label);

        styleBlock.generateError(styleDiscriminator.getError());
        if (optimizerDistribution != null) {
            encoder.setOptimizer(optimizerDistribution);
        }

        encoder.setOptimizer(optimizerDistribution);
        encoder.train(styleBlock.getError());
        styleDiscriminator.setTrainable(true);

        return accuracy;
    }

    private float trainLabelDiscriminator(NNArray[] input, NNArray[] labels) {
        //generate input data for labelDiscriminator
        encoder.queryTrain(input);
        classificationBlock.generateTrainOutput(encoder.getOutputs());
        NNArray[] fake = classificationBlock.getOutput();
        NNData data = GANGeneratorData.generateData(labels, fake);

        //trainA labelDiscriminator
        float accuracy = labelDiscriminator.train(data.getInput(), data.getOutput());

        //generate data for generator
        NNVector[] label = new NNVector[input.length];
        for (int i = 0; i < label.length; i++) {
            label[i] = new NNVector(new float[]{1}, new short[]{1});
        }

        //trainA generator
        labelDiscriminator.setTrainable(false);
        labelDiscriminator.forwardBackpropagation(classificationBlock.getOutput(), label);

        classificationBlock.generateError(labelDiscriminator.getError());

        if (optimizerLabel != null) {
            encoder.setOptimizer(optimizerLabel);
        }

        encoder.setOptimizer(optimizerLabel);
        encoder.train(classificationBlock.getError());
        labelDiscriminator.setTrainable(true);

        return accuracy;
    }
}

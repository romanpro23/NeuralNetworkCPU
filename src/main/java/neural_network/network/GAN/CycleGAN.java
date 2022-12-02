package neural_network.network.GAN;

import neural_network.network.NeuralNetwork;
import nnarrays.NNArray;

public class CycleGAN extends GAN {
    protected final NeuralNetwork generatorInput;
    protected final NeuralNetwork discriminatorInput;

    private boolean identityLoss;

    public CycleGAN(NeuralNetwork generatorA, NeuralNetwork discriminatorA, NeuralNetwork generatorInput, NeuralNetwork discriminatorInput) {
        super(generatorA, discriminatorA);
        this.generatorInput = generatorInput;
        this.discriminatorInput = discriminatorInput;

        this.identityLoss = true;
    }

    public CycleGAN setIdentityLoss(boolean identityLoss) {
        this.identityLoss = identityLoss;

        return this;
    }

    public NNArray[] query(NNArray[] input) {
        return generator.query(input);
    }

    public float train(NNArray[] input, NNArray[] output) {
        return train(input, output, 1, 1);
    }

    public float train(NNArray[] input, NNArray[] output, float lambda) {
        return train(input, output, lambda, lambda * 0.5f);
    }

    public float train(NNArray[] input, NNArray[] output, float lambdaCycle, float lambdaIdentity) {
        float accuracy = 0;
        accuracy += generatorOutputTrain(input, output);
        accuracy += forwardCycleTrain(input, lambdaCycle);

        accuracy += generatorInputTrain(input, output);
        accuracy += backCycleTrain(output, lambdaCycle);

        if (identityLoss) {
            accuracy += identityTrain(input, output, lambdaIdentity);
        }

        generator.update();
        discriminator.update();
        generatorInput.update();
        discriminatorInput.update();

        return accuracy;
    }

    private float identityTrain(NNArray[] input, NNArray[] output, float lambda) {
        float accuracy = 0;

        generator.queryTrain(output);
        generatorInput.queryTrain(input);

        NNArray[] errGenerator = generator.findDerivative(output, lambda);
        NNArray[] errGeneratorInput = generatorInput.findDerivative(input, lambda);
        accuracy += generator.accuracy(output) * lambda;
        accuracy += generatorInput.accuracy(input) * lambda;

        generator.train(errGenerator, false);
        generatorInput.train(errGeneratorInput, false);

        return accuracy;
    }

    private float generatorOutputTrain(NNArray[] input, NNArray[] output) {
        generator.queryTrain(input);

        float accuracy = 0;
        accuracy += discriminator.train(output, getRealLabel(input.length), false, 0.5f);
        accuracy += discriminator.train(generator.getOutputs(), getFakeLabel(input.length), false, 0.5f);

        discriminator.setTrainable(false);
        accuracy += discriminator.train(generator.getOutputs(), getRealLabel(input.length), false);
        discriminator.setTrainable(true);

        return accuracy;
    }

    private float generatorInputTrain(NNArray[] input, NNArray[] output) {
        generatorInput.queryTrain(output);

        float accuracy = 0;
        accuracy += discriminatorInput.train(input, getRealLabel(input.length), false, 0.5f);
        accuracy += discriminatorInput.train(generatorInput.getOutputs(), getFakeLabel(input.length), false, 0.5f);

        discriminatorInput.setTrainable(false);
        accuracy += discriminatorInput.train(generatorInput.getOutputs(), getRealLabel(input.length), false);
        discriminatorInput.setTrainable(true);

        return accuracy;
    }

    private float forwardCycleTrain(NNArray[] input, float lambda) {
        float accuracy = 0;
        //train generators
        generatorInput.queryTrain(generator.getOutputs());
        accuracy += generatorInput.accuracy(input);

        NNArray[] errorGenerator = discriminator.getError();
        NNArray[] errorRecognition = generatorInput.findDerivative(input, lambda);
        generatorInput.train(errorRecognition, false);

        for (int i = 0; i < generatorInput.getError().length; i++) {
            errorGenerator[i].add(generatorInput.getError()[i]);
        }
        generator.train(errorGenerator, false);

        return accuracy;
    }

    private float backCycleTrain(NNArray[] output, float lambda) {
        float accuracy = 0;
        //train generators
        generator.queryTrain(generatorInput.getOutputs());
        accuracy += generator.accuracy(output);

        NNArray[] errorGenerator = discriminatorInput.getError();
        NNArray[] errorRecognition = generator.findDerivative(output, lambda);

        generator.train(errorRecognition, false);

        for (int i = 0; i < generator.getError().length; i++) {
            errorGenerator[i].add(generator.getError()[i]);
        }
        generatorInput.train(errorGenerator, false);

        return accuracy;
    }
}

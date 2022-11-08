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
        System.out.println("Train forward generator");
        accuracy += generatorOutputTrain(input, output);
        System.out.println("Train forward cycle");
        accuracy += forwardCycleTrain(input, lambdaCycle);

        System.out.println("Train back generator");
        accuracy += generatorInputTrain(input, output);
        System.out.println("Train back cycle");
        accuracy += backCycleTrain(output, lambdaCycle);

        if (identityLoss) {
            System.out.println("Train identity");
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

        NNArray[] errGenerator = generator.findDerivative(output);
        NNArray[] errGeneratorInput = generatorInput.findDerivative(input);
        accuracy += generator.accuracy(output);
        accuracy += generatorInput.accuracy(input);

        for (int i = 0; i < input.length; i++) {
            errGenerator[i].mul(lambda);
            errGeneratorInput[i].mul(lambda);
        }

        generator.train(errGenerator, false);
        generatorInput.train(errGeneratorInput, false);

        return accuracy;
    }

    private float generatorOutputTrain(NNArray[] input, NNArray[] output) {
        generator.queryTrain(input);

        float accuracy = 0;
        accuracy += discriminator.train(output, getRealLabel(input.length), false);
        accuracy += discriminator.train(generator.getOutputs(), getFakeLabel(input.length), false);

        discriminator.setTrainable(false);
        accuracy += discriminator.train(generator.getOutputs(), getRealLabel(input.length), false);
        discriminator.setTrainable(true);

        return accuracy;
    }

    private float generatorInputTrain(NNArray[] input, NNArray[] output) {
        generatorInput.queryTrain(output);

        float accuracy = 0;
        accuracy += discriminatorInput.train(input, getRealLabel(input.length), false);
        accuracy += discriminatorInput.train(generatorInput.getOutputs(), getFakeLabel(input.length), false);

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
        NNArray[] errorRecognition = generatorInput.findDerivative(input);
        for (NNArray array : errorRecognition) {
            if (lambda != 1) {
                array.mul(lambda);
            }
        }
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
        NNArray[] errorRecognition = generator.findDerivative(output);
        for (NNArray array : errorRecognition) {
            if (lambda != 1) {
                array.mul(lambda);
            }
        }
        generator.train(errorRecognition, false);

        for (int i = 0; i < generator.getError().length; i++) {
            errorGenerator[i].add(generator.getError()[i]);
        }
        generatorInput.train(errorGenerator, false);

        return accuracy;
    }
}

package neural_network.network.GAN;

import neural_network.network.NeuralNetwork;
import nnarrays.NNArray;
import nnarrays.NNArrays;
import nnarrays.NNTensor;

public class SegmentDiscriminator extends NeuralNetwork {
    private final NeuralNetwork discriminator;
    private NNTensor[] error;

    private int stepX, stepY;

    public SegmentDiscriminator(NeuralNetwork discriminator) {
        this(discriminator, discriminator.getInputSize()[0], discriminator.getInputSize()[1]);
    }

    public SegmentDiscriminator(NeuralNetwork discriminator, int stepY, int stepX) {
        this.discriminator = discriminator;
        this.stepX = stepX;
        this.stepY = stepY;
    }

    @Override
    public float train(NNArray[] input, NNArray[] idealOutput, boolean update){
        float accuracy = 0;
        int[] sizeImage = input[0].getSize();

        NNTensor[] inputDiscriminator = inputDiscriminator(input.length);
        error = errorDiscriminator(NNArrays.isTensor(input));

        for (int i = 0; i < sizeImage[0]; i+= stepY) {
            for (int j = 0; j < sizeImage[1]; j+= stepX) {
                NNArrays.subTensor(input, inputDiscriminator, i, j);
                accuracy += discriminator.train(inputDiscriminator, idealOutput, false);
                NNArrays.addSubTensor(input, inputDiscriminator, i, j);
            }
        }

        return accuracy;
    }

    @Override
    public int[] getOutputSize(){
        return discriminator.getOutputSize();
    }

    @Override
    public int[] getInputSize(){
        return discriminator.getInputSize();
    }

    @Override
    public NeuralNetwork setTrainable(boolean trainable){
        discriminator.setTrainable(trainable);
        return this;
    }

    @Override
    public void update(){
        discriminator.update();
    }

    @Override
    public NNArray[] getError(){
        return error;
    }

    private NNTensor[] inputDiscriminator(int sizeBatch){
        NNTensor[] input = new NNTensor[sizeBatch];
        for (int i = 0; i < sizeBatch; i++) {
            input[i] = new NNTensor(discriminator.getInputSize());
        }

        return input;
    }

    private NNTensor[] errorDiscriminator(NNTensor[] input){
        NNTensor[] error = new NNTensor[input.length];
        for (int i = 0; i < input.length; i++) {
            error[i] = new NNTensor(input[i].getSize());
        }

        return error;
    }
}

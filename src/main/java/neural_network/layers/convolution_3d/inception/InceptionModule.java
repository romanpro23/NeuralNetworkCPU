package neural_network.layers.convolution_3d.inception;

import lombok.Getter;
import neural_network.layers.NeuralLayer;
import neural_network.layers.convolution_3d.ConvolutionNeuralLayer;
import neural_network.optimizers.Optimizer;
import nnarrays.NNArray;
import nnarrays.NNArrays;
import nnarrays.NNTensor;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Scanner;

public class InceptionModule extends ConvolutionNeuralLayer {
    @Getter
    private ArrayList<InceptionBlock> module;

    public InceptionModule() {
        module = new ArrayList<>();
    }

    @Override
    public void initialize(Optimizer optimizer) {
        for (int i = 0; i < module.size(); i++) {
            module.get(i).initialize(optimizer);
        }
    }

    @Override
    public void initialize(int[] size) {
        if (size.length != 3) {
            throw new ExceptionInInitializerError("Error size pre layer!");
        }

        height = size[0];
        width = size[1];
        depth = size[2];

        outDepth = 0;
        for (InceptionBlock inceptionBlock : module) {
            inceptionBlock.initialize(size);
            outDepth += inceptionBlock.size()[2];
            outHeight = inceptionBlock.size()[0];
            outWidth = inceptionBlock.size()[1];
        }
    }

    @Override
    public void generateOutput(NNArray[] inputs) {
        this.input = NNArrays.isTensor(inputs);
        for (InceptionBlock inceptionBlock : module) {
            inceptionBlock.generateOutput(inputs);
        }
        generateOutputModule();
    }

    private void generateOutputModule() {
        if (module.size() == 1) {
            this.output = NNArrays.isTensor(module.get(0).getOutput());
        } else {
            this.output = NNArrays.concatTensor(module.get(0).getOutput(), module.get(1).getOutput());
            for (int i = 2; i < module.size(); i++) {
                this.output = NNArrays.concatTensor(this.output, module.get(i).getOutput());
            }
        }
    }

    @Override
    public void generateTrainOutput(NNArray[] input) {
        this.input = NNArrays.isTensor(input);
        for (InceptionBlock inceptionBlock : module) {
            inceptionBlock.generateTrainOutput(input);
        }
        generateOutputModule();
    }

    @Override
    public void generateError(NNArray[] errors) {
        this.errorNL = getErrorNextLayer(errors);
        this.error = new NNTensor[errorNL.length];
        int index = 0;
        for (InceptionBlock inceptionBlock : module) {
            inceptionBlock.generateError(NNArrays.subTensor(errorNL, inceptionBlock.getOutput(), index));
            index += inceptionBlock.getOutput()[0].size();
        }

        for (int i = 0; i < errorNL.length; i++) {
            error[i] = new NNTensor(height, width, depth);
            for (InceptionBlock inceptionBlock : module) {
                error[i].add(inceptionBlock.getError()[i]);
            }
        }
    }

    @Override
    public void write(FileWriter writer) throws IOException {
        writer.write("Inception module\n");
        for (InceptionBlock layer : module) {
            layer.write(writer);
        }
        writer.write("End\n");
        writer.flush();
    }

    @Override
    public int info() {
        int countParam = 0;
        System.out.println("            |         Inception module      |             ");
        System.out.println("____________|_______________________________|_____________");
        for (InceptionBlock inceptionBlock : module) {
            countParam += inceptionBlock.info();
        }
        System.out.println("            |  " + height + ",\t" + width + ",\t" + depth + "\t|  "
                + outHeight + ",\t" + outWidth + ",\t" + outDepth + "\t|\t" + countParam);
        return countParam;
    }

    public InceptionModule addInceptionBlock(InceptionBlock inceptionBlock) {
        module.add(inceptionBlock);
        return this;
    }

    public InceptionModule setTrainable(boolean trainable) {
        for (InceptionBlock inceptionBlock : module) {
            inceptionBlock.setTrainable(trainable);
        }
        return this;
    }

    public static InceptionModule read(Scanner scanner) {
        InceptionModule inceptionModule = new InceptionModule();

        String layer = scanner.nextLine();
        while (!layer.equals("End")) {
            inceptionModule.module.add(InceptionBlock.read(scanner));
            layer = scanner.nextLine();
        }

        return inceptionModule;
    }
}

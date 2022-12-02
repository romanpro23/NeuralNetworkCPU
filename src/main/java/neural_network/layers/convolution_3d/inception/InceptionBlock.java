package neural_network.layers.convolution_3d.inception;

import lombok.Getter;
import neural_network.layers.convolution_3d.ConvolutionNeuralLayer;
import neural_network.optimizers.Optimizer;
import nnarrays.NNArray;
import nnarrays.NNArrays;
import nnarrays.NNTensor;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Scanner;

public class InceptionBlock extends ConvolutionNeuralLayer {
    @Getter
    private ArrayList<InceptionUnit> module;

    public InceptionBlock() {
        module = new ArrayList<>();
    }

    @Override
    public void initialize(Optimizer optimizer) {
        for (InceptionUnit inceptionUnit : module) {
            inceptionUnit.initialize(optimizer);
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
        for (InceptionUnit inceptionUnit : module) {
            inceptionUnit.initialize(size);
            outDepth += inceptionUnit.size()[2];
            outHeight = inceptionUnit.size()[0];
            outWidth = inceptionUnit.size()[1];
        }
    }

    @Override
    public void generateOutput(NNArray[] inputs) {
        this.input = NNArrays.isTensor(inputs);
        for (InceptionUnit inceptionUnit : module) {
            inceptionUnit.generateOutput(inputs);
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
        for (InceptionUnit inceptionUnit : module) {
            inceptionUnit.generateTrainOutput(input);
        }
        generateOutputModule();
    }

    @Override
    public void generateError(NNArray[] errors) {
        this.errorNL = getErrorNextLayer(errors);
        this.error = new NNTensor[errorNL.length];
        int index = 0;
        for (InceptionUnit inceptionUnit : module) {
            inceptionUnit.generateError(NNArrays.subTensor(errorNL, inceptionUnit.getOutput(), index));
            index += inceptionUnit.getOutput()[0].size();
        }

        for (int i = 0; i < errorNL.length; i++) {
            error[i] = new NNTensor(height, width, depth);
            for (InceptionUnit inceptionUnit : module) {
                error[i].add(inceptionUnit.getError()[i]);
            }
        }
    }

    @Override
    public void save(FileWriter writer) throws IOException {
        writer.write("Inception block\n");
        for (InceptionUnit layer : module) {
            layer.save(writer);
        }
        writer.write("End\n");
        writer.flush();
    }

    @Override
    public int info() {
        int countParam = 0;
        System.out.println("            |         Inception block       |             ");
        System.out.println("____________|_______________________________|_____________");
        for (InceptionUnit inceptionUnit : module) {
            countParam += inceptionUnit.info();
        }
        System.out.println("            |  " + height + ",\t" + width + ",\t" + depth + "\t|  "
                + outHeight + ",\t" + outWidth + ",\t" + outDepth + "\t|\t" + countParam);
        return countParam;
    }

    public InceptionBlock addInceptionUnit(InceptionUnit inceptionUnit) {
        module.add(inceptionUnit);
        return this;
    }

    public InceptionBlock setTrainable(boolean trainable) {
        for (InceptionUnit inceptionUnit : module) {
            inceptionUnit.setTrainable(trainable);
        }
        return this;
    }

    @Override
    public void trainable(boolean trainable) {
        for (InceptionUnit inceptionUnit : module) {
            inceptionUnit.trainable(trainable);
        }
    }

    public static InceptionBlock read(Scanner scanner) {
        InceptionBlock inceptionBlock = new InceptionBlock();

        String layer = scanner.nextLine();
        while (!layer.equals("End")) {
            inceptionBlock.module.add(InceptionUnit.read(scanner));
            layer = scanner.nextLine();
        }

        return inceptionBlock;
    }
}

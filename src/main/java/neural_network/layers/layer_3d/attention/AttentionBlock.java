package neural_network.layers.layer_3d.attention;

import neural_network.layers.LayersBlock;
import neural_network.layers.NeuralLayer;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Scanner;

public class AttentionBlock extends LayersBlock {
    private int depth, height, width;
    private int outHeight, outWidth, outDepth;

    public AttentionBlock() {
        layers = new ArrayList<>();
    }

    @Override
    public void initialize(int[] size) {
        if (size.length != 3) {
            throw new ExceptionInInitializerError("Error size pre layer!");
        }

        height = size[0];
        width = size[1];
        depth = size[2];
        outWidth = width;
        outHeight = height;
        outDepth = depth;

        super.initialize(size);
    }

    public AttentionBlock addChannelAttentionModule(int hidden){
        addLayer(new ChannelAttentionModule(hidden));

        return this;
    }

    public AttentionBlock addSpatialAttentionModule(int sizeKernel){
        addLayer(new SpatialAttentionModule(sizeKernel));

        return this;
    }

    public AttentionBlock addLayer(NeuralLayer layer){
        super.addLayer(layer);
        return this;
    }

    @Override
    public void save(FileWriter writer) throws IOException {
        writer.write("Attention block\n");
        for (NeuralLayer neuralLayer : layers) {
            neuralLayer.save(writer);
        }
        writer.write("End\n");
        writer.flush();
    }

    public static AttentionBlock read(Scanner scanner) {
        AttentionBlock attentionBlock = new AttentionBlock();
        NeuralLayer.read(scanner, attentionBlock.layers);

        return attentionBlock;
    }

    @Override
    public int info() {
        int countParam = 0;
        System.out.println("            | Convolutional Attention Block |             ");
        System.out.println("____________|_______________________________|_____________");
        for (NeuralLayer neuralLayer : layers) {
            countParam += neuralLayer.info();
            System.out.println("____________|_______________|_______________|_____________");
        }
        System.out.println("            |  " + height + ",\t" + width + ",\t" + depth + "\t|  "
                + outHeight + ",\t" + outWidth + ",\t" + outDepth + "\t|\t" + countParam);
        System.out.println("____________|_______________|_______________|_____________");
        return countParam;
    }

    public AttentionBlock setTrainable(boolean trainable) {
        super.setTrainable(trainable);

        return this;
    }
}

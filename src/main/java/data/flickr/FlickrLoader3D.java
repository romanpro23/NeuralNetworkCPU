package data.flickr;

import data.loaders.*;
import lombok.SneakyThrows;
import nnarrays.NNTensor;
import nnarrays.NNVector;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Scanner;

public class FlickrLoader3D extends DataLoader3D {
    private final int sizeImage;
    private final int sizeVocabulary;
    private int sizeTestBatch, imageIndex;

    private Scanner scanner;
    private LinkedHashMap<String, Integer> vocabulary;

    private TransformData transformData;

    public FlickrLoader3D(int sizeImage, int sizeVocabulary, int sizeTestBatch) {
        this(sizeImage, sizeVocabulary, sizeTestBatch, new TransformData.Sigmoid());
    }

    public FlickrLoader3D(int sizeImage, int sizeVocabulary, int sizeTestBatch, TransformData transformData) {
        this.sizeImage = sizeImage;
        this.sizeVocabulary = sizeVocabulary;
        this.sizeTestBatch = sizeTestBatch;

        this.transformData = transformData;

        test = new ArrayList<>(sizeTestBatch);
        train = new ArrayList<>(64000);

        loadData();
    }

    public FlickrLoader3D setTransformData(TransformData transformData){
        this.transformData = transformData;

        return this;
    }

    @SneakyThrows
    private void loadData(){
        Scanner scannerVoc = new Scanner(new File("D:/datasets/flickr/vocabulary.txt"));
        scanner = new Scanner(new File("D:/datasets/flickr/result.txt"));

        vocabulary = new LinkedHashMap<>();
        vocabulary.put("<START>", 0);
        vocabulary.put("<END>", 1);
        vocabulary.put("<UNK>", 2);

        for (int i = 3; i < sizeVocabulary; i++) {
            vocabulary.put(scannerVoc.nextLine(), i);
        }

        imageIndex = 1;

        for (int i = 0; i < sizeTestBatch; i++) {
            test.add(loadImage());
        }

        reloadTrainData();

        Collections.shuffle(test);
        Collections.shuffle(train);
    }

    @SneakyThrows
    @Override
    protected void reloadTrainData(){
        train.removeAll(train);
        for (int i = 0; i < 6400; i++) {
            if(imageIndex > 31783){
                imageIndex = sizeTestBatch + 1;
                scanner = new Scanner(new File("D:/datasets/flickr/result.txt"));
                for (int j = 0; j < sizeTestBatch; j++) {
                    scanner.nextLine();
                }
            }
            train.add(loadImage());
        }
    }

    @SneakyThrows
    private  FlickImageData3D loadImage(){
        NNVector[] outputs = new NNVector[5];

        for (int s = 0; s < outputs.length; s++) {
            String[] data = scanner.nextLine().split(" ");

            NNVector output = new NNVector(data.length);
            int index;
            for (int i = 0; i < data.length; i++) {
                if(vocabulary.get(data[i]) != null){
                    index = vocabulary.get(data[i]);
                } else {
                    index = 2;
                }
                output.set(i, index);
            }

            outputs[s] = output;
        }

        NNTensor input = new NNTensor(sizeImage, sizeImage, 3);
        BufferedImage image = ImageIO.read(new File("D:/datasets/flickr/train_" + sizeImage + "/" + imageIndex + ".png"));

        for (int i = 0; i < sizeImage; i++) {
            for (int j = 0; j < sizeImage; j++) {
                Color color = new Color(image.getRGB(i, j));
                input.set(i, j, 0, transformData.transformR(color.getRed()));
                input.set(i, j, 1, transformData.transformG(color.getGreen()));
                input.set(i, j, 2, transformData.transformB(color.getBlue()));
            }
        }
        imageIndex++;
        return new FlickImageData3D(input, outputs);
    }

    class FlickImageData3D extends ImageData3D {
        public NNVector[] output;

        public FlickImageData3D(NNTensor inputs, NNVector[] output) {
            super(inputs, output[0]);
            this.output = output;
        }

        @Override
        public NNVector getOutputs(){
            return output[(int) (Math.random() * 5)];
        }
    }
}

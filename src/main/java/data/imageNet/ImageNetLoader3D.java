package data.imageNet;

import data.loaders.DataLoader3D;
import data.loaders.ImageData3D;
import data.loaders.TransformData;
import lombok.SneakyThrows;
import nnarrays.NNTensor;
import nnarrays.NNVector;
import org.apache.commons.lang3.StringUtils;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Scanner;

public class ImageNetLoader3D extends DataLoader3D {
    private final int sizeImage;
    private int sizeTestBatch, imageIndexTest, imageIndexTrain;

    private Scanner scanner;

    private TransformData transformData;

    public ImageNetLoader3D(int sizeImage) {
        this(sizeImage, new TransformData.Sigmoid());
    }

    public ImageNetLoader3D(int sizeImage, TransformData transformData) {
        this.sizeImage = sizeImage;

        this.transformData = transformData;

        test = new ArrayList<>(3200);
        train = new ArrayList<>(6400);

        loadData();
    }

    public ImageNetLoader3D setTransformData(TransformData transformData){
        this.transformData = transformData;

        return this;
    }

    @SneakyThrows
    private void loadData(){
        imageIndexTest = 1;
        imageIndexTrain = 35000;

        reloadTrainData();
        reloadTestData();

        Collections.shuffle(test);
        Collections.shuffle(train);
    }

    @SneakyThrows
    @Override
    protected void reloadTestData(){
        test.removeAll(test);
        for (int i = 0; i < 3200; i++) {
            if(imageIndexTest > 49999){
                imageIndexTest = 1;
            }
            test.add(loadTestImage());
        }
    }

    @SneakyThrows
    @Override
    protected void reloadTrainData(){
        train.removeAll(train);
        for (int i = 0; i < 6400; i++) {
            if(imageIndexTrain > 1281149){
                imageIndexTrain = 1;
            }
            train.add(loadTrainImage());
        }
    }

    @SneakyThrows
    private  ImageData3D loadTrainImage(){
        NNTensor input = new NNTensor(sizeImage, sizeImage, 3);
        BufferedImage image = ImageIO.read(new File("D:/datasets/ImageNet/train_" + sizeImage + "/" + StringUtils.leftPad(String.valueOf(imageIndexTrain), 7, "0") + ".png"));

        for (int i = 0; i < sizeImage; i++) {
            for (int j = 0; j < sizeImage; j++) {
                Color color = new Color(image.getRGB(i, j));
                input.set(i, j, 0, transformData.transform(color.getRed()));
                input.set(i, j, 1, transformData.transform(color.getGreen()));
                input.set(i, j, 2, transformData.transform(color.getBlue()));
            }
        }
        imageIndexTrain++;
        return new ImageData3D(input, null);
    }

    @SneakyThrows
    private  ImageData3D loadTestImage(){
        NNTensor input = new NNTensor(sizeImage, sizeImage, 3);
        BufferedImage image = ImageIO.read(new File("D:/datasets/ImageNet/valid_" + sizeImage + "/" + StringUtils.leftPad(String.valueOf(imageIndexTest), 5, "0")  + ".png"));

        for (int i = 0; i < sizeImage; i++) {
            for (int j = 0; j < sizeImage; j++) {
                Color color = new Color(image.getRGB(i, j));
                input.set(i, j, 0, transformData.transform(color.getRed()));
                input.set(i, j, 1, transformData.transform(color.getGreen()));
                input.set(i, j, 2, transformData.transform(color.getBlue()));
            }
        }
        imageIndexTest++;
        return new ImageData3D(input, null);
    }
}

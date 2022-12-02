package data.imageNet;

import data.loaders.*;
import lombok.SneakyThrows;
import nnarrays.NNTensor;
import org.apache.commons.lang3.StringUtils;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Scanner;

public class ScaleImageNetLoader3D extends ImageTranslationDataLoader3D {
    private final int sizeImageHR;
    private final int sizeImageLR;
    private int sizeTestBatch, imageIndexTest, imageIndexTrain;

    private Scanner scanner;

    private TransformData transformData;

    public ScaleImageNetLoader3D(int sizeImageLR, int sizeImageHR) {
        this(sizeImageLR, sizeImageHR, new TransformData.Sigmoid());
    }

    public ScaleImageNetLoader3D(int sizeImageLR, int sizeImageHR, TransformData transformData) {
        this.sizeImageHR = sizeImageHR;
        this.sizeImageLR = sizeImageLR;

        this.transformData = transformData;

        test = new ArrayList<>(3200);
        train = new ArrayList<>(6400);

        loadData();
    }

    public ScaleImageNetLoader3D setTransformData(TransformData transformData){
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
            test.add(new Img2ImgData3D(loadTestImage(sizeImageLR), loadTestImage(sizeImageHR)));
            imageIndexTest++;
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
            train.add(new Img2ImgData3D(loadTrainImage(sizeImageLR), loadTrainImage(sizeImageHR)));
            imageIndexTrain++;
        }
    }

    @SneakyThrows
    private  NNTensor loadTrainImage(int sizeImage){
        NNTensor input = new NNTensor(sizeImage, sizeImage, 3);
        BufferedImage image = ImageIO.read(new File("D:/datasets/ImageNet/train_" + sizeImage + "/" + StringUtils.leftPad(String.valueOf(imageIndexTrain), 7, "0") + ".png"));

        for (int i = 0; i < sizeImage; i++) {
            for (int j = 0; j < sizeImage; j++) {
                Color color = new Color(image.getRGB(i, j));
                input.set(i, j, 0, transformData.transformR(color.getRed()));
                input.set(i, j, 1, transformData.transformG(color.getGreen()));
                input.set(i, j, 2, transformData.transformB(color.getBlue()));
            }
        }
        return input;
    }

    @SneakyThrows
    private  NNTensor loadTestImage(int sizeImage){
        NNTensor input = new NNTensor(sizeImage, sizeImage, 3);
        BufferedImage image = ImageIO.read(new File("D:/datasets/ImageNet/valid_" + sizeImage + "/" + StringUtils.leftPad(String.valueOf(imageIndexTest), 5, "0")  + ".png"));

        for (int i = 0; i < sizeImage; i++) {
            for (int j = 0; j < sizeImage; j++) {
                Color color = new Color(image.getRGB(i, j));
                input.set(i, j, 0, transformData.transformR(color.getRed()));
                input.set(i, j, 1, transformData.transformG(color.getGreen()));
                input.set(i, j, 2, transformData.transformB(color.getBlue()));
            }
        }
        return input;
    }
}

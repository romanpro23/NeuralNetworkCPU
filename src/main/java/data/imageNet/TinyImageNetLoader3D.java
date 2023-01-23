package data.imageNet;

import data.loaders.DataLoader3D;
import data.loaders.ImageData3D;
import data.loaders.TransformData;
import lombok.SneakyThrows;
import nnarrays.NNTensor;
import nnarrays.NNVector;
import org.apache.commons.lang3.StringUtils;
import org.imgscalr.Scalr;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Scanner;

public class TinyImageNetLoader3D extends DataLoader3D {
    private int imageIndexTest, imageIndexTrain;
    private static final int sizeImage = 64;

    private TransformData transformData;

    public TinyImageNetLoader3D() {
        this(new TransformData.Sigmoid());
    }

    public TinyImageNetLoader3D(TransformData transformData) {
        this.transformData = transformData;

        test = new ArrayList<>(1600);
        train = new ArrayList<>(6400);

        loadData();
    }

    public TinyImageNetLoader3D setTransformData(TransformData transformData){
        this.transformData = transformData;

        return this;
    }

    @SneakyThrows
    private void loadData(){
        imageIndexTest = (int) (Math.random() * 10000);
        imageIndexTrain = (int) (Math.random() * 100000);

        reloadTrainData();
        reloadTestData();

        Collections.shuffle(test);
        Collections.shuffle(train);
    }

    @SneakyThrows
    @Override
    protected void reloadTestData(){
        test.removeAll(test);
        for (int i = 0; i < 1600; i++) {
            if(imageIndexTest > 9999){
                imageIndexTest = 0;
            }
            test.add(loadTestImage());
        }
    }

    @SneakyThrows
    @Override
    protected void reloadTrainData(){
        train.removeAll(train);
        for (int i = 0; i < 6400; i++) {
            if(imageIndexTrain > 99999){
                imageIndexTrain = 0;
            }
            train.add(loadTrainImage());
        }
    }

    @SneakyThrows
    private  ImageData3D loadTrainImage(){
        NNTensor input = new NNTensor(sizeImage, sizeImage, 3);
        BufferedImage image = ImageIO.read(new File("D:/datasets/TinyImagenet/train/" + imageIndexTrain + ".png"));

        if(useCrop){
            BufferedImage scalImage = Scalr.resize(image, Scalr.Method.ULTRA_QUALITY, 72, 72);
            image = scalImage.getSubimage((int) (Math.random() * 8), (int) (Math.random() * 8), 64, 64 );
        }

        for (int i = 0; i < sizeImage; i++) {
            for (int j = 0; j < sizeImage; j++) {
                Color color = new Color(image.getRGB(i, j));
                input.set(i, j, 0, transformData.transformR(color.getRed()));
                input.set(i, j, 1, transformData.transformG(color.getGreen()));
                input.set(i, j, 2, transformData.transformB(color.getBlue()));
            }
        }
        NNVector output = new NNVector(200);
        output.set(imageIndexTrain % 200, 1);
        imageIndexTrain++;
        return new ImageData3D(input, output);
    }

    public TinyImageNetLoader3D useCrop(){
        this.useCrop = true;

        return this;
    }

    public TinyImageNetLoader3D useReverse(){
        this.useReverse = true;

        return this;
    }

    @SneakyThrows
    private  ImageData3D loadTestImage(){
        NNTensor input = new NNTensor(sizeImage, sizeImage, 3);
        BufferedImage image = ImageIO.read(new File("D:/datasets/TinyImagenet/val/" + imageIndexTest + ".png"));

        for (int i = 0; i < sizeImage; i++) {
            for (int j = 0; j < sizeImage; j++) {
                Color color = new Color(image.getRGB(i, j));
                input.set(i, j, 0, transformData.transformR(color.getRed()));
                input.set(i, j, 1, transformData.transformG(color.getGreen()));
                input.set(i, j, 2, transformData.transformB(color.getBlue()));
            }
        }
        NNVector output = new NNVector(200);
        output.set(imageIndexTest % 200, 1);
        imageIndexTest++;
        return new ImageData3D(input, output);
    }
}

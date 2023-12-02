package data.image2image;

import data.loaders.DataLoader3D;
import data.loaders.ImageData3D;
import data.loaders.Img2ImgDataLoader3D;
import data.loaders.TransformData;
import lombok.SneakyThrows;
import nnarrays.NNTensor;
import org.apache.commons.lang3.StringUtils;
import org.imgscalr.Scalr;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Scanner;

public class AppleToOrangeLoader3D extends Img2ImgDataLoader3D {
    private final int sizeImage;

    private TransformData transformData;

    public AppleToOrangeLoader3D(int sizeImage) {
        this(sizeImage, new TransformData.Sigmoid());
    }

    public AppleToOrangeLoader3D() {
        this(64, new TransformData.Sigmoid());
    }

    public AppleToOrangeLoader3D(TransformData transformData) {
        this(64, transformData);
    }

    public AppleToOrangeLoader3D(int sizeImage, TransformData transformData) {
        this.sizeImage = sizeImage;

        this.transformData = transformData;

        testA = new ArrayList<>(300);
        trainA = new ArrayList<>(1100);
        testB = new ArrayList<>(300);
        trainB = new ArrayList<>(1100);

        loadData();
    }

    public AppleToOrangeLoader3D setTransformData(TransformData transformData) {
        this.transformData = transformData;

        return this;
    }

    @SneakyThrows
    private void loadData() {
        File dir = new File("C:/datasets/apple_orange/trainA_" + sizeImage + "/");
        for (File file : dir.listFiles()) {
            trainA.add(loadImage("trainA", file.getName()));
        }
        dir = new File("C:/datasets/apple_orange/trainB_" + sizeImage + "/");
        for (File file : dir.listFiles()) {
            trainB.add(loadImage("trainB", file.getName()));
        }

        dir = new File("C:/datasets/apple_orange/testA_" + sizeImage + "/");
        for (File file : dir.listFiles()) {
            testA.add(loadImage("testA", file.getName()));
        }
        dir = new File("C:/datasets/apple_orange/testB_" + sizeImage + "/");
        for (File file : dir.listFiles()) {
            testB.add(loadImage("testB", file.getName()));
        }

        Collections.shuffle(testA);
        Collections.shuffle(trainA);

        Collections.shuffle(testB);
        Collections.shuffle(trainB);
    }

    @SneakyThrows
    private NNTensor loadImage(String str, String fileName) {
        NNTensor input = new NNTensor(sizeImage, sizeImage, 3);
        BufferedImage image = ImageIO.read(new File("C:/datasets/apple_orange/" + str + "_" + sizeImage + "/" + fileName));
        BufferedImage scaledImg = Scalr.resize(image, Scalr.Method.ULTRA_QUALITY, (int) (sizeImage * 1.1), (int) (sizeImage * 1.1));

        int i0 = (int) (Math.random() * (scaledImg.getHeight() - sizeImage));
        int j0 = (int) (Math.random() * (scaledImg.getWidth() - sizeImage));
        for (int i = i0; i < sizeImage + i0; i++) {
            for (int j = j0; j < sizeImage + j0; j++) {
                Color color = new Color(scaledImg.getRGB(i, j));
                int i1 = i - i0;
                int j1 = j - j0;
                input.set(i1, j1, 0, transformData.transformR(color.getRed()));
                input.set(i1, j1, 1, transformData.transformG(color.getGreen()));
                input.set(i1, j1, 2, transformData.transformB(color.getBlue()));
            }
        }

        return input;
    }
}

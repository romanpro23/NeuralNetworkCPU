package data.image2image;

import data.loaders.Img2ImgDataLoader3D;
import data.loaders.TransformData;
import lombok.SneakyThrows;
import nnarrays.NNTensor;
import org.imgscalr.Scalr;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;

public class FishOldToNewLoader3D extends Img2ImgDataLoader3D {
    private final int sizeImage;

    private TransformData transformData;

    public FishOldToNewLoader3D(int sizeImage) {
        this(sizeImage, new TransformData.Sigmoid());
    }

    public FishOldToNewLoader3D() {
        this(64, new TransformData.Sigmoid());
    }

    public FishOldToNewLoader3D(TransformData transformData) {
        this(64, transformData);
    }

    public FishOldToNewLoader3D(int sizeImage, TransformData transformData) {
        this.sizeImage = sizeImage;

        this.transformData = transformData;

        testA = new ArrayList<>(0);
        trainA = new ArrayList<>(700);
        testB = new ArrayList<>(0);
        trainB = new ArrayList<>(700);

        loadData();
    }

    public FishOldToNewLoader3D setTransformData(TransformData transformData) {
        this.transformData = transformData;

        return this;
    }

    @SneakyThrows
    private void loadData() {
        File dir = new File("D:/FishOld128/");
        for (File file : dir.listFiles()) {
            NNTensor input = new NNTensor(sizeImage, sizeImage, 3);
            BufferedImage image = ImageIO.read(new File("D:/FishOld128/" + file.getName()));

            for (int i = 0; i < sizeImage; i++) {
                for (int j = 0; j < sizeImage; j++) {
                    Color color = new Color(image.getRGB(i, j), true);
                    if (color.getAlpha() > 0) {
                        input.set(i, j, 0, transformData.transform(color.getRed()));
                        input.set(i, j, 1, transformData.transform(color.getGreen()));
                        input.set(i, j, 2, transformData.transform(color.getBlue()));
                    } else {
                        input.set(i, j, 0, 1);
                        input.set(i, j, 1, 1);
                        input.set(i, j, 2, 1);
                    }
                }
            }
            trainA.add(input);
        }
        dir = new File("D:/FishNew128/");

        for (File file : dir.listFiles()) {
            NNTensor input = new NNTensor(sizeImage, sizeImage, 3);
            BufferedImage image = ImageIO.read(new File("D:/FishNew128/" + file.getName()));

            for (int i = 0; i < sizeImage; i++) {
                for (int j = 0; j < sizeImage; j++) {
                    Color color = new Color(image.getRGB(i, j), true);

                    if (color.getAlpha() > 0) {
                        input.set(i, j, 0, transformData.transform(color.getRed()));
                        input.set(i, j, 1, transformData.transform(color.getGreen()));
                        input.set(i, j, 2, transformData.transform(color.getBlue()));
                    } else {
                        input.set(i, j, 0, 1);
                        input.set(i, j, 1, 1);
                        input.set(i, j, 2, 1);
                    }
                }
            }
            trainB.add(input);
        }

        Collections.shuffle(trainA);
        Collections.shuffle(trainB);
    }
}

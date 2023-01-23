package test.utilitas;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

public class Test {
    public static void main(String[] args) throws IOException {
        File train = new File("D:\\datasets\\ImagenetTrain\\");
        String pathTrain = "D:\\datasets\\ImageNet100\\train\\";
        String pathValid = "D:\\datasets\\ImageNet100\\valid\\";
        for (int i = 0; i < train.listFiles().length; i++) {
            String name = train.listFiles()[i].getName();
            System.out.println(name);
            File fT = new File(pathTrain + name);
            File fV = new File(pathValid + name);
            if (fT.listFiles() == null) {
                Files.createDirectories(Path.of(fT.getPath()));
                Files.createDirectories(Path.of(fV.getPath()));
                File images = train.listFiles()[i];
                for (int j = 0; j < images.listFiles().length; j++) {
                    try {
                        if (j < 50) {
                            String val = fV.getPath() + "\\" + images.listFiles()[j].getName() + ".png";
                            Files.copy(Path.of(images.listFiles()[j].getPath()), Path.of(val));
                        } else {
                            String tr = fT.getPath() + "\\" + images.listFiles()[j].getName() + ".png";
                            Files.copy(Path.of(images.listFiles()[j].getPath()), Path.of(tr));
                        }
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
            }
        }
    }
}

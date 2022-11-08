package test.image;

import org.imgscalr.Scalr;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class ResizeImage {
    public static void main(String[] args) throws IOException {
        File dir = new File("D:/FishNew/"); //path указывает на директорию
        int i = 0;
        for (File file : dir.listFiles()) {
            BufferedImage img = ImageIO.read(file); // load image
            BufferedImage scaledImg = Scalr.resize(img, Scalr.Method.ULTRA_QUALITY, 128, 128);
            try {
                i++;
                ImageIO.write(scaledImg, "png", new File("D:/FishNew128/" + file.getName()));
            } catch (IOException e) {
                e.printStackTrace();
            }
            System.out.println(file.getName());
        }
    }
}

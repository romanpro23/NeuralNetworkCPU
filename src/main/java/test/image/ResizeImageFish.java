package test.image;

import org.imgscalr.Scalr;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class ResizeImageFish {
    public static void main(String[] args) throws IOException {
        File dir = new File("D:/Fishing/OldStyle/"); //path указывает на директорию
        for (File file : dir.listFiles()) {
            BufferedImage img = ImageIO.read(file); // load image
            BufferedImage scaledImg = Scalr.resize(img, Scalr.Method.ULTRA_QUALITY, 256, 256);

            BufferedImage result = new BufferedImage(256, 256, BufferedImage.TYPE_INT_ARGB);

            int i0, j0;

            j0 = (256 - scaledImg.getHeight()) / 2;
            i0 = (256 - scaledImg.getWidth()) / 2;

            System.out.println(i0 +  " " + j0);

            for (int i = 0; i < scaledImg.getWidth(); i++) {
                for (int j = 0; j < scaledImg.getHeight(); j++) {
                    try {
                        Color color = new Color(scaledImg.getRGB(i, j), true);
                        result.setRGB(i + i0, j + j0, color.getRGB());
                    } catch (Exception e) {

                    }
                }
            }
            try {
                ImageIO.write(result, "png", new File("D:/Fishing/Old/" + file.getName()));
            } catch (IOException e) {
                e.printStackTrace();
            }
            System.out.println(file.getName());
        }
    }
}

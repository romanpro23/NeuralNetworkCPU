package test.image;

import org.imgscalr.Scalr;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class ResizeImage {
    public static void main(String[] args) throws IOException {
        File dir = new File("D:\\Fishing\\Old\\"); //path указывает на директорию
        for (File file : dir.listFiles()) {
            BufferedImage img = ImageIO.read(file); // load image
            int h = 256, w = 256;
            BufferedImage result = new BufferedImage(h, w, BufferedImage.TYPE_INT_RGB);
            for (int i = 0; i < w; i++) {
                for (int j = 0; j < h; j++) {
                    try {
                        int RGBA = img.getRGB(i, j);
                        int alpha = (RGBA >> 24) & 255;
                        int red = (RGBA >> 16) & 255;
                        int green = (RGBA >> 8) & 255;
                        int blue = RGBA & 255;
                        Color color;
                        if(alpha > 50) {
                            color = new Color(red, green, blue);
                        } else {
                            color = Color.WHITE;
                        }
                        result.setRGB(i, j,  color.getRGB());
                    } catch (Exception e) {

                    }
                }
            }
            try {
                ImageIO.write(result, "png", new File("D:\\Fishing\\Old\\" + file.getName()));
            } catch (IOException e) {
                e.printStackTrace();
            }
            System.out.println(file.getName());
        }
    }
}

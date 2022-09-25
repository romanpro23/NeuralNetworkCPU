package data;

import nnarrays.NNVector;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class ImageCreator {
    public static void drawImage(NNVector vector, int h, int w, String nameImg, String path) {
        drawImage(vector, h, w, nameImg, path, false);
    }
    public static void drawImage(NNVector vector, int h, int w, String nameImg, String path, boolean isTanh) {
        Color color;
        BufferedImage result = new BufferedImage(h, w, BufferedImage.TYPE_INT_RGB);
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                try {
                    float data = vector.get(i*w + j);
                    if(isTanh){
                        data *= 0.5f;
                        data += 0.5f;
                    }
                    color = new Color(data, data, data);
                    result.setRGB(j, i, color.getRGB());
                } catch (Exception e) {

                }
            }
        }

        File output = new File(path + "/" + nameImg + ".png");
        try {
            ImageIO.write(result, "png", output);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}

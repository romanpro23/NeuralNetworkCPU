package data;

import nnarrays.NNTensor;
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

    public static void drawColorImage(NNTensor tensor, int h, int w, String nameImg, String path) {
        drawColorImage(tensor, h, w, nameImg, path, false);
    }
    public static void drawColorImage(NNTensor tensor, int h, int w, String nameImg, String path, boolean tanh) {
        Color color;
        BufferedImage result = new BufferedImage(h, w, BufferedImage.TYPE_INT_RGB);
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                try {
                    float red = tensor.get(i, j, 0);
                    float green = tensor.get(i, j, 1);
                    float blue = tensor.get(i, j, 2);
                    if(tanh){
                        red *= 0.5f;
                        red += 0.5f;
                        green *= 0.5f;
                        green += 0.5f;
                        blue *= 0.5f;
                        blue += 0.5f;
                    }
                    color = new Color(red, green, blue);
                    result.setRGB(i, j, color.getRGB());
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

    public static void drawColorImage(NNVector vector, int h, int w, String nameImg, String path) {
        Color color;
        BufferedImage result = new BufferedImage(h, w, BufferedImage.TYPE_INT_RGB);
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                try {
                    float red = vector.get(i*w + j);
                    float green = vector.get(i*w + j + h*w);
                    float blue = vector.get(i*w + j + h*w*2);
                    color = new Color(red, green, blue);
                    result.setRGB(i, j, color.getRGB());
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

    public static void drawImage(NNTensor tensor, int h, int w, String nameImg, String path) {
        Color color;
        BufferedImage result = new BufferedImage(h, w, BufferedImage.TYPE_INT_RGB);
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                try {
                    float data = tensor.get(i, j, 0);
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

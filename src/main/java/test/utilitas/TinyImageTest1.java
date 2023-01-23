package test.utilitas;

import org.imgscalr.Scalr;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

public class TinyImageTest1 {
    public static void main(String[] args) throws IOException {
        String pathOut = "D:\\datasets\\ImageNet\\valid\\";

        FileWriter writer = new FileWriter("D:\\datasets\\ImageNet\\labels1k.txt");
        File dir = new File("D:\\datasets\\ImageNet100\\train");
        for (File file : dir.listFiles()) {
            File f = new File(pathOut + file.getName());
//            Files.createDirectories(Path.of(f.getPath()));
            writer.write(file.getName() + "\n");
            writer.flush();
            System.out.println(file.getName());
        }

//        int[] counter = new int[1000];
//        File dirV = new File("D:\\datasets\\ImageNet100\\valid");
//        Scanner scanner = new Scanner(new File("D:\\datasets\\ImageNet\\val_labels.txt"));
//        for (File img : dirV.listFiles()) {
//            try {
//                int n = scanner.nextInt();
//                BufferedImage imgs = ImageIO.read(img); // load image
//                imgs = Scalr.resize(imgs, Scalr.Method.ULTRA_QUALITY, Scalr.Mode.FIT_EXACT, 64, 64);
//                ImageIO.write(imgs, "png", new File(pathOut + dir.listFiles()[n].getName() + "\\" + counter[n] + ".png"));
//                counter[n]++;
//            } catch (Exception e) {
//                e.printStackTrace();
//            }
//        }
    }
}

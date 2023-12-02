package test.speechtotext.organizeintofolders;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Objects;

public class OrganizeIntoFolders {
    public static void main(String[] arg) throws IOException {

        String fileName = "C:/Levani/cv-corpus-15.0-2023-09-08/ka/validated.tsv";
        File file = new File(fileName);
        ArrayList<String[]> data = tsvr(file);

        File folder = new File("C:/Levani/cv-corpus-15.0-2023-09-08/ka/clips/");
        File[] listOfFiles = folder.listFiles();

        int ss = 0;
        int ss2 = 0;
        for (File afile : listOfFiles) {
            if (afile.isFile()) {
                var label = "";
                for (String[] datum : data) {
                    if (Objects.equals(datum[1], afile.getName())) {
                        label = datum[2];
                    }
                }

                if (!Objects.equals(label, "")) {
                    if (ss % 2000 == 0) {
                        ss2++;
                    }

                    File theDir = new File("C:/Levani/cv-corpus-15.0-2023-09-08/voice/" + ss2);
                    if (!theDir.exists()) {
                        theDir.mkdirs();
                    }

                    Path p1 = Paths.get("C:/Levani/cv-corpus-15.0-2023-09-08/voice/" + ss2 + "/" + afile.getName());
                    File thep1 = new File(p1.toString());
                    if (!thep1.exists()) {
                        Files.copy(afile.toPath(), p1, StandardCopyOption.REPLACE_EXISTING);
                    }

                    ss++;
                }
            }
        }
    }

    public static ArrayList<String[]> tsvr(File test2) {
        ArrayList<String[]> Data = new ArrayList<>(); //initializing a new ArrayList out of String[]'s
        try (BufferedReader TSVReader = new BufferedReader(new FileReader(test2))) {
            String line = null;
            while ((line = TSVReader.readLine()) != null) {
                String[] lineItems = line.split("\t"); //splitting the line and adding its items in String[]
                Data.add(lineItems); //adding the splitted line array to the ArrayList
            }
        } catch (Exception e) {
            System.out.println("Something went wrong");
        }
        return Data;
    }
}

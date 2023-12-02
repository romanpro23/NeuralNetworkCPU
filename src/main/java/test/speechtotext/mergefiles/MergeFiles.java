package test.speechtotext.mergefiles;

import static java.nio.file.StandardOpenOption.*;

import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.Paths;

public class MergeFiles
{
    public static void main(String[] arg) throws IOException {
        //if(arg.length<2) {
        //    System.err.println("Syntax: infiles... outfile");
        //    System.exit(1);
        //}

        arg = new String[8];
        arg[0] = "C:\\Levan\\SpeechToSpeech\\speech\\Speech\\data_train_1.csv";
        arg[1] = "C:\\Levan\\SpeechToSpeech\\speech\\Speech\\data_train_2.csv";
        arg[2] = "C:\\Levan\\SpeechToSpeech\\speech\\Speech\\data_train_3.csv";
        arg[3] = "C:\\Levan\\SpeechToSpeech\\speech\\Speech\\data_train_4.csv";
        arg[4] = "C:\\Levan\\SpeechToSpeech\\speech\\Speech\\data_train_5.csv";
        arg[5] = "C:\\Levan\\SpeechToSpeech\\speech\\Speech\\data_train_6.csv";
        arg[6] = "C:\\Levan\\SpeechToSpeech\\speech\\Speech\\data_train_7.csv";
        arg[7] = "C:\\Levan\\SpeechToSpeech\\speech\\Speech\\data_train.csv";

        /*arg = new String[9];
        arg[0] = "C:\\Levan\\SpeechToSpeech\\speech\\Speech\\onlytextdata_1.txt";
        arg[1] = "C:\\Levan\\SpeechToSpeech\\speech\\Speech\\onlytextdata_2.txt";
        arg[2] = "C:\\Levan\\SpeechToSpeech\\speech\\Speech\\onlytextdata_3.txt";
        arg[3] = "C:\\Levan\\SpeechToSpeech\\speech\\Speech\\onlytextdata_4.txt";
        arg[4] = "C:\\Levan\\SpeechToSpeech\\speech\\Speech\\onlytextdata_5.txt";
        arg[5] = "C:\\Levan\\SpeechToSpeech\\speech\\Speech\\onlytextdata_6.txt";
        arg[6] = "C:\\Levan\\SpeechToSpeech\\speech\\Speech\\onlytextdata_7.txt";
        arg[7] = "C:\\Levan\\SpeechToSpeech\\speech\\Speech\\1.txt";
        arg[8] = "C:\\Levan\\SpeechToSpeech\\speech\\Speech\\onlytextdata.txt";*/

        Path outFile=Paths.get(arg[arg.length-1]);
        System.out.println("TO "+outFile);
        try(FileChannel out=FileChannel.open(outFile, CREATE, WRITE)) {
            for(int ix=0, n=arg.length-1; ix<n; ix++) {
                Path inFile=Paths.get(arg[ix]);
                System.out.println(inFile+"...");
                try(FileChannel in=FileChannel.open(inFile, READ)) {
                    for(long p=0, l=in.size(); p<l; )
                        p+=in.transferTo(p, l-p, out);
                }
            }
        }
        System.out.println("DONE.");
    }
}
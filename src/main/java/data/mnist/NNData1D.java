package data.mnist;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;
import nnarrays.NNVector;

@AllArgsConstructor
@Getter
@Setter
public class NNData1D{
    NNVector[] input;
    NNVector[] output;
}
package nnarrays;

import lombok.Getter;

public class NNTensor extends NNArray{
    @Getter
    private final int column;
    @Getter
    private final int row;
    @Getter
    private final int depth;
    @Getter
    private int[] depthIndex;
    @Getter
    private int[] rowIndex;

    public NNTensor(int depth, int row, int column) {
        super(column * row * depth);
        this.column = column;
        this.row = row;
        this.depth = depth;
        countAxes = 3;

        initialize();
    }

    private void initialize(){
        depthIndex = new int[depth];
        rowIndex = new int[row];
        int sq = column * row;
        for (int i = 0; i < depth; i++) {
            depthIndex[i] = i * sq;
        }
        for (int i = 0; i < row; i++) {
            rowIndex[i] = i * column;
        }
    }

    public NNTensor(int depth, int row, int column, float[] data) {
        super(data);
        this.column = column;
        this.row = row;
        this.depth = depth;
        countAxes = 3;

        initialize();
    }

    @Override
    public int[] getSize(){
        return new int[]{depth, row, column};
    }
}

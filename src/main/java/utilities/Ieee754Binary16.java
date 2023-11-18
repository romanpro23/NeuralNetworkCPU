/*package utilities;

import static java.lang.Float.floatToIntBits;
import static java.lang.Float.intBitsToFloat;
import static java.lang.Math.pow;
import static java.lang.Math.scalb;

public class Ieee754Binary16 {//implements Comparable<Half> {

    // -----------------------------------------------------------------------------------------------------------------
    public static final int SIZE = 16;

    // -----------------------------------------------------------------------------------------------------------------
    public static final short SHORT_BITS_MAX_VALUE = 0b0_11110_1111111111;

    public static final short SHORT_BITS_MIN_VALUE = 0b0_00000_0000000001;

    public static final short SHORT_BITS_MIN_NORMAL = 0b0_00001_0000000000;

    public static final short SHORT_BITS_POSITIVE_ZERO = 0b0_00000_0000000000;

    public static final short SHORT_BITS_NEGATIVE_ZERO = (short) 0b1_00000_0000000000;

    private static final short SHORT_BITS_POSITIVE_INFINITY = 0b0_11111_0000000000;

    private static final short SHORT_BITS_NEGATIVE_INFINITY = (short) 0b1_11111_0000000000;

    private static final short SHORT_BITS_NaN = 0b0_11111_1000000000;

    public static final int MIN_EXPONENT = -14;

    public static final int MAX_EXPONENT = 15;

    // -----------------------------------------------------------------------------------------------------------------
    private static final int SIZE_SIGN = 1;

    static final int MASK_SIGN = 0b1_00000_0000000000;

    // -----------------------------------------------------------------------------------------------------------------
    static final int SIZE_EXPONENT = 5;

    static final int MASK_EXPONENT = 0b0_11111_0000000000;

    // -----------------------------------------------------------------------------------------------------------------
    static final int SIZE_SIGNIFICAND = 10;

    static final int MASK_SIGNIFICAND = 0b0_00000_1111111111;

    // -----------------------------------------------------------------------------------------------------------------
    private static final double SUBNORMAL_FACTOR = pow(2, 112); //pow(2, 126) / pow(2, 14);

    private static final double NORMAL_FACTOR = pow(2.0d, Float.MAX_EXPONENT - MAX_EXPONENT);

    // -----------------------------------------------------------------------------------------------------------------
    private static final int SIZE_SIGN_BINARY32 = SIZE_SIGN;

    static final int MASK_SIGN_32 = 0b1_00000000_00000000000000000000000;

    static final int SIZE_EXPONENT_BINARY32 = 8;

    static final int MASK_EXPONENT_32 = 0b0_11111111_00000000000000000000000;

    private static final int SIZE_EXPONENT_D = SIZE_EXPONENT_BINARY32 - SIZE_EXPONENT;

    static final int SIZE_SIGNIFICAND_BINARY32 = 23;

    static final int MASK_SIGNIFICAND_32 = 0b0_00000000_11111111111111111111111;

    private static final int SIZE_SIGNIFICAND_D = SIZE_SIGNIFICAND_BINARY32 - SIZE_SIGNIFICAND;

    // -----------------------------------------------------------------------------------------------------------------
    private static int binary32IntBits(final short value) {
//        return ((value & 0x8000) << Short.SIZE)
//               | (((value & 0x7C00) + 0x1C000) << SIZE_SIGNIFICAND_D)
//               | ((value & 0x03FF) << SIZE_SIGNIFICAND_D);
        // http://www.fox-toolkit.org/ftp/fasthalffloatconversion.pdf
        // f = ((h&0x8000)<<16) | (((h&0x7c00)+0x1C000)<<13) | ((h&0x03FF)<<13)
        return ((value & MASK_SIGN) << Short.SIZE)
                | (((value & MASK_EXPONENT) + 0x1C000) << SIZE_SIGNIFICAND_D)
                | ((value & MASK_SIGNIFICAND) << SIZE_SIGNIFICAND_D);
    }

    private static final int FP32_DENORMAL_MAGIC = 126 << 23;

    private static final float FP32_DENORMAL_FLOAT = intBitsToFloat(FP32_DENORMAL_MAGIC);

    public static float binary16ShortBitsToFloat(final short value) {
        if ((value & MASK_EXPONENT) == 0) {
            if ((value & MASK_SIGNIFICAND) == 0) { // zero
                if ((value & MASK_SIGN) == MASK_SIGN) {
                    return -.0f;
                } else {
                    return +.0f;
                }
            } else { // subnormal
                if (true) {
                    float o = intBitsToFloat(FP32_DENORMAL_MAGIC + (value & MASK_SIGNIFICAND));
                    o -= FP32_DENORMAL_FLOAT;
                    return (value & MASK_SIGN) == MASK_SIGN ? -o : o;
                }
                final int sign = (value & MASK_SIGN) << Short.SIZE;
                final int significand = (value & MASK_SIGNIFICAND) << SIZE_SIGNIFICAND_D;
//                return (float) (intBitsToFloat(sign | significand) * SUBNORMAL_FACTOR);
                return scalb(intBitsToFloat(sign | significand), 112);
            }
        }
        if ((value & MASK_EXPONENT) == MASK_EXPONENT) { // exponent -> all ones
            if ((value & MASK_SIGNIFICAND) == 0) { // significand -> all zeros -> (positive|negative) infinity
                return (value & MASK_SIGN) == 0 ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY;
            } else { // significand -> not all zeros -> not a number
                return Float.NaN;
            }
        }
        // normalized value
        return intBitsToFloat(binary32IntBits(value));
    }

    // -----------------------------------------------------------------------------------------------------------------
    public static short floatToBinary16ShortBits(final float value) {
        if (Float.isNaN(value)) {
            return SHORT_BITS_NaN;
        }
        if (Float.POSITIVE_INFINITY == value) {
            return SHORT_BITS_POSITIVE_INFINITY;
        }
        if (Float.NEGATIVE_INFINITY == value) {
            return SHORT_BITS_NEGATIVE_INFINITY;
        }
        final int i = floatToIntBits(value);
        final int exponent = i & MASK_EXPONENT_32;
        final int significand = i & MASK_SIGNIFICAND_32;
        if (exponent == 0) {
            if (significand == 0) { // zero
                if ((i & MASK_SIGN_32) == MASK_SIGN_32) {
                    return (short) 0b1_00000_0000000000;
                } else {
                    return 0b0_00000_0000000000;
                }
            } else { // subnormal
//                return (short) (((i & MASK_SIGN_32) >>> Short.SIZE) | (significand >> SIZE_SIGNIFICAND_D));
//                return floatToBinary16ShortBits((float) (value / SUBNORMAL_FACTOR));
//                return floatToBinary16ShortBits((float) (value / SUBNORMAL_FACTOR));
                return floatToBinary16ShortBits(scalb(value, -112));
            }
        }
        // http://www.fox-toolkit.org/ftp/fasthalffloatconversion.pdf
        // h = ((f>>16)&0x8000)|((((f&0x7f800000)-0x38000000)>>13)&0x7c00)|((f>>13)&0x03ff)
        // normalized
        return (short) (
                ((i >> Short.SIZE) & MASK_SIGN)
                        | ((((i & 0x7F800000) - 0x38000000) >> SIZE_SIGNIFICAND_D) & MASK_EXPONENT)
                        | ((i >> SIZE_SIGNIFICAND_D) & MASK_SIGNIFICAND)
        );
    }
}*/

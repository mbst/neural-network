package com.metabroadcast.machinelearning.math;

import static com.google.common.base.Preconditions.checkArgument;

/**
 * Class for normalizing input for the {@link com.metabroadcast.machinelearning.FeedForwardNetwork}.
 * 
 * @author james
 */
public class Normaliser {
    
    private Normaliser(){};
    
    /**
     * Normalises a value to between min and max.
     */
    public static double normalise(double x, double min, double max) {
        checkArgument(min < max, "min should be bigger than max, min was " + min + " and max was " + max);
        
        if (x < min || x > max) {
            x = Math.min(1, Math.max(0, x));
        }
        
        double standardised = (x - min) / (max - min);
        if (standardised > 1) {
            return 1;
        }
        return standardised;
    }
    
    /**
     * Normalises a value to between 0 and max.
     */
    public static double normalise(double x, double max) {
        return normalise(x, 0, max);
    }
    
    /**
     * Denormalises a value which was normalised in the range min to max.
     */
    public static double denormalise(double x, double min, double max) {
        return ((min - max) * x - 1 * min) * -1;
    }
    
    /**
     * Denormalises a value which was normalised in the range 0 to max
     */
    public static double denormalise(double x, double max) {
        return denormalise(x, 0, max);
    }

}

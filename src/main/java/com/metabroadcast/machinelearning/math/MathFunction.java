package com.metabroadcast.machinelearning.math;

/**
 * A mathematical function that can be used by the {@link com.metabroadcast.machinelearning.FeedForwardNetwork}.
 * 
 * @author james
 */
public interface MathFunction {
    
    double process(double input);
    
}

package com.metabroadcast.neuralnetwork.math;

/**
 * A mathematical function that can be used by the {@link com.metabroadcast.neuralnetwork.FeedForwardNetwork}.
 * 
 * @author james
 */
public interface MathFunction {
    
    double process(double input);
    
}

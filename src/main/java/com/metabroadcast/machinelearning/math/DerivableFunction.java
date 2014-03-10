package com.metabroadcast.machinelearning.math;

/**
 * A mathematical function that can be derived.
 * 
 * This is required for {@link com.metabroadcast.canary.machinelearning.Backpropagation} learning when used in neural networks.
 * 
 * @author james
 */
public interface DerivableFunction extends MathFunction {
    
    /**
     * Returns the derivative of the {@link com.metabroadcast.machinelearning.math.MathFunction#process(double)} method.
     */
    double derivative(double input);
}

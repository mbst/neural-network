package com.metabroadcast.neuralnetwork.math;

import java.io.Serializable;

/**
 * Function that represents a sine wave.
 * Note that this is not the same as {@link java.lang.Math#sin()}.
 * 
 * @author james
 */
public class SineDerivableFunction implements DerivableFunction, Serializable {

    private static final long serialVersionUID = 1L;

    /**
     * @return a value between 0 and 1.
     */
    public double process(double num) {
        return 1 / (1 + Math.exp(-num));
    }

    public double derivative(double num) {
        return num * (1 - num);
    }

}

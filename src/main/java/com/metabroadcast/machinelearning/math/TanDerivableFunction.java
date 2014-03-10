package com.metabroadcast.machinelearning.math;

import java.io.Serializable;

public class TanDerivableFunction implements DerivableFunction, Serializable {

    private static final long serialVersionUID = 1L;

    public double process(double num) {

        double denominator = Math.exp(num * 2.0) + 1.0;
        double numerator = Math.exp(num * 2.0) - 1.0;

        return numerator / denominator;
    }

    public double derivative(double num) {

        double value = (1.0 - Math.pow(process(num), 2.0));

        if (value > Double.MAX_VALUE)
            value = Double.MAX_VALUE;
        else if (value < -Double.MAX_VALUE)
            value = -Double.MAX_VALUE;

        return value;
    }

}

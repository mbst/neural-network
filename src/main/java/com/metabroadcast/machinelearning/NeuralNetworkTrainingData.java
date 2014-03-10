package com.metabroadcast.machinelearning;

import static com.google.common.base.Preconditions.checkNotNull;

public class NeuralNetworkTrainingData {
    
    private final double[][] inputs;
    private final double[] actuals;
    
    public NeuralNetworkTrainingData(double[][] inputs, double[] actuals) {
        this.inputs = checkNotNull(inputs);
        this.actuals = checkNotNull(actuals);
    }
    
    public double[][] getInputs() {
        return inputs;
    }
    
    public double[] getActuals() {
        return actuals;
    }

}

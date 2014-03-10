package com.metabroadcast.machinelearning;

/**
 * Trains a {@link com.metabroadcast.machinelearning.FeedForwardNetwork}.
 * 
 * @author james
 */
public interface Trainer {
    
    /**
     * @return the error of the current network based on the training set.
     */
    double getError();
    
    /**
     * @return the network being trained
     */
    FeedForwardNetwork getNetwork();
    
    /**
     * Does one iteration of training over all of the training set.
     */
    void iterate();
    
    /**
     * Does an iteration of training over a single item in the training set.
     * 
     * @param input the network input 
     * @param reinforcement the expected output of the network
     */
    void updateDelta(double[] input, double reinforcement);

}

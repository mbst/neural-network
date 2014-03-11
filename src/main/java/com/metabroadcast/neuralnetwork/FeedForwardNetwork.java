package com.metabroadcast.neuralnetwork;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.metabroadcast.neuralnetwork.math.MathFunction;
import com.metabroadcast.neuralnetwork.math.SineDerivableFunction;

public class FeedForwardNetwork {

    private static final Logger log = LoggerFactory.getLogger(FeedForwardNetwork.class);

    public double[] inputLayer;
    public double[][] hiddenWeights;
    public double[] hiddenLayer;
    public double[] outputWeights;
    public double output;
    private boolean squashOutput;
    private MathFunction mathFunction;

    private FeedForwardNetwork() {}

    private FeedForwardNetwork(int inputNodes, int hiddenNodes, boolean squashOutput, MathFunction mathFunction) {

        checkArgument(inputNodes > 0, "input nodes must be greater than 0");
        checkArgument(hiddenNodes > 0, "hidden nodes must be greater than 0");

        // add one for the bias
        this.inputLayer = new double[inputNodes + 1];
        this.hiddenWeights = new double[hiddenNodes][inputNodes + 1];
        this.hiddenLayer = new double[hiddenNodes + 1];
        this.outputWeights = new double[hiddenNodes + 1];
        this.squashOutput = squashOutput;
        this.mathFunction = checkNotNull(mathFunction, "A math function is required"); 
    }

    protected void initialiseWeightsRandomly() {

        // initialises all the weights to be random
        for (int i = 0; i < hiddenWeights.length; i++) {
            for (int j = 0; j < hiddenWeights[i].length; j++) {
                hiddenWeights[i][j] = Math.random() * 2 - 1;
            }
        }
        for (int i = 0; i < outputWeights.length; i++) {
            outputWeights[i] = Math.random() * 2 - 1;
        }
    }
    
    protected void initialiseWeightsToSetAmount(double amount) {
     // initialises all the weights to be random
        for (int i = 0; i < hiddenWeights.length; i++) {
            for (int j = 0; j < hiddenWeights[i].length; j++) {
                hiddenWeights[i][j] = amount;
            }
        }
        for (int i = 0; i < outputWeights.length; i++) {
            outputWeights[i] = amount;
        }
    }

    public int hiddenLayerLength() {
        return hiddenLayer.length;
    }

    public MathFunction getOutputFunction() {
        return mathFunction;
    }

    public double computeOutput(double[] input) {

        checkArgument(input.length < inputLayer.length,
                "input is longer than input layer for network");

        // input is input into the system
        log.trace("computing output..");
        log.trace("input length {}", input.length);
        log.trace("input layer length {}", inputLayer.length);

        // add in the bias for the system
        inputLayer[0] = 1;

        for (int i = 1; i <= input.length; i++) {
            inputLayer[i] = input[i - 1];
        }

        // goes through one by one
        for (int i = 0; i < hiddenWeights.length; i++) {
            double hiddenTotal = 0;

            for (int j = 0; j < hiddenWeights[i].length; j++) {
                hiddenTotal += hiddenWeights[i][j] * inputLayer[j];
            }

            hiddenLayer[i] = getOutputFunction().process(hiddenTotal);
        }

        hiddenLayer[hiddenLayer.length - 1] = 1;

        double outputTotal = 0;

        for (int i = 0; i < outputWeights.length; i++) {
            outputTotal += hiddenLayer[i] * outputWeights[i];
        }

        output = squashOutput ? limiter(getOutputFunction().process(outputTotal)) : getOutputFunction().process(outputTotal);

        return output;

    }

    public double calculateError(double[][] input, double[] ideal) {

        double error = 0;

        for (int i = 0; i < input.length; i++) {
            error += (computeOutput(input[i]) - ideal[i]) * (computeOutput(input[i]) - ideal[i]);
        }

        double meanError = error / ideal.length;

        return Math.sqrt(meanError);
    }

    public static double limiter(double num) {
        return Math.min(1, Math.max(0, num));
    }
    
    public static Builder builder() {
        return new Builder();
    }
    
    public static class Builder {
        
        private int inputNodes;
        private int hiddenNodes;
        private boolean squashOutput;
        private MathFunction outputFunction;
        
        public Builder withInputNodes(int inputNodes) {
            this.inputNodes = inputNodes;
            return this;
        }
        
        public Builder withHiddenNodes(int hiddenNodes) {
            this.hiddenNodes = hiddenNodes;
            return this;
        }
        
        public Builder squashOutput() {
            this.squashOutput = true;
            return this;
        }
        
        public Builder withSineFunction() {
            this.outputFunction = new SineDerivableFunction();
            return this;
        }
        
        /**
         * Builds and distributes the weights in the network randomly.
         * Use the {@link #buildAndDistributeWeights(double)} when testing.
         */
        public FeedForwardNetwork buildAndRandomlyDistributeWeights() {
            FeedForwardNetwork network = new FeedForwardNetwork(inputNodes, hiddenNodes, squashOutput, outputFunction);
            network.initialiseWeightsRandomly();
            return network;
        }
        
        /**
         * When testing, you will want to remove the random factor to ensure unit tests are consistent.
         * Use this method to assign the same weight across all of the network on initialization.
         */
        public FeedForwardNetwork buildAndDistributeWeights(double startingWeight) {
            FeedForwardNetwork network = new FeedForwardNetwork(inputNodes, hiddenNodes, squashOutput, outputFunction);
            network.initialiseWeightsToSetAmount(startingWeight);
            return network;
        }
    }

}

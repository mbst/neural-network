package com.metabroadcast.machinelearning;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;

import com.metabroadcast.machinelearning.math.MathFunction;
import com.metabroadcast.machinelearning.math.SineDerivableFunction;


public class SineFeedForwardNetwork extends FeedForwardNetwork implements Serializable {

    private static final long serialVersionUID = 1L;
    private static final SineDerivableFunction SINE_FUNCTION = new SineDerivableFunction();
    
    private SineFeedForwardNetwork() {};
    
    private SineFeedForwardNetwork(int inputNodes, int hiddenNodes) {
        super(inputNodes, hiddenNodes);
    }
      
    private SineFeedForwardNetwork(int inputNodes, int hiddenNodes, boolean squashOutput) {
        super(inputNodes, hiddenNodes, squashOutput);
    }

    @Override
    public MathFunction getOutputFunction() {
        return SINE_FUNCTION;
    }
    
    public void save(String fileName) throws IOException, ClassNotFoundException {
        try (ObjectOutputStream s = new ObjectOutputStream(new FileOutputStream(fileName, false))) {
            s.writeObject(this);
        }
    }

    public static SineFeedForwardNetwork load(String fileName) throws IOException, ClassNotFoundException {
        try (ObjectInputStream s = new ObjectInputStream(new FileInputStream(fileName))) {
            return (SineFeedForwardNetwork) s.readObject();
        }
    }
    
    public static Builder builder() {
        return new Builder();
    }
    
    public static class Builder {
        
        private int inputNodes;
        private int hiddenNodes;
        private boolean squashOutput;
        
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
        
        /**
         * Builds and distributes the weights in the network randomly.
         * Use the {@link #buildAndDistributeWeights(double)} when testing.
         */
        public SineFeedForwardNetwork buildAndRandomlyDistributeWeights() {
            SineFeedForwardNetwork network = new SineFeedForwardNetwork(inputNodes, hiddenNodes, squashOutput);
            network.initialiseWeightsRandomly();
            return network;
        }
        
        /**
         * When testing, you will want to remove the random factor to ensure unit tests are consistent.
         * Use this method to assign the same weight across all of the network on initialization.
         */
        public SineFeedForwardNetwork buildAndDistributeWeights(double startingWeight) {
            SineFeedForwardNetwork network = new SineFeedForwardNetwork(inputNodes, hiddenNodes, squashOutput);
            network.initialiseWeightsToSetAmount(startingWeight);
            return network;
        }
    }

}

package com.metabroadcast.neuralnetwork;

import static com.google.common.base.Preconditions.checkArgument;

import com.metabroadcast.neuralnetwork.math.DerivableFunction;

/**
 * An implementation of {@link Trainer} that uses the backpropagation
 * algorithm to train up the network. See <a href="http://en.wikipedia.org/wiki/Backpropagation">this</a> for more details
 * on how the algorithm works.
 * 
 * @author james
 */
public class BackPropagation implements Trainer {

    private final FeedForwardNetwork network;
    private final DerivableFunction outputFunction;    

    private final double learningRate;
    
    // the training set
    private final double[][] input;
    private final double[] ideal;
    
    //deltas
    private final double[] hiddenDeltas;
    private double outputDelta;

    public BackPropagation(FeedForwardNetwork network, double[][] input, double[] ideal,
            double learningRate) {
        this.network = network;
        
        checkArgument(network.getOutputFunction() instanceof DerivableFunction, 
                "The network given does not implement a derivable function, this is required for backpropagation.");
        this.outputFunction = (DerivableFunction)network.getOutputFunction();
        
        this.input = input;
        this.ideal = ideal;
        this.learningRate = learningRate;
        this.hiddenDeltas = new double[network.hiddenLayerLength()];
    }

    @Override
    public double getError() {
        return network.calculateError(input, ideal);
    }
    
    @Override
    public FeedForwardNetwork getNetwork() {
        return network;
    }

    @Override
    public void iterate() {
        for (int j = 0; j < input.length; j++) {

            updateDelta(input[j], ideal[j]);
        }
    }
    
    @Override
    public void updateDelta(double[] input, double reinforcement) {

        network.computeOutput(input);
        calcError(reinforcement);

    }

    private void calcError(double ideal) {
        
        findDeltas(ideal);
        learn();
    }

    private void findDeltas(double ideal) {

        outputDelta = 0;

        for (int i = 0; i < hiddenDeltas.length; i++) {
            hiddenDeltas[i] = 0;
        }

        // As its name implies, the backpropagation propagates backwards through
        // the neural network.
        outputDelta = outputFunction.derivative(network.output) * (ideal - network.output);

        // then do hidden deltas
        for (int i = 0; i < network.hiddenLayerLength(); i++) {
            hiddenDeltas[i] = outputFunction.derivative(network.hiddenLayer[i])
                * outputDelta
                * network.outputWeights[i];
        }
    }

    private void learn() {
        // all layers done now do learning

        for (int i = 0; i < network.hiddenWeights.length; i++) {
            for (int j = 0; j < network.hiddenWeights[i].length; j++) {
                network.hiddenWeights[i][j] += (learningRate * network.inputLayer[j] * hiddenDeltas[i]);
            }
        }

        for (int i = 0; i < network.outputWeights.length; i++) {
            network.outputWeights[i] += (learningRate * network.hiddenLayer[i] * outputDelta);
        }
    }

}

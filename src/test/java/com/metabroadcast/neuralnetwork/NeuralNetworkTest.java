package com.metabroadcast.neuralnetwork;

import static org.junit.Assert.fail;

import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.metabroadcast.neuralnetwork.BackPropagation;
import com.metabroadcast.neuralnetwork.FeedForwardNetwork;
import com.metabroadcast.neuralnetwork.Trainer;

public class NeuralNetworkTest {
    
    private Logger logger = LoggerFactory.getLogger(NeuralNetworkTest.class);

    @Test(expected = IllegalArgumentException.class)
    public void testWhenInputLongerThanInputLayerThrowError() {
        FeedForwardNetwork network = FeedForwardNetwork.builder()
                .withInputNodes(2)
                .withHiddenNodes(1)
                .withSineFunction()
                .buildAndRandomlyDistributeWeights();
        double largeTrainingSet[] = { 1, 0, 0, 0 };

        network.computeOutput(largeTrainingSet);
    }

    @Test
    public void testLearnsAnd() {
        FeedForwardNetwork network = FeedForwardNetwork.builder()
                .withInputNodes(3)
                .withHiddenNodes(1)
                .withSineFunction()
                .buildAndDistributeWeights(0.5);

        double trainingAndInput[][] = { { 1, 1, 1 }, { 0, 0, 0 }, { 1, 0, 1 }, { 0, 1, 0 } };

        double trainingAndActual[] = { 1, 0, 0, 0 };

        double andTestInput[][] = { { 0, 1, 0 }, { 1, 0, 0 }, { 1, 1, 1 }, { 0, 1, 0 } };

        double andTestActual[] = { 0, 0, 1, 0 };

        Trainer train = new BackPropagation(network, trainingAndInput, trainingAndActual, 0.1);

        int epoch = 0;
        double error = 1;

        while ((epoch < 5000) && (error > 0.01)) {
            train.iterate();
            error = train.getError();
            logger.debug("iteration #" + epoch + " : " + error);
            epoch++;
        }

        for (int i = 0; i < andTestInput.length; i++) {
            double actual = train.getNetwork().computeOutput(andTestInput[i]);
            
            logger.debug("actual " + actual + ", expected " + andTestActual[i]);
            if (Math.abs(actual - andTestActual[i]) > 0.1) {
                fail("Output is not close enough");
            }
        }
    }
}

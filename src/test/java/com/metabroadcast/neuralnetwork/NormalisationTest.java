package com.metabroadcast.neuralnetwork;

import static com.metabroadcast.neuralnetwork.math.Normaliser.denormalise;
import static com.metabroadcast.neuralnetwork.math.Normaliser.normalise;
import static org.junit.Assert.assertEquals;

import org.junit.Test;


public class NormalisationTest {
    
    @Test
    public void testNormaliseAndDenormaliseWithZeroMininimum() {
        
        double normalised = normalise(50, 0, 1000);
        
        double val = denormalise(normalised, 0, 1000);
        
        assertEquals(50, (int)val);
        
    }
    
    @Test
    public void testNormaliseAndDenormaliseWithFortyMininimum() {
        
        double normalised = normalise(50, 40, 1000);
        
        double val = denormalise(normalised, 40, 1000);
        
        assertEquals(50, (int)val);
        
    }

}

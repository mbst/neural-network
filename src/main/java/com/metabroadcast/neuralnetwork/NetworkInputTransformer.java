package com.metabroadcast.neuralnetwork;

import java.util.List;

/**
 * Transforms an item into a set of weights to be used by a neural network.
 * 
 * @author james
 *
 * @param <T> the object to transform
 */
public interface NetworkInputTransformer <T> {
    
    /**
     * Transforms an item T to a set of weights.
     * 
     * @param item the item to transform
     * @return a set of weights
     */
    double[] transform(T item);
    
    /**
     * Transforms a string set of fields to a set of weights.
     * This is likely to be used when you have output the item to a file.
     * 
     * @param fields the fields which correspond to T
     * @return a set of weights
     */
    double[] transform(List<String> fields);

}

package com.metabroadcast.machinelearning;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.metabroadcast.machinelearning.math.MathFunction;
import com.metabroadcast.machinelearning.math.MathFunctionSerializer;

/**
 * Provides a persistence method to save or load a {@link com.metabroadcast.machinelearning.FeedForwardNetwork}.
 * This implementation uses gson to output the file to json.
 * 
 * @author james
 */
public class NetworkFilePersistence {

    private static final Gson gson = new GsonBuilder()
            .registerTypeAdapter(MathFunction.class, new MathFunctionSerializer())
            .create();

    public static void save(FeedForwardNetwork network, String fileName) throws IOException {
        String json = gson.toJson(network);
        try (FileWriter writer = new FileWriter(fileName, false)) {
            writer.append(json);
        }
    }

    public static FeedForwardNetwork load(String fileName) throws IOException {
        try (BufferedReader reader = new BufferedReader(new FileReader(fileName))) {
            return gson.fromJson(reader, FeedForwardNetwork.class);
        }
    }

}

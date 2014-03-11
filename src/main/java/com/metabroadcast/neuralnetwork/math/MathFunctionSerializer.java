package com.metabroadcast.neuralnetwork.math;

import java.lang.reflect.Type;

import com.google.gson.JsonDeserializationContext;
import com.google.gson.JsonDeserializer;
import com.google.gson.JsonElement;
import com.google.gson.JsonParseException;
import com.google.gson.JsonPrimitive;
import com.google.gson.JsonSerializationContext;
import com.google.gson.JsonSerializer;


public class MathFunctionSerializer implements JsonSerializer<MathFunction>, JsonDeserializer<MathFunction> {

    private static final String SIN = "SIN";
    private static final String TAN = "TAN";

    @Override
    public MathFunction deserialize(JsonElement json, Type typeOfT,
            JsonDeserializationContext context) throws JsonParseException {
        String type = json.getAsString();
        if (type.equals(SIN)) {
            return new SineDerivableFunction();
        } else if (type.equals(TAN)) {
            return new TanDerivableFunction();
        }
        throw new JsonParseException("Unknown math function");
    }

    @Override
    public JsonElement serialize(MathFunction src, Type typeOfSrc, JsonSerializationContext context) {
        if (src instanceof SineDerivableFunction) {
            return new JsonPrimitive("SINE");
        } else if (src instanceof TanDerivableFunction) {
            return new JsonPrimitive(TAN);
        }
        throw new IllegalArgumentException("Unknown math function");
    }

}

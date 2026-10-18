package com.virtualtouchpad;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

public class ConfigLoader {

    private final int fps;
    private final String sleepHotkey;
    private final Map<Integer, Integer> gestureToCommand;

    public ConfigLoader(String path) throws IOException {
        ObjectMapper mapper = new ObjectMapper();
        JsonNode root = mapper.readTree(new File(path));

        this.fps = root.has("FPS") ? root.get("FPS").asInt(30) : 30;
        this.sleepHotkey = root.has("sleep") ? root.get("sleep").asText("CTRL+ALT+Q") : "CTRL+ALT+Q";

        this.gestureToCommand = new HashMap<>();
        JsonNode alph = root.get("alph");
        if (alph != null && alph.isObject()) {
            Iterator<Map.Entry<String, JsonNode>> it = alph.fields();
            while (it.hasNext()) {
                Map.Entry<String, JsonNode> e = it.next();
                gestureToCommand.put(Integer.parseInt(e.getKey()), e.getValue().asInt());
            }
        }
    }

    public int getFps()                  { return fps; }
    public String getSleepHotkey()       { return sleepHotkey; }
    public Map<Integer, Integer> getGestureToCommand() { return gestureToCommand; }
}
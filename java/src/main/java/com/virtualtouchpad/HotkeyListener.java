package com.virtualtouchpad;

import com.github.kwhat.jnativehook.GlobalScreen;
import com.github.kwhat.jnativehook.keyboard.NativeKeyEvent;
import com.github.kwhat.jnativehook.keyboard.NativeKeyListener;

import java.util.HashSet;
import java.util.Set;

public class HotkeyListener implements NativeKeyListener {

    private final Set<Integer> pressedKeys = new HashSet<>();
    private final String hotkey;
    private final CameraCapture camera;

    public HotkeyListener(String hotkey, CameraCapture camera) {
        this.hotkey = hotkey.toUpperCase();
        this.camera = camera;
    }

    public void register() throws Exception {
        GlobalScreen.registerNativeHook();
        GlobalScreen.addNativeKeyListener(this);
    }

    @Override
    public void nativeKeyPressed(NativeKeyEvent e) {
        pressedKeys.add(e.getKeyCode());
        if (isHotkeyPressed()) {
            camera.setSleeping(true);
            System.out.println("Переход в спящий режим...");
            // Для выхода из сна можно предусмотреть отдельную комбинацию
        }
    }

    @Override
    public void nativeKeyReleased(NativeKeyEvent e) {
        pressedKeys.remove(e.getKeyCode());
    }

    @Override
    public void nativeKeyTyped(NativeKeyEvent e) {
        // не используется
    }

    private boolean isHotkeyPressed() {
        // Простая проверка: все клавиши из строки hotkey нажаты
        // Пример: "CTRL+ALT+Q"
        String[] parts = hotkey.split("\\+");
        for (String part : parts) {
            int code = keyCodeFromString(part.trim());
            if (code == -1 || !pressedKeys.contains(code)) return false;
        }
        return true;
    }

    private int keyCodeFromString(String key) {
        return switch (key) {
            case "CTRL"  -> NativeKeyEvent.VC_CONTROL;
            case "ALT"   -> NativeKeyEvent.VC_ALT;
            case "SHIFT" -> NativeKeyEvent.VC_SHIFT;
            case "Q"     -> NativeKeyEvent.VC_Q;
            case "W"     -> NativeKeyEvent.VC_W;
            case "E"     -> NativeKeyEvent.VC_E;
            case "R"     -> NativeKeyEvent.VC_R;
            case "T"     -> NativeKeyEvent.VC_T;
            case "Y"     -> NativeKeyEvent.VC_Y;
            case "U"     -> NativeKeyEvent.VC_U;
            case "I"     -> NativeKeyEvent.VC_I;
            case "O"     -> NativeKeyEvent.VC_O;
            case "P"     -> NativeKeyEvent.VC_P;
            case "A"     -> NativeKeyEvent.VC_A;
            case "S"     -> NativeKeyEvent.VC_S;
            case "D"     -> NativeKeyEvent.VC_D;
            case "F"     -> NativeKeyEvent.VC_F;
            case "G"     -> NativeKeyEvent.VC_G;
            case "H"     -> NativeKeyEvent.VC_H;
            case "J"     -> NativeKeyEvent.VC_J;
            case "K"     -> NativeKeyEvent.VC_K;
            case "L"     -> NativeKeyEvent.VC_L;
            case "Z"     -> NativeKeyEvent.VC_Z;
            case "X"     -> NativeKeyEvent.VC_X;
            case "C"     -> NativeKeyEvent.VC_C;
            case "V"     -> NativeKeyEvent.VC_V;
            case "B"     -> NativeKeyEvent.VC_B;
            case "N"     -> NativeKeyEvent.VC_N;
            case "M"     -> NativeKeyEvent.VC_M;
            default -> -1;
        };
    }
}
package com.virtualtouchpad;

import com.github.kwhat.jnativehook.GlobalScreen;
import java.util.logging.Level;
import java.util.logging.Logger;

public class Main {

    // ===== КОНСТАНТЫ (менять перед сборкой) =====
    public static final int W       = 160;
    public static final int H       = 160;
    public static final int W_BOX   = 80;
    public static final int H_BOX   = 80;
    public static final int DEFAULT_FPS = 30;

    public static void main(String[] args) throws Exception {
        // Отключаем назойливое логирование JNativeHook
        Logger logger = Logger.getLogger(GlobalScreen.class.getPackage().getName());
        logger.setLevel(Level.OFF);

        // 1. Загружаем конфигурацию
        ConfigLoader config = new ConfigLoader(".virtual_touchpad");

        // 2. Инициализируем TensorFlow и загружаем модели
        TensorFlowManager tf = new TensorFlowManager(
                "src/main/resources/detector_box",
                "src/main/resources/detector_coord"
        );
        tf.loadModels();

        // 3. Создаём контроллер системы (мышь, скриншоты, приложения)
        SystemController system = new SystemController();

        // 4. Создаём обработчик жестов
        GestureProcessor gestures = new GestureProcessor(system, config);

        // 5. Запускаем захват камеры в отдельном потоке
        CameraCapture camera = new CameraCapture(config.getFps(), tf, gestures);
        Thread cameraThread = new Thread(camera, "camera-thread");
        cameraThread.setDaemon(true);
        cameraThread.start();

        // 6. Слушаем глобальную горячую клавишу для сна
        HotkeyListener hotkey = new HotkeyListener(config.getSleepHotkey(), camera);
        hotkey.register();

        System.out.println("VirtualTouchpad запущен. Нажмите " + config.getSleepHotkey() + " для сна.");

        // Держим главный поток живым
        Thread.currentThread().join();
    }
}
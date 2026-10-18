package com.virtualtouchpad;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;

public class CameraCapture implements Runnable {

    private final int targetFps;
    private final TensorFlowManager tf;
    private final GestureProcessor gestures;

    private volatile boolean sleeping = false;
    private volatile boolean running  = true;

    public CameraCapture(int fps, TensorFlowManager tf, GestureProcessor gestures) {
        this.targetFps = fps;
        this.tf = tf;
        this.gestures = gestures;
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    @Override
    public void run() {
        VideoCapture camera = new VideoCapture(0);
        camera.set(Videoio.CAP_PROP_FRAME_WIDTH, Main.W);
        camera.set(Videoio.CAP_PROP_FRAME_HEIGHT, Main.H);
        camera.set(Videoio.CAP_PROP_FPS, targetFps);

        Mat frame = new Mat();
        long frameInterval = 1000L / targetFps;

        while (running) {
            if (sleeping) {
                try { Thread.sleep(200); } catch (InterruptedException ignored) {}
                continue;
            }

            long start = System.currentTimeMillis();

            if (!camera.read(frame) || frame.empty()) {
                continue;
            }

            // 1. Приводим к нужному размеру и оттенкам серого
            Mat gray = new Mat();
            Imgproc.cvtColor(frame, gray, Imgproc.COLOR_BGR2GRAY);
            Imgproc.resize(gray, gray, new Size(Main.W, Main.H));

            // 2. Первая нейросеть → маска (0..1)
            float[][] mask = tf.runDetectorBox(gray);

            // 3. Находим bounding-box'ы по маске
            var boxes = MaskUtils.extractBoxes(mask, Main.W_BOX, Main.H_BOX);

            if (!boxes.isEmpty()) {
                // 4. Для каждого бокса запускаем вторую нейросеть → 21 точка
                var hands = tf.runDetectorCoord(gray, boxes);
                gestures.update(hands);
            } else {
                gestures.update(null);
            }

            // 5. Отрисовка (курсор/подсветка) — можно вывести в окно или оверлей
            //    Здесь только консольный вывод для наглядности
            System.out.printf("FPS: %.1f | рук: %d%n",
                    1000.0 / (System.currentTimeMillis() - start + 1), boxes.size());

            // Поддержание целевого FPS
            long elapsed = System.currentTimeMillis() - start;
            if (elapsed < frameInterval) {
                try { Thread.sleep(frameInterval - elapsed); } catch (InterruptedException ignored) {}
            }
        }

        camera.release();
    }

    public void setSleeping(boolean sleeping) { this.sleeping = sleeping; }
    public void stop()                        { this.running = false; }
}
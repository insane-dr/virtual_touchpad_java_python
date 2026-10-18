package com.virtualtouchpad;

import org.tensorflow.*;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.types.TFloat32;

import java.util.Collections;
import java.util.List;

public class TensorFlowManager {

    private SavedModelBundle detectorBox;
    private SavedModelBundle detectorCoord;

    private final String boxPath;
    private final String coordPath;

    public TensorFlowManager(String boxPath, String coordPath) {
        this.boxPath = boxPath;
        this.coordPath = coordPath;
    }

    public void loadModels() {
        detectorBox   = SavedModelBundle.load(boxPath, "serve");
        detectorCoord = SavedModelBundle.load(coordPath, "serve");
        System.out.println("Модели TensorFlow загружены.");
    }

    /**
     * Первая сеть: вход [1, W, H, 1] → выход [1, W, H, 1] (маска 0..1)
     */
    public float[][] runDetectorBox(org.opencv.core.Mat gray) {
        // TFloat32 уже является FloatNdArray, создаём тензор нужной формы
        try (TFloat32 input = TFloat32.tensorOf(Shape.of(1, Main.W, Main.H, 1))) {

            // Заполняем входной тензор пикселями напрямую через input
            for (int y = 0; y < Main.H; y++) {
                for (int x = 0; x < Main.W; x++) {
                    double[] px = gray.get(y, x);
                    input.setFloat((float)(px[0] / 255.0), 0, y, x, 0);
                }
            }

            try (Result result = detectorBox.session()
                    .runner()
                    .feed("serving_default_input", input)
                    .fetch("StatefulPartitionedCall")
                    .run()) {

                TFloat32 output = (TFloat32) result.get(0);
                Shape shape = output.shape();
                // Shape.get(int) вместо устаревшего size(int)
                int h = (int) shape.get(1);
                int w = (int) shape.get(2);

                float[][] mask = new float[h][w];
                for (int y = 0; y < h; y++) {
                    for (int x = 0; x < w; x++) {
                        // TFloat32 напрямую предоставляет getFloat
                        mask[y][x] = output.getFloat(0, y, x, 0);
                    }
                }
                return mask;
            }
        }
    }

    /**
     * Вторая сеть: для каждого бокса возвращает 21 точку (x, y).
     */
    public List<float[][]> runDetectorCoord(org.opencv.core.Mat gray,
                                            List<org.opencv.core.Rect> boxes) {

        if (boxes.isEmpty()) return Collections.emptyList();

        org.opencv.core.Rect box = boxes.get(0);

        // Вырезаем область бокса и масштабируем под вход сети (например, 224x224)
        org.opencv.core.Mat crop = new org.opencv.core.Mat(gray, box);
        org.opencv.core.Mat resized = new org.opencv.core.Mat();
        org.opencv.imgproc.Imgproc.resize(crop, resized, new org.opencv.core.Size(224, 224));

        try (TFloat32 input = TFloat32.tensorOf(Shape.of(1, 224, 224, 1))) {
            // Заполняем входной тензор
            for (int y = 0; y < 224; y++) {
                for (int x = 0; x < 224; x++) {
                    double[] px = resized.get(y, x);
                    input.setFloat((float)(px[0] / 255.0), 0, y, x, 0);
                }
            }

            try (Result result = detectorCoord.session()
                    .runner()
                    .feed("serving_default_input", input)
                    .fetch("StatefulPartitionedCall")
                    .run()) {

                TFloat32 output = (TFloat32) result.get(0); // Shape: [1, 21, 2]

                float[][] points = new float[21][2];
                for (int i = 0; i < 21; i++) {
                    points[i][0] = output.getFloat(0, i, 0);
                    points[i][1] = output.getFloat(0, i, 1);
                }
                // Явное приведение типа, чтобы вернуть List<float[][]>
                return Collections.singletonList(points);
            }
        }
    }
}
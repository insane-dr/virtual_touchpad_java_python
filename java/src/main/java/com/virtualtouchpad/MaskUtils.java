package com.virtualtouchpad;

import org.opencv.core.Rect;
import java.util.ArrayList;
import java.util.List;

public class MaskUtils {

    /**
     * Бинаризует маску и возвращает прямоугольники, масштабированные
     * под размер W_BOX × H_BOX.
     */
    public static List<Rect> extractBoxes(float[][] mask, int wBox, int hBox) {
        int h = mask.length;
        int w = mask[0].length;
        boolean[][] visited = new boolean[h][w];
        List<Rect> boxes = new ArrayList<>();

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                if (mask[y][x] > 0.5 && !visited[y][x]) {
                    // BFS для поиска связной компоненты
                    int minX = x, maxX = x, minY = y, maxY = y;
                    var queue = new java.util.LinkedList<int[]>();
                    queue.add(new int[]{x, y});
                    visited[y][x] = true;

                    while (!queue.isEmpty()) {
                        int[] p = queue.poll();
                        int px = p[0], py = p[1];
                        minX = Math.min(minX, px);
                        maxX = Math.max(maxX, px);
                        minY = Math.min(minY, py);
                        maxY = Math.max(maxY, py);

                        for (int dy = -1; dy <= 1; dy++) {
                            for (int dx = -1; dx <= 1; dx++) {
                                int nx = px + dx, ny = py + dy;
                                if (nx >= 0 && nx < w && ny >= 0 && ny < h
                                        && !visited[ny][nx] && mask[ny][nx] > 0.5) {
                                    visited[ny][nx] = true;
                                    queue.add(new int[]{nx, ny});
                                }
                            }
                        }
                    }

                    // Масштабируем координаты под W_BOX × H_BOX
                    int scaledX = (int)((float)minX / w * wBox);
                    int scaledY = (int)((float)minY / h * hBox);
                    int scaledW = Math.max(1, (int)((float)(maxX - minX) / w * wBox));
                    int scaledH = Math.max(1, (int)((float)(maxY - minY) / h * hBox));

                    boxes.add(new Rect(scaledX, scaledY, scaledW, scaledH));
                }
            }
        }
        return boxes;
    }
}
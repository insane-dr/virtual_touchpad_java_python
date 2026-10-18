package com.virtualtouchpad;

import java.util.List;

public class GestureProcessor {

    private final SystemController system;
    private final ConfigLoader config;

    // Последние координаты рук (21 точка для каждой руки)
    private List<float[][]> lastHands = List.of();

    // Текущий жест (числовой ID)
    private int currentGesture = 0;

    // Флаг, зажат ли «палец» (для перетаскивания)
    private boolean fingerDown = false;

    public GestureProcessor(SystemController system, ConfigLoader config) {
        this.system = system;
        this.config = config;
    }

    /**
     * Обновляет состояние на основе новых данных от нейросетей.
     */
    public void update(List<float[][]> hands) {
        this.lastHands = hands;

        if (hands == null || hands.isEmpty()) {
            // Рука исчезла — отпускаем кнопку мыши, если была зажата
            if (fingerDown) {
                system.mouseRelease();
                fingerDown = false;
            }
            return;
        }

        // === Распознавание жеста (заглушка) ===
        // Здесь должна быть логика классификации 21 точки в номер жеста.
        // Пока используем простую эвристику: количество «поднятых» пальцев.
        int gesture = recognizeGesture(hands.get(0));

        if (gesture != currentGesture) {
            currentGesture = gesture;
            onGestureChanged(gesture);
        }

        // === Управление мышью (пример) ===
        // Используем запястье (точка 0) как позицию курсора.
        float[] wrist = hands.get(0)[0];
        int screenX = (int)(wrist[0] * java.awt.Toolkit.getDefaultToolkit()
                .getScreenSize().width);
        int screenY = (int)(wrist[1] * java.awt.Toolkit.getDefaultToolkit()
                .getScreenSize().height);
        system.mouseMove(screenX, screenY);

        // Пример: если жест = 1 (указательный палец поднят) — зажимаем ЛКМ
        if (gesture == 1 && !fingerDown) {
            system.mousePress();
            fingerDown = true;
        } else if (gesture != 1 && fingerDown) {
            system.mouseRelease();
            fingerDown = false;
        }
    }

    /**
     * Примитивная эвристика: считаем «поднятые» пальцы по y-координате.
     * В реальном проекте замените на полноценный классификатор.
     */
    private int recognizeGesture(float[][] points) {
        // Точки 4, 8, 12, 16, 20 — кончики пальцев.
        // Сравниваем их y с y основания соответствующего пальца (3, 6, 10, 14, 18).
        int count = 0;
        int[][] pairs = {{4,3},{8,6},{12,10},{16,14},{20,18}};
        for (int[] p : pairs) {
            if (points[p[0]][1] < points[p[1]][1]) count++;
        }
        return count; // 0..5
    }

    /**
     * Реакция на смену жеста: вызываем системную команду из словаря alph.
     */
    private void onGestureChanged(int gesture) {
        Integer command = config.getGestureToCommand().get(gesture);
        if (command != null) {
            system.executeCommand(command);
        }
    }
}
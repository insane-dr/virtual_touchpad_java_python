package com.virtualtouchpad;

import java.awt.Toolkit;
import java.util.ArrayList;
import java.util.List;

public class GestureProcessor {

    private final SystemController system;
    private final ConfigLoader config;

    // История координат Y для распознавания свайпа (взмаха)
    private final List<Float> yHistory = new ArrayList<>();
    
    // Флаги состояний, чтобы не спамить командами каждый кадр
    private boolean isPinching = false;
    private boolean isFist = false;
    private long lastActionTime = 0;
    
    // Пороговые значения (тюнинг под вашу модель и камеру)
    private static final float PINCH_THRESHOLD = 0.08f; // Расстояние между пальцами для "щипка"
    private static final float SWIPE_THRESHOLD = 0.15f; // Насколько высоко должен быть взмах
    private static final int SWIPE_FRAMES = 8;          // За сколько кадров должен произойти взмах

    public GestureProcessor(SystemController system, ConfigLoader config) {
        this.system = system;
        this.config = config;
    }

    /**
     * Обновляет состояние на основе новых данных от нейросетей.
     */
    public void update(List<float[][]> hands) {
        if (hands == null || hands.isEmpty()) {
            resetStates();
            return;
        }

        float[][] pts = hands.get(0); // Берем первую руку
        
        // 1. Проверка на кулак (Скриншот)
        if (isFistGesture(pts)) {
            if (!isFist && canTriggerAction(1000)) {
                System.out.println("Жест: Кулак -> Скриншот");
                system.takeScreenshot("screenshot_" + System.currentTimeMillis() + ".png");
                isFist = true;
            }
            return; // Если кулак, другие жесты не проверяем
        } else {
            isFist = false;
        }

        // 2. Проверка на взмах вверх (Скрытие окон)
        if (detectSwipeUp(pts)) {
            if (canTriggerAction(1500)) {
                System.out.println("Жест: Взмах вверх -> Скрытие окон");
                system.hideAllWindows();
                yHistory.clear(); // Очищаем историю после срабатывания
            }
        }

        // 3. Проверка на щипок (Перемещение мышки / клик)
        if (isPinchGesture(pts)) {
            if (!isPinching) {
                // Зажимаем ЛКМ при начале щипка (опционально)
                system.mousePress();
                isPinching = true;
            }
            moveMouse(pts[8]); // Двигаем мышку за указательным пальцем
        } else {
            if (isPinching) {
                // Отпускаем ЛКМ при разжатии
                system.mouseRelease();
                isPinching = false;
            }
        }
    }

    // ================= ЛОГИКА ЖЕСТОВ =================

    /**
     * Жест "Кулак": все кончики пальцев ниже своих сгибов.
     * (В координатах картинки Y растет сверху вниз, поэтому кончик > сгиба)
     */
    private boolean isFistGesture(float[][] pts) {
        boolean indexFolded = pts[8][1] > pts[6][1];
        boolean middleFolded = pts[12][1] > pts[10][1];
        boolean ringFolded = pts[16][1] > pts[14][1];
        boolean pinkyFolded = pts[20][1] > pts[18][1];
        
        return indexFolded && middleFolded && ringFolded && pinkyFolded;
    }

    /**
     * Жест "Щипок": большой, указательный и средний пальцы сведены вместе.
     */
    private boolean isPinchGesture(float[][] pts) {
        float d1 = distance(pts[4], pts[8]);  // большой + указательный
        float d2 = distance(pts[8], pts[12]); // указательный + средний
        float d3 = distance(pts[4], pts[12]); // большой + средний
        
        return d1 < PINCH_THRESHOLD && d2 < PINCH_THRESHOLD && d3 < PINCH_THRESHOLD;
    }

    /**
     * Жест "Взмах вверх": отслеживаем резкое изменение Y координаты запястья (точка 0).
     */
    private boolean detectSwipeUp(float[][] pts) {
        float wristY = pts[0][1];
        yHistory.add(wristY);
        
        // Держим в памяти только последние N кадров
        if (yHistory.size() > SWIPE_FRAMES) {
            yHistory.remove(0);
        }

        if (yHistory.size() == SWIPE_FRAMES) {
            float oldestY = yHistory.get(0);
            float newestY = yHistory.get(yHistory.size() - 1);
            
            // Если Y уменьшился (рука пошла вверх), и разница больше порога
            // Примечание: если координаты нейросети 0..1, то порог = 0.15
            // Если координаты в пикселях, нужно ставить порог вроде 50.0f
            return (oldestY - newestY) > SWIPE_THRESHOLD;
        }
        return false;
    }

    // ================= ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ =================

    private void moveMouse(float[] indexTip) {
        // Предполагаем, что координаты нормализованы (от 0.0 до 1.0).
        // Если нейросеть выдает пиксели, нужно делить на W и H.
        int screenW = Toolkit.getDefaultToolkit().getScreenSize().width;
        int screenH = Toolkit.getDefaultToolkit().getScreenSize().height;

        // Инвертируем X, если управление кажется зеркальным
        int targetX = (int) (indexTip[0] * screenW);
        int targetY = (int) (indexTip[1] * screenH);

        system.mouseMove(targetX, targetY);
    }

    private float distance(float[] p1, float[] p2) {
        return (float) Math.hypot(p1[0] - p2[0], p1[1] - p2[1]);
    }

    private void resetStates() {
        if (isPinching) {
            system.mouseRelease();
            isPinching = false;
        }
        isFist = false;
        yHistory.clear();
    }

    /**
     * Защита от спама (кулдаун между вызовами команд).
     */
    private boolean canTriggerAction(long cooldownMs) {
        long now = System.currentTimeMillis();
        if (now - lastActionTime > cooldownMs) {
            lastActionTime = now;
            return true;
        }
        return false;
    }
}
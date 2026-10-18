package com.virtualtouchpad;

import java.awt.*;
import java.awt.event.InputEvent;
import java.awt.event.KeyEvent;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;

public class SystemController {

    private final Robot robot;

    public SystemController() throws AWTException {
        this.robot = new Robot();
        this.robot.setAutoDelay(50);
    }

    // ===== МЫШЬ =====
    public void mouseMove(int x, int y) {
        robot.mouseMove(x, y);
    }

    public void mousePress() {
        robot.mousePress(InputEvent.BUTTON1_DOWN_MASK);
    }

    public void mouseRelease() {
        robot.mouseRelease(InputEvent.BUTTON1_DOWN_MASK);
    }

    // ===== СКРИНШОТ =====
    public void takeScreenshot(String path) {
        try {
            Rectangle screen = new Rectangle(
                    Toolkit.getDefaultToolkit().getScreenSize());
            ImageIO.write(robot.createScreenCapture(screen), "png", new File(path));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // ===== СКРЫТИЕ ВСЕХ ОКОН =====
    public void hideAllWindows() {
        // Win+D (Windows) / Cmd+F3 (macOS) — зависит от ОС
        int mask = Toolkit.getDefaultToolkit().getMenuShortcutKeyMaskEx();
        robot.keyPress(KeyEvent.VK_D);
        robot.keyRelease(KeyEvent.VK_D);
        // На Windows это работает через Win+D
        // Для macOS/Linux логику нужно адаптировать
    }

    // ===== ЗАПУСК ПРИЛОЖЕНИЙ =====
    public void launchApplication(String command) {
        try {
            Runtime.getRuntime().exec(command);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // ===== ВЫПОЛНЕНИЕ КОМАНДЫ ПО НОМЕРУ =====
    public void executeCommand(int commandId) {
        switch (commandId) {
            case 2 -> takeScreenshot("screenshot_" + System.currentTimeMillis() + ".png");
            case 3 -> hideAllWindows();
            case 4 -> launchApplication("notepad.exe");      // Windows
            case 5 -> launchApplication("calc.exe");          // Windows
            case 6 -> launchApplication("explorer.exe");      // Windows
            // Добавьте свои команды
            default -> System.out.println("Неизвестная команда: " + commandId);
        }
    }
}
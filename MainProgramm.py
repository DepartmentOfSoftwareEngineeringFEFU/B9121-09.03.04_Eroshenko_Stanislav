import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from collections import Counter
from matplotlib import pyplot as plt
import datetime

loaded_image = None
selected_region = None
start_x, start_y = 0, 0
rect_id = None 
image_width, image_height = 0, 0
corrected_image = None 
excluded_region = None  
canvas_exclusion = None
exclusion_window = None

base_dir = os.path.dirname(os.path.abspath(__file__))

logs_dir = os.path.join(base_dir, 'logs')

log_dir = os.path.join(logs_dir, 'log1')
log2_dir = os.path.join(logs_dir, 'log2')
log3_dir = os.path.join(logs_dir, 'log3')
log4_dir = os.path.join(logs_dir, 'log4')

def clear_logs_directories():
    """Удаляет все файлы из папок log1, log2, log3, log4 при старте программы"""
    for folder in [log_dir, log2_dir, log3_dir, log4_dir]:
        if not os.path.exists(folder):
            os.makedirs(folder)
        else:
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path) 
                    elif os.path.isdir(file_path):
                        import shutil
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Ошибка при удалении {file_path}. Причина: {e}')

clear_logs_directories()

def show_error(message):
    """Функция для отображения ошибки"""
    messagebox.showerror("Ошибка", message)


def fix_breaks(cropped_image):
    """Функция исправления разрывов"""

    direction_window = tk.Toplevel(root)
    direction_window.title("Выбор направления сдвига")
    direction_var = tk.StringVar(value="down")
    
    directions = {
        "down": "Вниз",
        "right": "Вправо",
        "diag_right": "Диагональ вправо-вниз",
        "diag_left": "Диагональ влево-вниз",
        "up": "Вверх"
    }

    tk.Label(direction_window, text="Выберите направление сдвига:").pack()
    for val, text in directions.items():
        tk.Radiobutton(direction_window, text=text, variable=direction_var, value=val).pack(anchor=tk.W)

    tk.Button(direction_window, text="OK", command=direction_window.destroy).pack()
    direction_window.wait_window()

    direction = direction_var.get()

    shift_step = 0.1 
    
    cropped_image = np.array(cropped_image)
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    cv2.imshow('Threshold Image (After Binarization)', thresh)
    cv2.waitKey(0)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)) 
    morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    cv2.imshow('Morphed Image (After Morphological Operations)', morphed)
    cv2.waitKey(0)
    
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    all_contours_image = cropped_image.copy()
    cv2.drawContours(all_contours_image, contours, -1, (255, 0, 0), 1)
    cv2.imshow('All Contours (Before Filtering)', all_contours_image)
    cv2.waitKey(0)
    
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
    
    debug_image = cropped_image.copy()
    cv2.drawContours(debug_image, contours, -1, (0, 255, 0), 2)
    cv2.imshow('Top 2 Contours (After Filtering)', debug_image)
    cv2.waitKey(0)
    
    mask1 = np.zeros_like(gray)
    mask2 = np.zeros_like(gray)
    cv2.drawContours(mask1, [contours[0]], -1, 255, -1)
    cv2.drawContours(mask2, [contours[1]], -1, 255, -1)
    
    part1 = cv2.bitwise_and(cropped_image, cropped_image, mask=mask1)
    part2 = cv2.bitwise_and(cropped_image, cropped_image, mask=mask2)
    
    cv2.imshow('Part 1 (Extracted)', part1)
    cv2.waitKey(0)
    cv2.imshow('Part 2 (Extracted)', part2)
    cv2.waitKey(0)
    
    h, w = cropped_image.shape[:2]

    if part1.shape[2] == 4:
        part1 = cv2.cvtColor(part1, cv2.COLOR_BGRA2BGR)
    if part2.shape[2] == 4:
        part2 = cv2.cvtColor(part2, cv2.COLOR_BGRA2BGR)

    combined_image = np.zeros((h, w, 3), dtype=np.uint8)
    combined_image[:h, :w] = part1
    
    x_offset = 0
    y_offset = 0

    while True:
        temp_image = combined_image.copy()

        translation_matrix = np.float32([[1, 0, x_offset], [0, 1, y_offset]])
        shifted_part2 = cv2.warpAffine(part2, translation_matrix, (w, h))

        overlap = cv2.bitwise_and(temp_image, shifted_part2)
        if np.any(overlap):
            print(f"Пересечение достигнуто при сдвиге: x={x_offset}, y={y_offset}")
            break

        debug_image = cv2.addWeighted(temp_image, 0.7, shifted_part2, 0.3, 0)
        cv2.imshow('Manual Shifting (Debug)', debug_image)
        key = cv2.waitKey(10)

        if direction == "down":
            y_offset += shift_step
        if direction == "up":
            y_offset -= shift_step
        elif direction == "right":
            x_offset += shift_step
        elif direction == "diag_right":
            x_offset += shift_step
            y_offset += shift_step
        elif direction == "diag_left":
            x_offset -= shift_step
            y_offset += shift_step
    
    combined_image = cv2.bitwise_or(combined_image, shifted_part2)

    combined_image = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)
    cv2.imshow('Combined Image (Final Result)', combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    combined_image = cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR)

    combined_image = fix_bends_black(combined_image)

    return combined_image

def fix_bends_black(cropped_image):
    """Функция исправления изгибов"""
    cropped_image = np.array(cropped_image)

    lower_white = np.array([0, 0, 0], dtype = np.uint8)   
    upper_white = np.array([20, 20, 20], dtype = np.uint8)  

    cropped_image = cropped_image[:, :, :3] #Проверка на альфа канал, мб занести в отчет надо

    mask = cv2.inRange(cropped_image, lower_white, upper_white)

    # Печать координат белых пикселей, по хорошему можно убрать, но надо проверить что бы код не сломался
    white_pixel_coords = []
    for y in range(1, cropped_image.shape[0] - 1):  
        for x in range(1, cropped_image.shape[1] - 1):
            if mask[y, x]:  
                white_pixel_coords.append((x, y))

    # Дэбаг, пометка белых пикселей, тоже можно убрать, но надо проверить что бы код не сломался
    debug_image = cropped_image.copy()
    for coord in white_pixel_coords:
        x, y = coord
        debug_image[y, x] = [0, 0, 255]

    debug_image_bgr = cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR)

    # Дэбаг, показ отмеченых белых линий
    cv2.imshow("Debug Image with White Lines", debug_image_bgr)
    cv2.waitKey(0)  
    cv2.destroyAllWindows()  

    smoothed_image = cropped_image.copy()
    for y in range(1, cropped_image.shape[0] - 1): 
        for x in range(1, cropped_image.shape[1] - 1):
            if mask[y, x]: 
                neighbors = cropped_image[y-10:y+10, x-10:x+10] # Отвечает за то какие пиксели првоеряются для исправления, тоже можно поиграться, поменять значения
                smoothed_image[y, x] = np.median(neighbors, axis=(0, 1))

    debug_image_bgr = cv2.cvtColor(smoothed_image, cv2.COLOR_RGB2BGR)

    # Дэбаг, показ готового изображения
    cv2.imshow("Fixed Image", debug_image_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return smoothed_image

# from sklearn.linear_model import RANSACRegressor

# def fix_bends(cropped_image):
#     """Функция исправления изгибов с использованием RANSAC с расширенным дебагом"""
#     cropped_image = np.array(cropped_image)

#     lower_white = np.array([150, 150, 150], dtype=np.uint8)
#     upper_white = np.array([255, 255, 255], dtype=np.uint8)

#     cropped_image = cropped_image[:, :, :3]  # Убираем альфа-канал, если есть

#     # Этап 1: Маска белых областей
#     mask = cv2.inRange(cropped_image, lower_white, upper_white)
#     cv2.imshow("Step 1: White Mask", mask)
#     cv2.waitKey(0)

#     # Этап 2: Поиск координат белых пикселей
#     white_pixel_coords = np.column_stack(np.where(mask > 0))
#     white_pixel_coords = np.flip(white_pixel_coords, axis=1)  # (x, y)

#     if len(white_pixel_coords) == 0:
#         print("Нет белых пикселей для коррекции.")
#         return cropped_image

#     # Этап 3: Отметим найденные белые пиксели на копии изображения
#     debug_white_pixels = cropped_image.copy()
#     for (x, y) in white_pixel_coords:
#         cv2.circle(debug_white_pixels, (x, y), radius=1, color=(255, 0, 0), thickness=-1)  # Синие точки

#     debug_white_pixels_bgr = cv2.cvtColor(debug_white_pixels, cv2.COLOR_RGB2BGR)
#     cv2.imshow("Step 2: White Pixels Marked", debug_white_pixels_bgr)
#     cv2.waitKey(0)

#     # Этап 4: Применение RANSAC для аппроксимации линии
#     X = white_pixel_coords[:, 0].reshape(-1, 1)  # x
#     y = white_pixel_coords[:, 1]                 # y

#     model_ransac = RANSACRegressor()
#     model_ransac.fit(X, y)

#     line_x = np.linspace(0, cropped_image.shape[1] - 1, cropped_image.shape[1])
#     line_y = model_ransac.predict(line_x.reshape(-1, 1))

#     # Этап 5: Нарисовать предсказанную линию на изображении
#     debug_line_image = cropped_image.copy()
#     for (x, y_pred) in zip(line_x.astype(int), line_y.astype(int)):
#         if 0 <= x < debug_line_image.shape[1] and 0 <= int(y_pred) < debug_line_image.shape[0]:
#             debug_line_image[int(y_pred), x] = [0, 255, 0]  # Зеленая линия

#     debug_line_image_bgr = cv2.cvtColor(debug_line_image, cv2.COLOR_RGB2BGR)
#     cv2.imshow("Step 3: RANSAC Line", debug_line_image_bgr)
#     cv2.waitKey(0)

#     # Этап 6: Исправление изображения на основе найденной линии
#     corrected_image = cropped_image.copy()

#     for i, (x, y_pred) in enumerate(zip(line_x.astype(int), line_y.astype(int))):
#         if 0 <= x < corrected_image.shape[1]:
#             shift = y_pred - (corrected_image.shape[0] // 2)
#             shift = int(np.clip(shift, -10, 10))  # ограничиваем сдвиг
#             corrected_image[:, x] = np.roll(corrected_image[:, x], -shift, axis=0)

#     corrected_image_bgr = cv2.cvtColor(corrected_image, cv2.COLOR_RGB2BGR)

#     # Этап 7: Показ финального исправленного изображения
#     cv2.imshow("Step 4: Final Corrected Image", corrected_image_bgr)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     return corrected_image

def fix_bends(cropped_image):
    """Функция исправления изгибов"""
    cropped_image = np.array(cropped_image)

    lower_white = np.array([150, 150, 150], dtype = np.uint8)   
    upper_white = np.array([255, 255, 255], dtype = np.uint8)  

    cropped_image = cropped_image[:, :, :3] #Проверка на альфа канал, мб занести в отчет надо

    mask = cv2.inRange(cropped_image, lower_white, upper_white)

    # Печать координат белых пикселей, по хорошему можно убрать, но надо проверить что бы код не сломался
    white_pixel_coords = []
    for y in range(1, cropped_image.shape[0] - 1):  
        for x in range(1, cropped_image.shape[1] - 1):
            if mask[y, x]:  
                white_pixel_coords.append((x, y))

    # Дэбаг, пометка белых пикселей, тоже можно убрать, но надо проверить что бы код не сломался
    debug_image = cropped_image.copy()
    for coord in white_pixel_coords:
        x, y = coord
        debug_image[y, x] = [0, 0, 255]

    debug_image_bgr = cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR)

    # Дэбаг, показ отмеченых белых линий
    cv2.imshow("Debug Image with White Lines", debug_image_bgr)
    cv2.waitKey(0)  
    cv2.destroyAllWindows()  

    smoothed_image = cropped_image.copy()
    for y in range(1, cropped_image.shape[0] - 1): 
        for x in range(1, cropped_image.shape[1] - 1):
            if mask[y, x]: 
                neighbors = cropped_image[y-10:y+10, x-10:x+10] # Отвечает за то какие пиксели првоеряются для исправления, тоже можно поиграться, поменять значения
                smoothed_image[y, x] = np.median(neighbors, axis=(0, 1))

    debug_image_bgr = cv2.cvtColor(smoothed_image, cv2.COLOR_RGB2BGR)

    # Дэбаг, показ готового изображения
    cv2.imshow("Fixed Image", debug_image_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return smoothed_image

def fix_fragmentation(image):
    """ Убирает фрагментацию, заменяя пиксели, которые слишком сильно отличаются от их окружения """
    radius_kernel=100 
    threshold=75 
    radius = 10

    image = np.array(image)
    image = image[:, :, :3] #убирание альфа канала
    kernel_size = 2 * radius_kernel + 1
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    diff = np.abs(image.astype(np.int16) - blurred_image.astype(np.int16))
    noise_mask = np.any(diff > threshold, axis=2)
    debug_image = image.copy()
    debug_image = cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR)

    #Дэбаг, можно убрать
    debug_image[noise_mask] = [255, 0, 0]
    cv2.imshow("Debug Image with Noise", debug_image)

    denoised_image = image.copy()
    for y in range(radius, image.shape[0] - radius):
        for x in range(radius, image.shape[1] - radius):
            if noise_mask[y, x]:
                patch = image[y-radius:y+radius+1, x-radius:x+radius+1]
                reshaped_patch = patch.reshape(-1, 3)
                most_common_color = np.array(Counter(map(tuple, reshaped_patch)).most_common(1)[0][0], dtype=np.uint8)
                denoised_image[y, x] = most_common_color
                for ny, nx in [(y-1, x-1), (y-1, x), (y-1, x+1),
                               (y, x-1), (y, x+1),
                               (y+1, x-1), (y+1, x), (y+1, x+1)]:
                    if 0 <= ny < image.shape[0] and 0 <= nx < image.shape[1]:
                        denoised_image[ny, nx] = most_common_color
    
    #denoised_image = cv2.cvtColor(denoised_image, cv2.COLOR_RGB2BGR)

    #Дэбаг, можно убрать
    cv2.imshow("Corrected Image", denoised_image)

    return denoised_image



#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



def remove_stain2(cropped_image, radius=20):
    """
    Функция для удаления пятна на изображении путем замены пикселей внутри контура на средний цвет пикселей в радиусе.
    """
    #Почему то не получается заставить его удалить пиксели которые уже обработали из контура. Надо попробовать повторную обработку через интерфейс, мб добавить в код если получится
    
    #Можно попробовать найти средний цвет пятна, и перекрасить все пиксели рядом со средним цветом пятна в соседние пиксели, если они не в среднем цвете пятна
    # Добавить в коменты идею исправления расплывчатыэ пятен
    
    cropped_image = np.array(cropped_image)
    
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    
    _, binary_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)  # 50 - порог, можно поиграться, но в целом вроде норм
    
    contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    contour_mask = np.zeros_like(binary_mask)
    
    cv2.drawContours(contour_mask, contours, -1, (255), thickness=cv2.FILLED)

    inpainted_image = cropped_image.copy()

    for y in range(cropped_image.shape[0]):
        for x in range(cropped_image.shape[1]):
            if contour_mask[y, x] == 255:
                y_start = max(0, y - radius)
                y_end = min(cropped_image.shape[0], y + radius)
                x_start = max(0, x - radius)
                x_end = min(cropped_image.shape[1], x + radius)
                
                neighbors = []
                for i in range(y_start, y_end):
                    for j in range(x_start, x_end):
                        if contour_mask[i, j] == 0:
                            neighbors.append(cropped_image[i, j])

                if neighbors:
                    mean_color = np.mean(neighbors, axis=0)
                    inpainted_image[y, x] = mean_color
                    #print(f"Пиксель заменен на: {mean_color}")
    
    #Дэбаг показ результата
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Исходное изображение")
    plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Изображение с заменой пикселей на средний цвет")
    plt.imshow(cv2.cvtColor(inpainted_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    return inpainted_image



def remove_stain3(cropped_image, radius=20):
    """
    Функция для удаления пятна на изображении с использованием метода inpainting
    """
    # Честно, результат плохой, но лучше ничего сделать с inpainting не смог. Скорее всего удалю всю функцию полностью, но пока пусть повисит, мало ли
    cropped_image = np.array(cropped_image)
    
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    
    _, binary_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)  # 50 - порог, можно поиграться, но в целом вроде норм
    
    contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    stain_mask = np.zeros_like(binary_mask)
    cv2.drawContours(stain_mask, contours, -1, (255), thickness=cv2.FILLED)
    
    contours_image = cropped_image.copy()
    cv2.drawContours(contours_image, contours, -1, (0, 255, 0), 2)
    inpainted_image = cv2.inpaint(cropped_image, stain_mask, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)

    # Дэбаг показ результата
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.title("Исходное изображение")
    plt.imshow(cropped_image)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Изображение с контурами пятна")
    plt.imshow(cv2.cvtColor(contours_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Изображение после исправления пятна")
    plt.imshow(inpainted_image)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    return inpainted_image




def remove_stain4(cropped_image, radius=20, patch_size=15, search_step=5):
    """
    Удаление пятен на изображении методом прецедентных патчей (без ИИ).
    Пятно заменяется на наиболее подходящий патч из окружающей области.
    """

    image = np.array(cropped_image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Детектим пятно
    _, stain_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    stain_mask = cv2.morphologyEx(stain_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    result = image.copy()

    h, w = gray.shape

    for y in range(0, h - patch_size, search_step):
        for x in range(0, w - patch_size, search_step):
            patch_mask = stain_mask[y:y+patch_size, x:x+patch_size]

            # Если патч попадает на пятно, ищем замену
            if np.mean(patch_mask) > 10:
                best_patch = None
                best_diff = float('inf')

                # Поиск подходящего патча вне пятна
                for yy in range(0, h - patch_size, search_step):
                    for xx in range(0, w - patch_size, search_step):
                        if np.mean(stain_mask[yy:yy+patch_size, xx:xx+patch_size]) < 100:
                            candidate = image[yy:yy+patch_size, xx:xx+patch_size]
                            target = image[y:y+patch_size, x:x+patch_size]

                            diff = np.sum((candidate.astype(float) - target.astype(float))**2)
                            if diff < best_diff:
                                best_diff = diff
                                best_patch = candidate.copy()

                if best_patch is not None:
                    result[y:y+patch_size, x:x+patch_size] = best_patch

    # Визуализация
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.title("Исходное изображение")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Маска пятна")
    plt.imshow(stain_mask, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("После замены патчей")
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    return result


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------




#Второе выделение области, может стоит сделать это через первоначальное выделение области, но пусть пока будет так, если руки дойдут подправлю
def start_exclusive_selection(event):
    """Начало выделения области исключения"""
    global start_x, start_y, rect_id
    start_x, start_y = event.x, event.y
    if rect_id:
        canvas_exclusion.delete(rect_id)

def update_exclusive_selection(event):
    """Обновление выделяемой области исключения"""
    global rect_id
    canvas_exclusion.delete(rect_id)
    rect_id = canvas_exclusion.create_rectangle(start_x, start_y, event.x, event.y, outline="blue", width=2)

def finish_exclusive_selection(event):
    """Сохранение выделенной области"""
    global excluded_region, exclusion_window
    excluded_region = (start_x, start_y, event.x, event.y)
    messagebox.showinfo("Область выбрана", "Выделенная область не будет исправляться")
    exclusion_window.destroy()


def remove_stain_hands(cropped_image):
    """функция ручного удаления пятен"""
    print("it worked")

def fix_fragmentation_hands(image):
    """ Убирает фрагментацию, давая пользователю возможность выделить область, которую не нужно исправлять """
    global excluded_region, canvas_exclusion, exclusion_window, rect_id

    radius_kernel = 100
    threshold = 75
    radius = 10

    image = np.array(image)
    image = image[:, :, :3]
    kernel_size = 2 * radius_kernel + 1
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    diff = np.abs(image.astype(np.int16) - blurred_image.astype(np.int16))
    noise_mask = np.any(diff > threshold, axis=2)

    debug_image = image.copy()
    debug_image = cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR)
    debug_image[noise_mask] = [255, 0, 0]

    exclusion_window = tk.Toplevel(root)
    exclusion_window.title("Выделите область исключения")

    image_tk = ImageTk.PhotoImage(Image.fromarray(debug_image))
    canvas_exclusion = tk.Canvas(exclusion_window, width=debug_image.shape[1], height=debug_image.shape[0])
    canvas_exclusion.pack()
    canvas_exclusion.create_image(0, 0, anchor="nw", image=image_tk)

    rect_id = None
    canvas_exclusion.bind("<ButtonPress-1>", start_exclusive_selection)
    canvas_exclusion.bind("<B1-Motion>", update_exclusive_selection)
    canvas_exclusion.bind("<ButtonRelease-1>", finish_exclusive_selection)

    exclusion_window.wait_window()

    denoised_image = image.copy()

    for y in range(radius, image.shape[0] - radius):
        for x in range(radius, image.shape[1] - radius):
            if excluded_region:
                x1, y1, x2, y2 = excluded_region
                if x1 <= x <= x2 and y1 <= y <= y2:
                    continue

            if noise_mask[y, x]:
                patch = image[y - radius:y + radius + 1, x - radius:x + radius + 1]
                reshaped_patch = patch.reshape(-1, 3)
                most_common_color = np.array(Counter(map(tuple, reshaped_patch)).most_common(1)[0][0], dtype=np.uint8)
                denoised_image[y, x] = most_common_color
                for ny, nx in [(y-1, x-1), (y-1, x), (y-1, x+1),
                               (y, x-1), (y, x+1),
                               (y+1, x-1), (y+1, x), (y+1, x+1)]:
                    if 0 <= ny < image.shape[0] and 0 <= nx < image.shape[1]:
                        denoised_image[ny, nx] = most_common_color
    
    cv2.imshow("Corrected Image", denoised_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return denoised_image

def fix_bends_hands(cropped_image):
    """Функция исправления изгибов с возможностью исключения области"""
    global excluded_region, canvas_exclusion, exclusion_window, rect_id

    cropped_image = np.array(cropped_image)
    lower_white = np.array([75, 75, 75], dtype=np.uint8)   
    upper_white = np.array([255, 255, 255], dtype=np.uint8)  

    cropped_image = cropped_image[:, :, :3]

    mask = cv2.inRange(cropped_image, lower_white, upper_white)

    white_pixel_coords = [(x, y) for y in range(cropped_image.shape[0]) for x in range(cropped_image.shape[1]) if mask[y, x]]

    debug_image = cropped_image.copy()
    for x, y in white_pixel_coords:
        debug_image[y, x] = [0, 0, 255]

    debug_image_bgr = cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR)

    exclusion_window = tk.Toplevel(root)
    exclusion_window.title("Выделите область исключения")

    image_tk = ImageTk.PhotoImage(Image.fromarray(debug_image_bgr))

    canvas_exclusion = tk.Canvas(exclusion_window, width=debug_image.shape[1], height=debug_image.shape[0])
    canvas_exclusion.pack()
    canvas_exclusion.create_image(0, 0, anchor="nw", image=image_tk)

    rect_id = None

    canvas_exclusion.bind("<ButtonPress-1>", start_exclusive_selection)
    canvas_exclusion.bind("<B1-Motion>", update_exclusive_selection)
    canvas_exclusion.bind("<ButtonRelease-1>", finish_exclusive_selection)

    exclusion_window.wait_window() 

    if excluded_region:
        x1, y1 = [excluded_region[0], excluded_region[1]]
        x2, y2 = [excluded_region[2], excluded_region[3]]

        debug_image_with_exclusion = debug_image.copy()
        cv2.rectangle(debug_image_with_exclusion, (x1, y1), (x2, y2), (0, 255, 0), 2)
        debug_image_bgr_with_exclusion = cv2.cvtColor(debug_image_with_exclusion, cv2.COLOR_RGB2BGR)

        plt.imshow(debug_image_bgr_with_exclusion)
        plt.title("Область")
        plt.axis('off')
        plt.show()

        cv2.imshow("Exclusion Area", debug_image_bgr_with_exclusion)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    smoothed_image = cropped_image.copy()
    for y in range(1, cropped_image.shape[0] - 1): 
        for x in range(1, cropped_image.shape[1] - 1):
            if excluded_region and x1 <= x <= x2 and y1 <= y <= y2:
                continue

            if mask[y, x]: 
                neighbors = cropped_image[max(0, y-20):min(cropped_image.shape[0], y+20),
                                          max(0, x-20):min(cropped_image.shape[1], x+20)]
                smoothed_image[y, x] = np.median(neighbors, axis=(0, 1))

    debug_image_bgr = cv2.cvtColor(smoothed_image, cv2.COLOR_RGB2BGR)
    cv2.imshow("Fixed Image", debug_image_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return smoothed_image


def fix_breaks_hands(cropped_image):
    """Функция исправления разрывов с возможностью ручной настройки параметров"""

    param_window = tk.Toplevel(root)
    param_window.title("Настройки исправления разрывов")

    tk.Label(param_window, text="Размер ядра размытия:").grid(row=0, column=0)
    blur_kernel_var = tk.IntVar(value=5)
    blur_kernel_entry = tk.Entry(param_window, textvariable=blur_kernel_var)
    blur_kernel_entry.grid(row=0, column=1)
    tk.Label(param_window, text="Чем больше значение, тем сильнее сглаживаются детали. Среднее: 5-7.").grid(row=1, column=0, columnspan=2)

    tk.Label(param_window, text="Порог: blockSize").grid(row=2, column=0)
    thresh_block_size_var = tk.IntVar(value=11)
    thresh_block_size_entry = tk.Entry(param_window, textvariable=thresh_block_size_var)
    thresh_block_size_entry.grid(row=2, column=1)
    tk.Label(param_window, text="Размер блока для вычисления порога. Больше = более плавный порог. Среднее: 11-15.").grid(row=3, column=0, columnspan=2)

    tk.Label(param_window, text="Порог: C").grid(row=4, column=0)
    thresh_C_var = tk.IntVar(value=2)
    thresh_C_entry = tk.Entry(param_window, textvariable=thresh_C_var)
    thresh_C_entry.grid(row=4, column=1)
    tk.Label(param_window, text="Чем выше C, тем больше деталей остается. Среднее: 2-5.").grid(row=5, column=0, columnspan=2)

    tk.Label(param_window, text="Размер ядра морфологии:").grid(row=6, column=0)
    morph_kernel_size_var = tk.IntVar(value=5)
    morph_kernel_size_entry = tk.Entry(param_window, textvariable=morph_kernel_size_var)
    morph_kernel_size_entry.grid(row=6, column=1)
    tk.Label(param_window, text="Определяет, какие области будут заполнены. Среднее: 3-5.").grid(row=7, column=0, columnspan=2)

    tk.Label(param_window, text="Количество итераций морфологии:").grid(row=8, column=0)
    morph_iterations_var = tk.IntVar(value=2)
    morph_iterations_entry = tk.Entry(param_window, textvariable=morph_iterations_var)
    morph_iterations_entry.grid(row=8, column=1)
    tk.Label(param_window, text="Чем больше итераций, тем сильнее заполняются разрывы. Среднее: 1-3.").grid(row=9, column=0, columnspan=2)

    tk.Label(param_window, text="Шаг сдвига:").grid(row=10, column=0)
    shift_step_var = tk.DoubleVar(value=0.1)
    shift_step_entry = tk.Entry(param_window, textvariable=shift_step_var)
    shift_step_entry.grid(row=10, column=1)
    tk.Label(param_window, text="Определяет скорость поиска совпадений. Среднее: 0.1-0.3.").grid(row=11, column=0, columnspan=2)

    def apply_params():
        param_window.destroy()

    tk.Button(param_window, text="Применить", command=apply_params).grid(row=12, column=0, columnspan=2)
    param_window.wait_window()

    blur_kernel = blur_kernel_var.get()
    thresh_block_size = thresh_block_size_var.get()
    thresh_C = thresh_C_var.get()
    morph_kernel_size = morph_kernel_size_var.get()
    morph_iterations = morph_iterations_var.get()
    shift_step = shift_step_var.get()

    direction_window = tk.Toplevel(root)
    direction_window.title("Выбор направления сдвига")
    direction_var = tk.StringVar(value="down")

    directions = {
        "down": "Вниз",
        "right": "Вправо",
        "diag_right": "Диагональ вправо-вниз",
        "diag_left": "Диагональ влево-вниз"
    }

    tk.Label(direction_window, text="Выберите направление сдвига:").pack()
    for val, text in directions.items():
        tk.Radiobutton(direction_window, text=text, variable=direction_var, value=val).pack(anchor=tk.W)

    tk.Button(direction_window, text="OK", command=direction_window.destroy).pack()
    direction_window.wait_window()

    direction = direction_var.get()

    cropped_image = np.array(cropped_image)
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, thresh_block_size, thresh_C
    )
    cv2.imshow('Threshold Image (After Binarization)', thresh)
    cv2.waitKey(0)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size))
    morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations)
    cv2.imshow('Morphed Image (After Morphological Operations)', morphed)
    cv2.waitKey(0)

    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    all_contours_image = cropped_image.copy()
    cv2.drawContours(all_contours_image, contours, -1, (255, 0, 0), 1)
    cv2.imshow('All Contours (Before Filtering)', all_contours_image)
    cv2.waitKey(0)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

    debug_image = cropped_image.copy()
    cv2.drawContours(debug_image, contours, -1, (0, 255, 0), 2)
    cv2.imshow('Top 2 Contours (After Filtering)', debug_image)
    cv2.waitKey(0)

    mask1 = np.zeros_like(gray)
    mask2 = np.zeros_like(gray)
    cv2.drawContours(mask1, [contours[0]], -1, 255, -1)
    cv2.drawContours(mask2, [contours[1]], -1, 255, -1)

    part1 = cv2.bitwise_and(cropped_image, cropped_image, mask=mask1)
    part2 = cv2.bitwise_and(cropped_image, cropped_image, mask=mask2)

    cv2.imshow('Part 1 (Extracted)', part1)
    cv2.waitKey(0)
    cv2.imshow('Part 2 (Extracted)', part2)
    cv2.waitKey(0)

    h, w = cropped_image.shape[:2]

    if part1.shape[2] == 4:
        part1 = cv2.cvtColor(part1, cv2.COLOR_BGRA2BGR)
    if part2.shape[2] == 4:
        part2 = cv2.cvtColor(part2, cv2.COLOR_BGRA2BGR)

    combined_image = np.zeros((h, w, 3), dtype=np.uint8)
    combined_image[:h, :w] = part1

    x_offset = 0
    y_offset = 0

    while True:
        temp_image = combined_image.copy()

        translation_matrix = np.float32([[1, 0, x_offset], [0, 1, y_offset]])
        shifted_part2 = cv2.warpAffine(part2, translation_matrix, (w, h))

        overlap = cv2.bitwise_and(temp_image, shifted_part2)
        if np.any(overlap):
            print(f"Пересечение достигнуто при сдвиге: x={x_offset}, y={y_offset}")
            break

        debug_image = cv2.addWeighted(temp_image, 0.7, shifted_part2, 0.3, 0)
        cv2.imshow('Manual Shifting (Debug)', debug_image)
        key = cv2.waitKey(10)

        if direction == "down":
            y_offset += shift_step
        elif direction == "right":
            x_offset += shift_step
        elif direction == "diag_right":
            x_offset += shift_step
            y_offset += shift_step
        elif direction == "diag_left":
            x_offset -= shift_step
            y_offset += shift_step

    combined_image = cv2.bitwise_or(combined_image, shifted_part2)

    combined_image = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)

    cv2.imshow('Combined Image (Final Result)', combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    combined_image = cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR)

    return combined_image




def clear_image():
    """Функция для очистки данных изображения"""
    global loaded_image, image_width, image_height
    loaded_image = None
    image_width, image_height = 0, 0
    canvas_original.delete("all")

def validate_image(file_path):
    """Функция проверки изображения"""
    try:
        img = Image.open(file_path)
        width, height = img.size
        file_size = os.path.getsize(file_path) / (1024 * 1024)
        dpi = img.info.get("dpi", (0, 0))[0]

        if dpi > 300:
            show_error("Ошибка: разрешение изображения должно быть не более 300 DPI.")
            return False

        if width > 1205 or height > 1795:
            show_error("Ошибка: изображение должно быть не больше 1205x1795 пикселей.")
            return False

        if file_size > 10:
            show_error("Ошибка: размер файла не должен превышать 10 МБ.")
            return False

        return True

    except Exception as e:
        show_error(f"Ошибка при проверке изображения: {e}")
        return False

def load_image():
    """Функция загрузки изображения с проверками"""
    #при загрузке нового изображения после выделения области в старом возникает такая проблема, что область как бы не выделена, но так как она была выделена
    #в старом изображении, используются данные об старой выделеной области, и это как бы ошибка. Исправить не сложно, но мне щас не до этого
    global loaded_image, image_width, image_height, selected_region  
    selected_region = None

    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    
    if file_path:
        if not validate_image(file_path):
            clear_image()
            return

        loaded_image = Image.open(file_path)
        image_width, image_height = loaded_image.size
        img_resized = loaded_image.resize((300, 300))
        img_tk = ImageTk.PhotoImage(img_resized)

        canvas_original.image = img_tk
        canvas_original.create_image(0, 0, anchor="nw", image=img_tk)

        save_path = os.path.join(log_dir, "Original_Image.png")

        if os.path.exists(save_path):
            os.remove(save_path)

        loaded_image.save(save_path, format="PNG")


#Ошибка с выделением за пределом изображения осталась, но на результат она не влияет


def start_selection(event):
    """Начало выделения области"""
    global start_x, start_y, rect_id
    start_x, start_y = event.x, event.y
    if rect_id:
        canvas_original.delete(rect_id)

def update_selection(event):
    """Обновление выделяемой области"""
    global rect_id
    canvas_original.delete(rect_id)
    rect_id = canvas_original.create_rectangle(start_x, start_y, event.x, event.y, outline="red", width=2)

def finish_selection(event):
    """Фиксация выделенной области"""
    global selected_region
    selected_region = (start_x, start_y, event.x, event.y)

def save_image_info(file_path):
    """Сохраняет информацию об изображении в log2"""
    try:
        img = Image.open(file_path)
        width, height = img.size
        file_size = os.path.getsize(file_path) / (1024 * 1024)
        dpi = img.info.get("dpi", (0, 0))[0]
        format = img.format

        info_file = os.path.join(log2_dir, "Image_Info.txt")

        if os.path.exists(info_file):
            os.remove(info_file)

        with open(info_file, "w") as f:
            f.write(f"Формат изображения: {format}\n")
            f.write(f"Размер изображения: {width}x{height} пикселей\n")
            f.write(f"Разрешение (DPI): {dpi}\n")
            f.write(f"Размер файла: {file_size:.2f} МБ\n")
            f.write(f"Цветовой режим: {img.mode}\n")

    except Exception as e:
        show_error(f"Ошибка при сохранении информации: {e}")


def process_image():
    """Функция обработки изображения"""
    global corrected_image
    selected_fix = selected_option.get()
    selected_mode_chosen = selected_mode.get()
    if loaded_image is None:
        show_error('Не загружено изображение')
    else:
        if selected_fix != "option1" and not selected_region:
            show_error('Выделение области не выполнено!')
            return

        if selected_region or selected_fix == "option1": 
            if selected_fix != "option1":
                x1, y1, x2, y2 = selected_region
                scale_x = image_width / 300
                scale_y = image_height / 300

                x1_original = int(x1 * scale_x)
                y1_original = int(y1 * scale_y)
                x2_original = int(x2 * scale_x)
                y2_original = int(y2 * scale_y)

                cropped_image = loaded_image.crop((x1_original, y1_original, x2_original, y2_original))
                cropped_image_path = os.path.join(log_dir, "cropped_image.png")
                cropped_image.save(cropped_image_path)

                log_file_path = os.path.join(log_dir, "Defect.txt")
                with open(log_file_path, 'w') as log_file:
                    log_file.write(selected_option.get())

                save_image_info(os.path.join(log_dir, "Original_Image.png"))

                
                if selected_mode_chosen == "option6":
                    if selected_fix == "option1":
                        corrected_image = fix_breaks(loaded_image)
                    elif selected_fix == "option2":
                        corrected_image = fix_bends(cropped_image)
                    elif selected_fix == "option3":
                        corrected_image = fix_fragmentation(cropped_image)
                    elif selected_fix == "option4":
                        corrected_image = remove_stain(cropped_image)
                    else:
                        corrected_image = cropped_image
                elif selected_mode_chosen == "option5":
                    if selected_fix == "option1":
                        corrected_image = fix_breaks_hands(loaded_image)
                    elif selected_fix == "option2":
                        corrected_image = fix_bends_hands(cropped_image)
                    elif selected_fix == "option3":
                        corrected_image = fix_fragmentation_hands(cropped_image)
                    elif selected_fix == "option4":
                        corrected_image = remove_stain_hands(cropped_image)
                    else:
                        corrected_image = cropped_image

                corrected_image = Image.fromarray(corrected_image.astype('uint8'))

                if selected_fix != "option1":
                    updated_image = loaded_image.copy()
                    updated_image.paste(corrected_image, (x1_original, y1_original, x2_original, y2_original))
                else:
                    updated_image = corrected_image.copy()
                updated_image_tk = ImageTk.PhotoImage(updated_image.resize((300, 300)))
                canvas_fixed.image = updated_image_tk
                canvas_fixed.create_image(0, 0, anchor="nw", image=updated_image_tk)
                corrected_image_path = os.path.join(log3_dir, "fixed_image.png")
                corrected_image.save(corrected_image_path)
                save_process_report(selected_fix, selected_mode_chosen)
                if selected_fix != "option1":
                    corrected_image = updated_image
            else:
                if selected_mode_chosen == "option6":
                    if selected_fix == "option1":
                        corrected_image = fix_breaks(loaded_image)
                elif selected_mode_chosen == "option5":
                    if selected_fix == "option1":
                        corrected_image = fix_breaks_hands(loaded_image)

                corrected_image = Image.fromarray(corrected_image.astype('uint8'))
                updated_image = corrected_image.copy()
                updated_image_tk = ImageTk.PhotoImage(updated_image.resize((300, 300)))
                canvas_fixed.image = updated_image_tk
                canvas_fixed.create_image(0, 0, anchor="nw", image=updated_image_tk)
                corrected_image_path = os.path.join(log3_dir, "fixed_image.png")
                corrected_image.save(corrected_image_path)
                save_process_report(selected_fix, selected_mode_chosen)

        else:
            show_error('Выделение области не выполнено!')

def reprocess_image():
    """Функция для повторной обработки изображения"""
    global loaded_image, image_width, image_height, corrected_image

    if 'corrected_image' not in globals() or corrected_image is None:
        show_error("Нет исправленного изображения для повторной обработки")
        return

    loaded_image = corrected_image
    image_width, image_height = loaded_image.size

    img_resized = loaded_image.resize((300, 300))
    img_tk = ImageTk.PhotoImage(img_resized)

    canvas_original.image = img_tk
    canvas_original.create_image(0, 0, anchor="nw", image=img_tk)


def save_process_report(defect, mode, params=None):
    """Сохраняет отчет о восстановлении изображения в log2"""
    if not os.path.exists(log4_dir):
        os.makedirs(log4_dir)

    report_path = os.path.join(log4_dir, "Report.txt")

    now = datetime.datetime.now()
    defect_names = {
        "option1": "Разрывы",
        "option2": "Изгибы",
        "option3": "Фрагментация",
        "option4": "Загрязнения"
    }
    mode_names = {
        "option5": "Ручная обработка",
        "option6": "Автоматическая обработка"
    }

    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write(f"Отчет об обработке изображения\n")
        report_file.write(f"Дата и время обработки: {now.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        report_file.write(f"Выбранный дефект для исправления: {defect_names.get(defect, 'Неизвестно')}\n")
        report_file.write(f"Режим обработки: {mode_names.get(mode, 'Неизвестно')}\n\n")
        
        report_file.write(f"Этапы обработки:\n")
        if defect == "option1":
            report_file.write(f"- Исправление разрывов с помощью функции fix_breaks.\n")
            report_file.write(f"  Параметры: стандартные или пользовательские (если ручной режим).\n")
        elif defect == "option2":
            report_file.write(f"- Исправление изгибов с помощью функции fix_bends.\n")
            report_file.write(f"  Обнаружение белых линий и их сглаживание медианным фильтром.\n")
        elif defect == "option3":
            report_file.write(f"- Исправление фрагментации с помощью функции fix_fragmentation.\n")
            report_file.write(f"  Обнаружение шумов и замена их на наиболее распространенные цвета в окрестности.\n")
        elif defect == "option4":
            report_file.write(f"- Удаление загрязнений с помощью функции remove_stain.\n")
            report_file.write(f"  Замена участков пятна наиболее похожими патчами из изображения.\n")
        
        if params:
            report_file.write("\nИспользованные параметры обработки:\n")
            for key, value in params.items():
                report_file.write(f"- {key}: {value}\n")

def save_corrected_image():
    """Сохраняет исправленное изображение в выбранную пользователем директорию"""
    global corrected_image
    if corrected_image is None:
        show_error("Нет исправленного изображения для сохранения")
        return

    file_path = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg"), ("All Files", "*.*")],
        title="Сохранить исправленное изображение"
    )

    if file_path:
        try:
            corrected_image.save(file_path)
            messagebox.showinfo("Сохранение", f"Изображение успешно сохранено в:\n{file_path}")
        except Exception as e:
            show_error(f"Ошибка при сохранении: {e}")

root = tk.Tk()
root.title("Обработка изображений")
root.configure(bg="grey")
root.geometry("700x500")
root.resizable(False, False)

frame_images = tk.Frame(root)
frame_images.grid(row=0, column=0, columnspan=4, pady=5, padx=10)

label_original = tk.Label(frame_images, text="Изначальное изображение", font=("Arial", 10))
label_original.grid(row=0, column=0, padx=10, pady=2)

label_fixed = tk.Label(frame_images, text="Исправленное изображение", font=("Arial", 10))
label_fixed.grid(row=0, column=1, padx=10, pady=2)

canvas_original = tk.Canvas(frame_images, width=300, height=300, bg="grey")
canvas_original.grid(row=1, column=0, padx=10, pady=5)

canvas_fixed = tk.Canvas(frame_images, width=300, height=300, bg="grey")
canvas_fixed.grid(row=1, column=1, padx=10, pady=5)

frame_controls = tk.Frame(root)
frame_controls.grid(row=1, column=0, columnspan=4, pady=10)

selected_option = tk.StringVar(value="option1")
radio1 = tk.Radiobutton(frame_controls, text="Разрывы", variable=selected_option, value="option1")
radio2 = tk.Radiobutton(frame_controls, text="Изгибы", variable=selected_option, value="option2")
radio3 = tk.Radiobutton(frame_controls, text="Фрагментация", variable=selected_option, value="option3")

selected_mode = tk.StringVar(value="option5")
radio5 = tk.Radiobutton(frame_controls, text="Ручная обработка", variable=selected_mode, value="option5")
radio6 = tk.Radiobutton(frame_controls, text="Автоматическая обработка", variable=selected_mode, value="option6")


btn_load = tk.Button(frame_controls, text="Загрузить", font=("Arial", 10), command=load_image)
btn_process = tk.Button(frame_controls, text="Обработать", font=("Arial", 10), command=process_image)
btn_reprocess = tk.Button(frame_controls, text="Повторная обработка", font=("Arial", 10), command=reprocess_image)
btn_save = tk.Button(frame_controls, text="Сохранить", font=("Arial", 10), command=save_corrected_image)

btn_reprocess.grid(row=2, column=0, columnspan=3, pady=5) 
btn_load.grid(row=0, column=0, padx=5)
btn_process.grid(row=0, column=1, padx=5)
btn_save.grid(row=0, column=2, padx=5)

radio1.grid(row=0, column=3, padx=5)
radio2.grid(row=0, column=4, padx=5)
radio3.grid(row=0, column=5, padx=5)
radio5.grid(row=1, column=3, padx=5)
radio6.grid(row=1, column=4, padx=5)

canvas_original.bind("<ButtonPress-1>", start_selection)
canvas_original.bind("<B1-Motion>", update_selection)
canvas_original.bind("<ButtonRelease-1>", finish_selection)

root.mainloop()
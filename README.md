# Фильтр изображений не содержащих автомобилей

Скрипт основан на предобученной на ImagNet модели resnet50

Для фильтрации вы выбраны следующие классы:
        "convertible", "sports car", "minivan", "pickup truck",
        "ambulance", "fire engine", "cab", "jeep", "limousine",
        "car", "truck", "bus", "automobile", "taxi", "estate car",
        "car wheel", "passenger car", "race car", "racing car",
        "sports car", "sport car", "fire truck", "police van",
        "police wagon".

### ❗ Скрипт требует наличие подключения к сети интернет ❗

---

## 📄 Описание

- Скрипт работает с файлами следующих форматов: '.png', '.jpg', '.jpeg'
- Получает на вход папку с изображениями, которые необходимо отфильтровать
- На выходе создает файл и папку с отфильтрованными изображениями.

---

## 🚀 Запуск скрипта

```
python checking.py
```
---

### 📜 Аргументы скрипта

```
python checking.py --input_folder <path_to_folder> --output_file <path_to_file> --no_cars_folder <path_to_folder>
```

- input_folder Путь к папке с изображениями, которые нужно отфильтровать (нет значения по умолчанию)
- output_file Путь к файлу, в который будут сохраняться названия отфильтрованных изображений (по умолчанию создается файл ```checking_result.txt``` в папке со скриптом)
- no_cars_folder Путь к папке, в которую будут перемещены отфильтрованные изображения (по умолчанию создает папку ```no_cars``` в папке со скриптом)

---

## 🐋 Docker 

Для билда докер образа в дериктории с репозиторием введите команду:
```
docker build -t car-filtering .
```
Для запуска докер образа выполните команду:
```
docker run -it --rm -v <path_to_your_folder>:/app/images -v <path_to_your_folder>:/app/no_cars -v <path_to_your_check.txt>:/app/check.txt car-filtering python checking.py --input_folder /app/images --output_file /app/check.txt --no_cars_folder /app/no_cars
```
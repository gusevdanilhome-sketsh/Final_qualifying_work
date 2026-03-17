# Используем официальный образ Python
FROM python:3.9-alpine

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Копируем файл с зависимостями и устанавливаем их
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь проект
COPY . .

# Создаём необходимые папки (если их нет)
RUN mkdir -p data models reports

# Указываем команду по умолчанию для запуска CLI
CMD ["python", "cli.py"]
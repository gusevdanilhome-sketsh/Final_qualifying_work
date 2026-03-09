# Используем официальный образ Python 3.9 (slim-версия для меньшего размера)
FROM python:3.10-slim

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Копируем файл с зависимостями
COPY requirements.txt .

# Устанавливаем основные зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Дополнительно устанавливаем colorama (используется в cli.py)
RUN pip install --no-cache-dir colorama

# Копируем весь проект в контейнер
COPY . .

# Указываем, что контейнер будет запускать интерактивную программу (cli.py)
# Можно переопределить при запуске
CMD ["python", "cli.py"]
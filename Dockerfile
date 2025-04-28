# Use the official Python 3.9.21 slim image
FROM python:3.9.21-slim

# Don’t write .pyc files; unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working dir
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy your application code (including task1.py, task2.py, task4.py)
COPY . .

# On container start, run the three tasks in order
CMD ["sh", "-c", "python task1.py && python task2.py && python task4.py"]

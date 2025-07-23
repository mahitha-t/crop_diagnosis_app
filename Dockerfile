FROM python:3.10

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

COPY requirement.txt .

RUN apt-get update && apt-get install -y \
    libjpeg-dev \
    zlib1g-dev \
 && pip install --no-cache-dir -r requirements.txt \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
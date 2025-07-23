FROM python:3.10

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

COPY requirements.txt .

RUN apt-get update
RUN apt-get install -y libjpeg-dev zlib1g-dev
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
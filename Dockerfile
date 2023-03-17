FROM python:3.9-slim-buster
WORKDIR /app
RUN apt-get update && install ffmpeg libsm6 libxext6 libgl1 -y
COPY . .
RUN pip install -r requirements.txt
EXPOSE 5000
ENV FLASK_APP=app.py
CMD ["flask", "run", "--host", "0.0.0.0"]
FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train.py predict.py ./

# train inside the image, so the model always matches the installed library versions
RUN python train.py

EXPOSE 9696
CMD ["gunicorn", "--bind", "0.0.0.0:9696", "predict:app"]

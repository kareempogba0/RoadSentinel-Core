FROM tensorflow/tensorflow:2.10.0-gpu-jupyter

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install protobuf==3.20.3 --no-cache-dir --force-reinstall

COPY . /app
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
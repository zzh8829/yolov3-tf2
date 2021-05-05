FROM tensorflow/tensorflow:2.4.0-gpu
COPY requirements-gpu.txt requirements-gpu.txt
RUN pip3 install -r requirements-gpu.txt
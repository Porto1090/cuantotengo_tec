# Imagen base mínima de inferencia de Azure ML (Python 3.12, CPU, incluye el servidor de inferencia)
FROM mcr.microsoft.com/azureml/minimal-py312-inference:latest

# Ultralytics guarda su configuración en el home; el contenedor corre sin root
ENV YOLO_CONFIG_DIR=/tmp/Ultralytics

COPY requirements.txt /tmp/requirements.txt

# 1) PyTorch solo CPU (evita varios GB de librerías CUDA)
# 2) Resto de dependencias
# 3) Ultralytics arrastra opencv-python (con dependencias gráficas); se reemplaza por la versión headless
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r /tmp/requirements.txt \
 && pip uninstall -y opencv-python \
 && pip install --no-cache-dir --force-reinstall --no-deps opencv-python-headless

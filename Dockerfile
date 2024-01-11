FROM python:3.10-bookworm
RUN apt-get update -qq && apt-get install -y build-essential git ffmpeg libsm6 libxext6
WORKDIR /home
RUN pip install torch
RUN pip install av
RUN pip install argparse
RUN pip install opencv-python
RUN git clone https://github.com/Megvii-BaseDetection/YOLOX
WORKDIR /home/YOLOX
RUN pip3 install -v -e .
COPY . ./
CMD ["python", "main.py", "--frames", "100, 200, 224", "-i", "/home/YOLOX/intern_script/develop_streem.ts", "--model", "yolox-x", "--model_path", "/home/YOLOX/intern_script/yolox_x.pth"]
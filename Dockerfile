FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

# set working directory
WORKDIR /workspace

# copy requirements file
COPY requirements.txt .

RUN apt-get update && apt-get upgrade -y
RUN DEBIAN_FRONTEND="noninteractive" apt-get install -y libopencv-dev 

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt
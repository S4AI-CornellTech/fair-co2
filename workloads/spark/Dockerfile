FROM ubuntu:latest

# Set the working directory
WORKDIR home

# Copy the current directory contents into the container at /spark_benchmark
COPY ./spark-3.5.2-bin-hadoop3 /home
COPY ./pyspark_benchmark.py /home
COPY ./spark-defaults.conf /home

ENV LANG=en_US.utf8

# Install dependencies

RUN apt-get update && yes | apt install openjdk-8-jre-headless 
RUN apt-get install -y python3 && apt-get install -y python3-pip \

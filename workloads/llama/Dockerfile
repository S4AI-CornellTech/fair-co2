FROM ubuntu:latest

ARG home

# Set the working directory
WORKDIR $home/llama.cpp

# Copy the current directory contents into the container at 
ENV LANG=en_US.utf8

# Install dependencies

RUN apt-get update && yes | apt-get install libgomp1
FROM ubuntu:latest

# Set the working directory
WORKDIR /x265

RUN apt-get update && apt-get install -y locales && rm -rf /var/lib/apt/lists/* \
	&& localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8
ENV LANG=en_US.utf8

# Install dependencies
RUN apt-get update \
    && apt-get install -y x265

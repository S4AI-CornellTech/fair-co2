FROM ubuntu:latest

# Set the working directory
WORKDIR /pbbsbench

# Copy the current directory contents into the container at /pbbsbench
COPY algorithm ./algorithm
COPY common ./common
COPY benchmarks ./benchmarks
COPY PAM ./PAM
COPY parlay ./parlay
COPY parlaylib ./parlaylib
COPY runall .
COPY Makefile .

RUN apt-get update && apt-get install -y locales && rm -rf /var/lib/apt/lists/* \
	&& localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8
ENV LANG=en_US.utf8

# Install dependencies
RUN apt-get update \
    && apt-get install -y libjemalloc2 \
    && apt-get install -y numactl \
    && apt-get install -y python3 \
    && apt-get install -y python3-pip \
    && apt-get install -y build-essential 

CMD ["./runall", "-par", "-small"]

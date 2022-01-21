FROM docker.io/intel/oneapi-basekit
RUN mkdir /opt/build
WORKDIR /opt/build
RUN apt-get update && apt-get install -y libhdf5-dev git
RUN git clone --recursive https://github.com/ndevenish/miniapp ../miniapp -b oneapi_2022
RUN cmake ../miniapp/dpcpp/plumbing
RUN cmake --build . --target fpga

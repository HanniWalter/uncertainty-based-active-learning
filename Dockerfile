FROM ubuntu:20.04
# install tools to install dependencies
RUN apt-get clean
RUN apt update
RUN apt-get update --fix-missing
RUN apt install -y -qq git wget unzip tar


WORKDIR /application

COPY . .

RUN git clone --depth=1 https://github.com/ChristianKaltenecker/Distance-Based_Data.git \
    && tar -xzf Distance-Based_Data/SupplementaryWebsite/MeasuredPerformanceValues/JavaGC/measurements.tar.gz -C Distance-Based_Data/SupplementaryWebsite/MeasuredPerformanceValues/JavaGC/ \
    && tar -xzf Distance-Based_Data/SupplementaryWebsite/MeasuredPerformanceValues/VP9/measurements.tar.gz -C Distance-Based_Data/SupplementaryWebsite/MeasuredPerformanceValues/VP9/

COPY ./subject-systems-energy Distance-Based_Data/SupplementaryWebsite/MeasuredPerformanceValues


RUN apt install -y python3 
RUN apt install -y python3-pip
#COPY requirements.txt /tmp/
#RUN pip3 install --requirement /tmp/requirements.txt
RUN pip3 install git+https://github.com/pymc-devs/pymc3.git@v3.8
RUN pip3 install git+https://github.com/HanniWalter/pycosa-toolbox
RUN pip3 install more_itertools

RUN git clone https://github.com/HanniWalter/Mastering-Uncertainty-in-Performance-Estimations-of-Configurable-Software-Systems
RUN pip3 install Mastering-Uncertainty-in-Performance-Estimations-of-Configurable-Software-Systems/code/activesampler
RUN pip3 install Mastering-Uncertainty-in-Performance-Estimations-of-Configurable-Software-Systems/code/p4
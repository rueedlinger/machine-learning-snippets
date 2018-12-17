FROM continuumio/miniconda3:4.5.4


RUN conda create -n env python=3.6
RUN echo "source activate env" > ~/.bashrc

ENV PATH="/opt/conda/envs/env/bin:$PATH"


# https://github.com/ContinuumIO/docker-images/issues/79
#RUN conda update conda -y

ADD requirements.txt /
ADD docker-entrypoint.sh /

#RUN conda config --add channels conda-forge

RUN conda install --yes --file /requirements.txt

VOLUME ["/notebooks"]
EXPOSE 8888

ENTRYPOINT ["/docker-entrypoint.sh"]

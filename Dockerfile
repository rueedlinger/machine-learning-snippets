FROM continuumio/anaconda3

ENV PATH="/opt/conda/bin:${PATH}"

RUN conda install jupyter matplotlib seaborn scikit-learn pandas numpy scipy statsmodels -y

RUN mkdir /opt/notebooks

COPY basics /opt/notebooks/basics
COPY supervised /opt/notebooks/supervised
COPY unsupervised /opt/notebooks/unsupervised

CMD jupyter notebook --notebook-dir=/opt/notebooks --ip='*' --port=8888 --no-browser --allow-root


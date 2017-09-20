FROM continuumio/anaconda3

ENV PATH="/opt/conda/bin:${PATH}"

RUN conda install jupyter matplotlib seaborn scikit-learn pandas numpy scipy statsmodels -y

VOLUME ["/notebooks"]

CMD jupyter notebook --notebook-dir=/notebooks --ip='*' --port=8888 --no-browser --allow-root


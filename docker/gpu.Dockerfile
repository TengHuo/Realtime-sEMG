# tensorflow gpu docker
FROM tensorflow/tensorflow:1.13.1-gpu-py3
MAINTAINER teng <teng_huo@outlook.com>

RUN mkdir /emg \
    && mkdir -p /home/notebook/.jupyter/ \
    && useradd notebook -g root -u 1000 -d /home/notebook \
    && chown -R notebook /home/notebook/ \
    && chown -R notebook /emg/ \
    && pip install pandas scikit-learn seaborn jupyter

USER notebook

RUN echo "c.NotebookApp.ip = '*'" >> /home/notebook/.jupyter/jupyter_notebook_config.py \
    && echo "c.NotebookApp.token = ''" >> /home/notebook/.jupyter/jupyter_notebook_config.py \
    && echo "c.NotebookApp.notebook_dir = '/emg/'" >> /home/notebook/.jupyter/jupyter_notebook_config.py

CMD ["sh", "-c", "jupyter notebook --ip=0.0.0.0 --no-browser"]

WORKDIR /emg
FROM ubuntu:20.04

# Install Java and other dependencies
RUN apt-get update -y \
    && apt-get install openjdk-8-jdk -y \
    && apt-get install python3-pip -y \
    && apt-get install -y wget

RUN adduser myuser1
RUN wget https://dlcdn.apache.org/spark/spark-3.3.1/spark-3.3.1-bin-hadoop3.tgz
RUN tar xvf spark-3.3.1-bin-hadoop3.tgz -C /opt
RUN chown -R myuser1:myuser1 /opt/spark-3.3.1-bin-hadoop3
RUN ln -fs spark-3.3.1-bin-hadoop3 /opt/spark
RUN echo -e "export SPARK_HOME=/opt/spark\nPATH=$PATH:$SPARK_HOME/bin\nexport PATH" >> ~/.bash_profile
RUN . ~/.bash_profile
COPY --chown=myuser1:myuser1 . .
RUN pip3 install -r requirements.txt
ENTRYPOINT ["runuser", "-u", "myuser1", "--", "python3"]
CMD ["model_prediction.py"]

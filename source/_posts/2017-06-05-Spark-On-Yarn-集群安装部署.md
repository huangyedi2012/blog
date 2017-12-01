---
title: Spark On Yarn 集群安装部署
date: 2017-06-05 15:24:19
categories: spark
tags:
 - spark
---

本文记录的是Spark在Yarn上的集群安装部署

<!-- more -->

# 安装环境

OS: CentOS release 6.7 (Final)
Hadoop: 2.7.3
Spark: spark-2.1.1-bin-hadoop2.7

# 先决条件

## 安装 Scala

从[官方下载地址](http://www.scala-lang.org/download/)下载scala。

修改环境变量sudo vi /etc/profile，添加以下内容：

```bash
export SCALA_HOME=$HOME/local/opt/scala-2.12.2
export PATH=$PATH:$SCALA_HOME/bin
```

验证 scala 是否安装成功：

```bash
scala -version        #如果打印出如下版本信息，则说明安装成功
```

## 安装配置 Hadoop Yarn

Hadoop Yarn的安装见[Hadoop 2.7.3 安装](/2017/04/14/Hadoop-2-7-3-安装/)

# Spark安装

## 下载解压

进入[官方下载地址](http://spark.apache.org/downloads.html)下载最新版Spark。

在`~/local/opt`目录下解压

```bash
tar xzvf spark-2.1.1-bin-hadoop2.7.tgz
```

## 配置 Spark

```bash
cd ~/local/opt/spark-2.1.1-bin-hadoop2.7/conf    #进入spark配置目录
cp spark-env.sh.template spark-env.sh   #从配置模板复制
vi spark-env.sh     #添加配置内容
```

在`spark-env.sh`末尾添加以下内容:

```bash
export SCALA_HOME=/home/hadoop/local/opt/scala-2.12.2
export JAVA_HOME=/usr/lib/jvm/jre-1.8.0-openjdk.x86_64
export HADOOP_HOME=/home/hadoop/local/opt/hadoop-2.7.3
export HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop
SPARK_MASTER_IP=master
SPARK_LOCAL_DIRS=/home/hadoop/local/opt/spark-2.1.1-bin-hadoop2.7
SPARK_DRIVER_MEMORY=1G
```

在`slaves`文件下填上slave主机名：

```bash
slave1
slave2
```

将配置好的`scala-2.12.2`和`spark-2.1.1-bin-hadoop2.7`文件夹分发给所有slaves:

```bash
scp -r ~/local/opt/scala-2.12.2/ hadoop@slave1:~/local/opt/
scp -r ~/local/opt/scala-2.12.2/ hadoop@slave2:~/local/opt/
scp -r ~/local/opt/spark-2.1.1-bin-hadoop2.7 hadoop@slave1:~/local/opt/
scp -r ~/local/opt/spark-2.1.1-bin-hadoop2.7 hadoop@slave2:~/local/opt/
```
## 启动Spark

```bash
sbin/start-all.sh
```

## 验证Spark是否安装成功

用jps检查，在 master 上应该有以下几个进程：

```
6484 SecondaryNameNode
9156 HQuorumPeer
9223 HMaster
24871 JobHistoryServer
19771 Master
6283 NameNode
6653 ResourceManager
20222 Jps
```

在 slave 上应该有以下几个进程：

```
17088 NodeManager
17779 HRegionServer
31145 Worker
17692 HQuorumPeer
16973 DataNode
31390 Jps
```

进入Spark的Web管理页面： http://master:8080

![Spark的Web管理页面](/imgs/Spark/spark-install.png)

# 参考文献
>[Spark On YARN 集群安装部署](http://wuchong.me/blog/2015/04/04/spark-on-yarn-cluster-deploy/)


---
title: Hadoop之MapReduce原理
date: 2017-04-17 10:11:31
categories: hadoop
tags:
 - hadoop
 - mapreduce
---

Hadoop Map/Reduce是一个使用简易的软件框架，基于它写出来的应用程序能够运行在由上千个商用机器组成的大型集群上，并以一种可靠容错的方式并行处理上T级别的数据集。

<!-- more -->

# 概述

一个Map/Reduce作业（job）通常会把输入的数据集切分为若干独立的数据块，由map任务（task）以完全并行的方式处理它们。框架会对map的输出先进行排序，然后把结果输入给reduce任务。通常作业的输入和输出都会被存储在文件系统中。整个框架负责任务的调度和监控，以及重新执行已经失败的任务。

通常，Map/Reduce框架和分布式文件系统是运行在一组相同的节点上的，也就是说，计算节点和存储节点通常在一起。这种配置允许框架在那些已经存好数据的节点上高效地调度任务，这可以使整个集群的网络带宽被非常高效地利用。

Map/Reduce框架由一个单独的master JobTracker和每个集群节点一个slave TaskTracker共同组成。master负责调度构成一个作业的所有任务，这些任务分布在不同的slave上，master监控它们的执行，重新执行已经失败的任务。而slave仅负责执行由master指派的任务。

# MapReduce的架构

和HDFS一样，MapReduce也是采用Master/Slave的架构，其架构图如下所示。

![MapReduce架构图](/imgs/Hadoop/mapreduce/architecture.jpg)

MapReduce包含四个组成部分，分别为Client、JobTracker、TaskTracker和Task，下面我们详细介绍这四个组成部分。

## Client 客户端

每一个Job 都会在用户端通过Client类将应用程序以及配置参数Configuration打包成JAR文件存储在HDFS，并把路径提交到JobTracker的master服务，然后由master创建每一个Task（即MapTask 和ReduceTask）将它们分发到各个TaskTracker服务中去执行。

## JobTracker

JobTracke负责资源监控和作业调度。JobTracker监控所有TaskTracker与job的健康状况，一旦发现失败，就将相应的任务转移到其他节点；同时，JobTracker会跟踪任务的执行进度、资源使用量等信息，并将这些信息告诉任务调度器，而调度器会在资源出现空闲时，选择合适的任务使用这些资源。在Hadoop 中，任务调度器是一个可插拔的模块，用户可以根据自己的需要设计相应的调度器。

## TaskTracker

TaskTracker会周期性地通过Heartbeat将本节点上资源的使用情况和任务的运行进度汇报给JobTracker，同时接收JobTracker发送过来的命令并执行相应的操作（如启动新任务、杀死任务等）。TaskTracker 使用"slot"等量划分本节点上的资源量。"slot"代表计算资源（CPU、内存等）。一个Task获取到一个slot后才有机会运行，而Hadoop调度器的作用就是将各个TaskTracker上的空闲slot分配给Task使用。slot分为Map slot和Reduce slot两种，分别供Map Task和Reduce Task使用。TaskTracker通过slot数目（可配置参数）限定Task的并发度。

## Task

Task 分为Map Task和Reduce Task两种，均由TaskTracker启动。HDFS以固定大小的block为基本单位存储数据，而对于MapReduce而言，其处理单位是split。

Map Task执行过程如下图所示：由该图可知，Map Task先将对应的split迭代解析成一个个key/value 对，依次调用用户自定义的map()函数进行处理，最终将临时结果存放到本地磁盘上, 其中临时数据被分成若干个partition，每个partition将被一个Reduce Task处理。

![Map Task执行过程](/imgs/Hadoop/mapreduce/maptask.jpg)

Reduce Task执行过程下图所示。该过程分为三个阶段：

![Reduce Task执行过程](/imgs/Hadoop/mapreduce/reducetask.jpg)

1. 从远程节点上读取Map Task 中间结果（称为“Shuffle 阶段”）；
2. 按照key 对key/value 对进行排序（称为“Sort 阶段”）；
3. 依次读取< key, value list>，调用用户自定义的reduce() 函数处理，并将最终结果存到HDFS 上（称为“Reduce 阶段”）。

# MapReduce运行机制

下面从逻辑实体的角度介绍mapreduce运行机制，这些按照时间顺序包括：输入分片（input split）、map阶段、combiner阶段、shuffle阶段和reduce阶段。

![MapReduce作业运行流程](/imgs/Hadoop/mapreduce/mrjob.png)

在MapReduce运行过程中，最重要的是Map，Shuffle，Reduce三个阶段，各个阶段的作用简述如下：

- Map:数据输入,做初步的处理,输出形式的中间结果；
- Shuffle:按照partition、key对中间结果进行排序合并,输出给reduce线程；
- Reduce:对相同key的输入进行最终的处理,并将结果写入到文件中。

![MapReduce Shuffle过程](/imgs/Hadoop/mapreduce/shuffle.png)

上图是把MapReduce过程分为两个部分，而实际上从两边的Map和Reduce到中间的那一大块都属于Shuffle过程，也就是说，Shuffle过程有一部分是在Map端，有一部分是在Reduce端。


## 输入分片（input split）

### InputFormat

InputFormat为Map/Reduce作业描述输入的细节规范。Map/Reduce框架根据作业的InputFormat来进行以下操作：

- 检查作业输入的有效性。
- 把输入文件切分成多个逻辑InputSplit实例，并把每一实例分别分发给一个Mapper。
- 提供RecordReader的实现，这个RecordReader从逻辑InputSplit中获得输入记录，这些记录将由Mapper处理。

InputFormat只包含了两个接口函数:

```java
InputSplit[] getSplits(JobConf job, int numSplits) throws IOException;

RecordReader < K, V> getRecordReader(InputSplit split, JobConf job, Reporter reporter) throws IOException;
```

getSplits就是现在要使用的划分函数。job参数是任务的配置集合，从中可以取到用户在启动MapReduce时指定的输入文件路径。而numSplits参数是一个Split数目的建议值，是否考虑这个值，由具体的InputFormat实现来决定。返回的是InputSplit数组，它描述了所有的Split信息，一个InputSplit描述一个Split。

getRecordReader方法返回一个RecordReader对象，该对象将输入的InputSplit解析成若干个key/value对，MapReduce框架在Map Task执行过程中，会不断的调用RecordReader对象中的方法，迭代获取key/value对并交给map函数处理。

### InputSplit

InputSplit是一个单独的Mapper要处理的数据块。一般的InputSplit是字节样式输入，然后由RecordReader处理并转化成记录样式。InputSplit也只有两个接口函数：

```java
long getLength() throws IOException;

String[] getLocations() throws IOException;
```

这个interface仅仅描述了Split有多长，以及存放这个Split的Location信息（也就是这个Split在HDFS上存放的机器。它可能有多个replication，存在于多台机器上）。除此之外，就再没有任何直接描述Split的信息了。

而Split中真正重要的描述信息还是只有InputFormat会关心。在需要读取一个Split的时候，其对应的InputSplit会被传递到InputFormat的第二个接口函数getRecordReader，然后被用于初始化一个RecordReader，以解析输入数据。

在分配Map任务时，Split的Location信息就要发挥作用了。JobTracker会根据TaskTracker的地址来选择一个Location与之最接近的Split所对应的Map任务（注意一个Split可以有多个Location）。这样一来，输入文件中Block的Location信息经过一系列的整合（by InputFormat）和传递，最终就影响到了Map任务的分配。其结果是Map任务倾向于处理存放在本地的数据，以保证效率。

### RecordReader

RecordReader从InputSlit读入&lt;key, value&gt;对。

一般的，RecordReader 把由InputSplit 提供的字节样式的输入文件，转化成由Mapper处理的记录样式的文件。因此RecordReader负责处理记录的边界情况和把数据表示成&lt;keys, values&gt;对形式。

## Map阶段

在进行海量数据处理时，外存文件数据I/O访问会成为一个制约系统性能的瓶颈，因此，Hadoop的Map过程实现的一个重要原则就是：计算靠近数据，这里主要指两个方面：

1. 代码靠近数据：
	- 原则：本地化数据处理（locality），即一个计算节点尽可能处理本地磁盘上所存储的数据；
	- 尽量选择数据所在DataNode启动Map任务；
	- 这样可以减少数据通信，提高计算效率；
2. 数据靠近代码：
	- 当本地没有数据处理时，尽可能从同一机架或最近其他节点传输数据进行处理（host选择算法）。

map的经典流程图如下：

![Map Shuffle过程](/imgs/Hadoop/mapreduce/map-shuffle.png)

### 输入

1. map task只读取split分片，split与block（HDFS的最小存储单位，默认为64MB）可能是一对一也能是一对多，但是对于一个split只会对应一个文件的一个block或多个block，不允许一个split对应多个文件的多个block；
2. 这里切分和输入数据的时会涉及到InputFormat的文件切分算法和host选择算法。

文件切分算法，主要用于确定InputSplit的个数以及每个InputSplit对应的数据段。FileInputFormat以文件为单位切分生成InputSplit，对于每个文件，由以下三个属性值决定其对应的InputSplit的个数：

- goalSize： 它是根据用户期望的InputSplit数目计算出来的，即totalSize/numSplits。其中，totalSize为文件的总大小；numSplits为用户设定的Map Task个数，默认情况下是1；
- minSize：InputSplit的最小值，由配置参数mapred.min.split.size确定，默认是1；
- blockSize：文件在hdfs中存储的block大小，不同文件可能不同，默认是64MB。

这三个参数共同决定InputSplit的最终大小，计算方法如下：
$$
splitSize=\max(minSize, \min(gogalSize,blockSize))
$$

### Partitioner

作用：将map的结果发送到相应的reduce端，总的partition的数目等于reducer的数量。

实现功能：

1. map输出的是&lt;key,value&gt;对，决定于当前的mapper的partition交给哪个reduce的方法是：mapreduce提供的Partitioner接口，对key进行hash后，再以reducetask数量取模，然后到指定的job上（HashPartitioner，可以通过`job.setPartitionerClass(MyPartition.class)`自定义）。
2. 然后将数据写入到内存缓冲区，缓冲区的作用是批量收集map结果，减少磁盘IO的影响。&lt;key,value&gt;对以及Partition的结果都会被写入缓冲区。在写入之前，key与value值都会被序列化成字节数组。

要求：负载均衡，效率；

### spill（溢写）

作用：把内存缓冲区中的数据写入到本地磁盘，在写入本地磁盘时先按照partition、再按照key进行排序（quick sort）；

注意：

1. 这个spill是由另外单独的线程来完成，不影响往缓冲区写map结果的线程；
2. 内存缓冲区默认大小限制为100MB，它有个溢写比例（spill.percent），默认为0.8，当缓冲区的数据达到阈值时，溢写线程就会启动，先锁定这80MB的内存，执行溢写过程，maptask的输出结果还可以往剩下的20MB内存中写，互不影响。然后再重新利用这块缓冲区，因此Map的内存缓冲区又叫做环形缓冲区（两个指针的方向不会变，下面会详述）；
3. 在将数据写入磁盘之前，先要对要写入磁盘的数据进行一次排序操作，先按&lt;key,value,partition&gt;中的partition分区号排序，然后再按key排序，这个就是sort操作，最后溢出的小文件是分区的，且同一个分区内是保证key有序的；

### Combiner

combine：执行combine操作要求开发者必须在程序中设置了combine（程序中通过`job.setCombinerClass(myCombine.class)`自定义combine操作）。

程序中有两个阶段可能会执行combine操作：

1. map输出数据根据分区排序完成后，在写入文件之前会执行一次combine操作（前提是作业中设置了这个操作）；
2. 如果map输出比较大，溢出文件个数大于3（此值可以通过属性`min.num.spills.for.combine`配置）时，在merge的过程（多个spill文件合并为一个大文件）中还会执行combine操作；

combine主要是把形如&lt;aa,1&gt;,&lt;aa,2&gt;这样的key值相同的数据进行计算，计算规则与reduce一致，比如：当前计算是求key对应的值求和，则combine操作后得到&lt;aa,3&gt;这样的结果。

注意事项：不是每种作业都可以做combine操作的，只有满足以下条件才可以：

1. reduce的输入输出类型都一样，因为combine本质上就是reduce操作；
2. 计算逻辑上，combine操作后不会影响计算结果，像求和就不会影响；

### merge

当map很大时，每次溢写会产生一个spill_file，这样会有多个spill_file，而最终的一个map task输出只有一个文件，因此，最终的结果输出之前会对多个中间过程进行多次溢写文件（spill_file）的合并，此过程就是merge过程。也即是，待Map Task任务的所有数据都处理完后，会对任务产生的所有中间数据文件做一次合并操作，以确保一个Map Task最终只生成一个中间数据文件。

注意：

1. 如果生成的文件太多，可能会执行多次合并，每次最多能合并的文件数默认为10，可以通过属性`min.num.spills.for.combine`配置；
2. 多个溢出文件合并时，会进行一次排序，排序算法是**多路归并排序**；
3. 是否还需要做combine操作，一是看是否设置了combine，二是看溢出的文件数是否大于等于3；
4. 最终生成的文件格式与单个溢出文件一致，也是按分区顺序存储，并且输出文件会有一个对应的索引文件，记录每个分区数据的起始位置，长度以及压缩长度，这个索引文件名叫做`file.out.index`。

### 内存缓冲区

在Map Task任务的业务处理方法map()中，最后一步通过`OutputCollector.collect(key,value)`或`context.write(key,value)`输出Map Task的中间处理结果，在相关的`collect(key,value)`方法中，会调用`Partitioner.getPartition(K2 key, V2 value, int numPartitions)`方法获得输出的&lt;key,value&gt;对应的分区号(分区号可以认为对应着一个要执行Reduce Task的节点)，然后将&lt;key,value,partition&gt;暂时保存在内存中的MapOutputBuffe内部的环形数据缓冲区，该缓冲区的默认大小是100MB，可以通过参数`io.sort.mb`来调整其大小。

当缓冲区中的数据使用率达到一定阀值后，触发一次Spill操作，将环形缓冲区中的部分数据写到磁盘上，生成一个临时的Linux本地数据的spill文件；然后在缓冲区的使用率再次达到阀值后，再次生成一个spill文件。直到数据处理完毕，在磁盘上会生成很多的临时文件。

缓存有一个阀值比例配置，当达到整个缓存的这个比例时，会触发spill操作；触发时，map输出还会接着往剩下的空间写入，但是写满的空间会被锁定，数据溢出写入磁盘。当这部分溢出的数据写完后，空出的内存空间可以接着被使用，形成像环一样的被循环使用的效果，所以又叫做环形内存缓冲区；

## Reduce阶段

Reduce过程的经典流程图如下：

![Redece过程流程图](/imgs/Hadoop/mapreduce/reduce-shuffle.png)

### copy

作用：拉取数据；

过程：Reduce进程启动一些数据copy线程(Fetcher)，通过HTTP方式请求map task所在的TaskTracker获取map task的输出文件。因为这时map task早已结束，这些文件就归TaskTracker管理在本地磁盘中。

默认情况下，当整个MapReduce作业的所有已执行完成的Map Task任务数超过Map Task总数的5%后，JobTracker便会开始调度执行Reduce Task任务。然后Reduce Task任务默认启动`mapred.reduce.parallel.copies`(默认为5）个MapOutputCopier线程到已完成的Map Task任务节点上分别copy一份属于自己的数据。 这些copy的数据会首先保存的内存缓冲区中，当内冲缓冲区的使用率达到一定阀值后，则写到磁盘上。

### merge

Copy过来的数据会先放入内存缓冲区中，这里的缓冲区大小要比map端的更为灵活，它基于JVM的heap size设置，因为Shuffle阶段Reducer不运行，所以应该把绝大部分的内存都给Shuffle用。

这里需要强调的是，merge有三种形式：1)内存到内存 2)内存到磁盘 3)磁盘到磁盘。默认情况下第一种形式是不启用的。当内存中的数据量到达一定阈值，就启动内存到磁盘的merge（图中的第一个merge，之所以进行merge是因为reduce端在从多个map端copy数据的时候，并没有进行sort，只是把它们加载到内存，当达到阈值写入磁盘时，需要进行merge） 。这和map端的很类似，这实际上就是溢写的过程，在这个过程中如果你设置有Combiner，它也是会启用的，然后在磁盘中生成了众多的溢写文件，这种merge方式一直在运行，直到没有 map 端的数据时才结束，然后才会启动第三种磁盘到磁盘的 merge （图中的第二个merge）方式生成最终的那个文件。

在远程copy数据的同时，Reduce Task在后台启动了两个后台线程对内存和磁盘上的数据文件做合并操作，以防止内存使用过多或磁盘生的文件过多。

### Reduce

merge的最后会生成一个文件，大多数情况下存在于磁盘中，但是需要将其放入内存中。当reducer 输入文件已定，整个 Shuffle 阶段才算结束。然后就是 Reducer 执行，把结果放到 HDFS 上。

Reduce的数目建议是0.95或1.75乘以 (`<no. of nodes> * mapred.tasktracker.reduce.tasks.maximum`)。

用0.95，所有reduce可以在maps一完成时就立刻启动，开始传输map的输出结果。用1.75，速度快的节点可以在完成第一轮reduce任务后，可以开始第二轮，这样可以得到比较好的负载均衡的效果。

如果没有归约要进行，那么设置reduce任务的数目为零是合法的。这种情况下，map任务的输出会直接被写入由 setOutputPath(Path)指定的输出路径。框架在把它们写入FileSystem之前没有对它们进行排序。

# 参考文献

>[Hadoop Map/Reduce教程](http://hadoop.apache.org/docs/r1.0.4/cn/mapred_tutorial.html)
>[深入理解MapReduce的架构及原理](http://blog.csdn.net/u010330043/article/details/51200712)
>[Hadoop InputFormat浅析--hadoop如何分配输入](http://blog.csdn.net/hsuxu/article/details/7673171/)
>[MapReduce之Shuffle过程详述](http://wangzzu.github.io/2016/03/02/hadoop-shuffle/)
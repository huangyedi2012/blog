---
title: Hadoop之HDFS架构
date: 2017-04-14 15:27:16
categories: hadoop
tags:
 - hadoop
 - hdfs
---

HDFS即Hadoop Distributed File System分布式文件系统，它的设计目标是把超大数据集存储到分布在网络中的多台普通商用计算机上，并且能够提供高可靠性和高吞吐量的服务。

<!-- more -->

# HDFS架构原理

HDFS主要由3个组件构成，分别是NameNode、SecondaryNameNode和DataNode，HDFS是以master/slave模式运行的，其中NameNode、SecondaryNameNode 运行在master节点，DataNode运行slave节点。

## 数据块

磁盘数据块是磁盘读写的基本单位，与普通文件系统类似，hdfs也会把文件分块来存储。hdfs默认数据块大小为64MB，磁盘块一般为512B，HDFS块为何如此之大呢？块增大可以减少寻址时间与文件传输时间的比例，若寻址时间为10ms，磁盘传输速率为100MB/s，那么寻址与传输比仅为1%。当然，磁盘块太大也不好，因为一个MapReduce通常以一个块作为输入，块过大会导致整体任务数量过小，降低作业处理速度。

hdfs按块存储还有如下好处：

1. 文件可以任意大，也不用担心单个结点磁盘容量小于文件的情况
2. 简化了文件子系统的设计，子系统只存储文件块数据，而文件元数据则交由其它系统（NameNode）管理
3. 有利于备份和提高系统可用性，因为可以以块为单位进行备份，hdfs默认备份数量为3。
4. 有利于负载均衡

## NameNode

### NameNode中的元信息

当一个客户端请求一个文件或者存储一个文件时，它需要先知道具体到哪个DataNode上存取，获得这些信息后，客户端再直接和这个DataNode进行交互，而这些信息的维护者就是NameNode。

NameNode管理着文件系统命名空间，它维护这文件系统树及树中的所有文件和目录。NameNode也负责维护所有这些文件或目录的打开、关闭、移动、重命名等操作。对于实际文件数据的保存与操作，都是由DataNode负责。当一个客户端请求数据时，它仅仅是从NameNode中获取文件的元信息，而具体的数据传输不需要经过NameNode，是由客户端直接与相应的DataNode进行交互。

NameNode保存元信息的种类有：

- 文件名目录名及它们之间的层级关系
- 文件目录的所有者及其权限
- 每个文件块的名及文件有哪些块组成

需要注意的是，**NameNode元信息并不包含每个块的位置信息**，这些信息会在NameNode启动时从各个DataNode获取并保存在内存中，因为这些信息会在系统启动时由数据节点重建。把块位置信息放在内存中，在读取数据时会减少查询时间，增加读取效率。

### 元信息的持久化

在NameNode中存放元信息的文件是fsimage。在系统运行期间所有对元信息的操作都保存在内存中并被持久化到另一个文件edits中。并且edits文件和fsimage文件会被SecondaryNameNode周期性的合并（合并过程会在SecondaryNameNode中详细介绍）。

### 其它问题

运行NameNode会占用大量内存和I/O资源，一般NameNode不会存储用户数据或执行MapReduce任务。

为了简化系统的设计，Hadoop只有一个NameNode，这也就导致了hadoop集群的单点故障问题。因此，对NameNode节点的容错尤其重要，hadoop提供了如下两种机制来解决：

- 将hadoop元数据写入到本地文件系统的同时再实时同步到一个远程挂载的网络文件系统（NFS）。
- 运行一个secondaryNameNode，它的作用是与NameNode进行交互，定期通过编辑日志文件合并命名空间镜像，当NameNode发生故障时它会通过自己合并的命名空间镜像副本来恢复。需要注意的是secondaryNameNode保存的状态总是滞后于NameNode，所以这种方式难免会导致丢失部分数据（后面会详细介绍）。

## SecondaryNameNode

需要注意，SecondaryNameNode并不是NameNode的备份。我们从前面的介绍已经知道，所有HDFS文件的元信息都保存在NameNode的内存中。在NameNode启动时，它首先会加载fsimage到内存中，在系统运行期间，所有对NameNode的操作也都保存在了内存中，同时为了防止数据丢失，这些操作又会不断被持久化到本地edits文件中。

Edits文件存在的目的是为了提高系统的操作效率，NameNode在更新内存中的元信息之前都会先将操作写入edits文件。在NameNode重启的过程中，edits会和fsimage合并到一起，但是合并的过程会影响到Hadoop重启的速度，SecondaryNameNode就是为了解决这个问题而诞生的。

SecondaryNameNode的角色就是定期的合并edits和fsimage文件，我们来看一下合并的步骤：

![  Secondary NameNode处理流程 ](/imgs/Hadoop/hdfs/copyfsimage.jpg)

1. 合并之前告知NameNode把所有的操作写到新的edites文件并将其命名为edits.new。
2. SecondaryNameNode从NameNode请求fsimage和edits文件
3. SecondaryNameNode把fsimage和edits文件合并成新的fsimage文件
4. NameNode从SecondaryNameNode获取合并好的新的fsimage并将旧的替换掉，并把edits用第一步创建的edits.new文件替换掉
5. 更新fstime文件中的检查点

最后再总结一下整个过程中涉及到NameNode中的相关文件

- fsimage ：保存的是上个检查点的HDFS的元信息
- edits ：保存的是从上个检查点开始发生的HDFS元信息状态改变信息
- fstime：保存了最后一个检查点的时间戳

## DataNode

DataNode是HDFS中的worker节点，它负责存储数据块，也负责为系统客户端提供数据块的读写服务，同时还会根据NameNode的指示来进行创建、删除、和复制等操作。此外，它还会通过心跳定期向NameNode发送所存储文件块列表信息。当对hdfs文件系统进行读写时，NameNode告知客户端每个数据驻留在哪个DataNode，客户端直接与DataNode进行通信，DataNode还会与其它DataNode通信，复制这些块以实现冗余。


## 数据备份

HDFS通过备份数据块的形式来实现容错，除了文件的最后一个数据块外，其它所有数据块大小都是一样的。NameNode负责各个数据块的备份，DataNode会通过心跳的方式定期的向NameNode发送自己节点上的Block报告，这个报告中包含了DataNode节点上的所有数据块的列表。

一个大型的HDFS文件系统一般都是需要跨很多机架的，不同机架之间的数据传输需要经过网关，并且，同一个机架中机器之间的带宽要大于不同机架机器之间的带宽。如果把所有的副本都放在不同的机架中，这样既可以防止机架失败导致数据块不可用，又可以在读数据时利用到多个机架的带宽，并且也可以很容易的实现负载均衡。但是，如果是写数据，各个数据块需要同步到不同的机架，会影响到写数据的效率。而在Hadoop中，如果副本数量是3的情况下，Hadoop默认是这么存放的，把第一个副本放到机架的一个节点上，另一个副本放到同一个机架的另一个节点上，把最后一个节点放到不同的机架上。这种策略减少了跨机架副本的个数提高了写的性能，也能够允许一个机架失败的情况，算是一个很好的权衡。

![副本摆放策略 ](/imgs/Hadoop/hdfs/copylocation.jpg)

## 安全模式

关于安全模式，当 Hadoop的NameNode节点启动时，会进入安全模式阶段。在此阶段，DataNode会向NameNode上传它们数据块的列表，让 NameNode得到块的位置信息，并对每个文件对应的数据块副本进行统计。当最小副本条件满足时，即一定比例的数据块都达到最小副本数，系统就会退出安全模式，而这需要一定的延迟时间。当最小副本条件未达到要求时，就会对副本数不足的数据块安排DataNode进行复制，直至达到最小副本数。而在安全模式下，系统会处于只读状态，NameNode不会处理任何块的复制和删除命令。

## HDFS负载均衡

HDFS的数据也许并不是非常均匀的分布在各个DataNode中。一个常见的原因是在现有的集群上经常会增添新的DataNode节点。当新增一个数据块（一个文件的数据被保存在一系列的块中）时，NameNode在选择DataNode接收这个数据块之前，会考虑到很多因素。其中的一些考虑的是：

- 将数据块的一个副本放在正在写这个数据块的节点上。
- 尽量将数据块的不同副本分布在不同的机架上，这样集群可在完全失去某一机架的情况下还能存活。
- 一个副本通常被放置在和写文件的节点同一机架的某个节点上，这样可以减少跨越机架的网络I/O。
- 尽量均匀地将HDFS数据分布在集群的DataNode中。

# HDFS健壮性

HDFS的主要目标就是即使在出错的情况下也要保证数据存储的可靠性。常见的三种出错情况是： Namenode出错 , Datanode出错和网络割裂(network partitions)。

## 磁盘数据错误，心跳检测和重新复制

每个Datanode节点周期性地向Namenode发送心跳信号。网络割裂可能导致一部分Datanode跟 Namenode失去联系。Namenode通过心跳信号的缺失来检测这一情况，并将这些近期不再发送心跳信号 Datanode标记为宕机，不会再将新的IO请求发给它们。任何存储在宕机Datanode上的数据将不再有效。Datanode的宕机可能会引起一些数据块的副本系数低于指定值，Namenode不断地检测这些需要复制的数据块，一旦发现就启动复制操作。

## 数据完整性

从某个Datanode获取的数据块有可能是损坏的，损坏可能是由Datanode的存储设备错误、网络错误或者软件bug造成的。HDFS客户端软件实现了对HDFS文件内容的校验和(checksum)检查。当客户端创建一个新的HDFS文件，会计算这个文件每个数据块的校验和，并将校验和作为一个单独的隐藏文件保存在同一个HDFS名字空间下。当客户端获取文件内容后，它会检验从Datanode获取的数据跟相应的校验和文件中的校验和是否匹配，如果不匹配，客户端可以选择从其他Datanode 获取该数据块的副本。

## 元数据磁盘错误

FsImage和Editlog是HDFS的核心数据结构。如果这些文件损坏了，整个HDFS 实例都将失效。因而，Namenode可以配置成支持维护多个FsImage和Editlog的副本。任何对FsImage或者Editlog的修改，都将同步到它们的副本上。这种多副本的同步操作可能会降低Namenode每秒处理的名字空间事 务数量。然而这个代价是可以接受的，因为即使HDFS的应用是数据密集的，它们也非元数据密集的。当 Namenode重启的时候，它会选取最近的完整的FsImage和Editlog来使用。

Namenode是HDFS集群中的单点故障(single point of failure)所在。如果Namenode机器故障，是需要手工干预的。目前，自动重启或在另一台机器上做Namenode故障转移的功能还没实现。

# HDFS网络

## HDFS中的沟通协议

所有的HDFS中的沟通协议都是基于tcp/ip协议，一个客户端通过指定的tcp端口与NameNode机器建立连接，并通过ClientProtocol协议与NameNode交互。而DataNode则通过DataNode Protocol协议与NameNode进行沟通。HDFS的RCP(远程过程调用)对ClientProtocol和DataNode Protocol做了封装。按照HDFS的设计，NameNode不会主动发起任何请求，只会被动接受来自客户端或DataNode的请求。

## HDFS机架感知与网络拓扑

通常，大型Hadoop集群是以机架的形式来组织的，而HDFS不能够自动判断集群中各个datanode的网络拓扑情况，因此Hadoop允许集群的管理员通过配置`dfs.network.script`参数来确定节点所处的机架。 文件提供了IP->rackid的翻译。NameNode通过这个得到集群中各个datanode机器的rackid。如果`topology.script.file.name`没有设定，则每个IP都会翻译 成`/default-rack`。

Hadoop把网络拓扑看成是一棵树，两个节点的距离=它们到最近共同祖先距离的总和，而树的层次可以这么划分：

- 同一节点中的进程
- 同一机架上的不同节点
- 同一数据中心不同机架
- 不同数据中心的节点

![datanode 网络拓扑图](/imgs/Hadoop/hdfs/datanodetop.jpg)

若数据中心$d_1$中一个机架$r_1$中一个节点$n_1$表示为$d_1/r_1/n_1$,则：

$$
 distance(d_1/r_1/n_1,d_1/r_1/n_1)=0; 相同的datanode\\\\
 distance(d_1/r_1/n_1,d_1/r_1/n_2)=2; 同一rack下的不同datanode\\\\
 distance(d_1/r_1/n_1,d_1/r_2/n_3)=4; 同一IDC下的不同datanode\\\\
 distance(d_1/r_1/n_1,d_2/r_3/n_4)=6; 不同IDC下的datanode
$$

# hdfs文件读写过程剖析

![NameNode和DataNode架构图](/imgs/Hadoop/hdfs/hdfs-architecture.gif)

## hdfs文件读取过程

HDFS有一个FileSystem实例，客户端通过调用这个实例的open()方法就可以打开系统中希望读取的文件。HDFS通过rpc调用NameNode获取文件块的位置信息，对于文件的每一个块，NameNode会返回含有该块副本的DataNode的节点地址，另外，客户端还会根据网络拓扑来确定它与每一个DataNode的位置信息，从离它最近的那个DataNode获取数据块的副本，最理想的情况是数据块就存储在客户端所在的节点上。

HDFS会返回一个FSDataInputStream对象，FSDataInputStream类转而封装成DFSDataInputStream对象,这个对象管理着与DataNode和NameNode的I/O，具体过程是：

1. 客户端发起读请求
2. 客户端与NameNode得到文件的块及位置信息列表
3. 客户端直接和DataNode交互读取数据
4. 读取完成关闭连接

当FSDataInputStream与DataNode通信时遇到错误，它会选取另一个较近的DataNode，并为出故障的DataNode做标记以免重复向其读取数据。FSDataInputStream还会对读取的数据块进行校验和确认，发现块损坏时也会重新读取并通知NameNode。

这样设计的巧妙之处：

1. 让客户端直接联系DataNode检索数据，可以使HDFS扩展到大量的并发客户端，因为数据流就是分散在集群的每个节点上的，在运行MapReduce任务时，每个客户端就是一个DataNode节点。
2. NameNode仅需相应块的位置信息请求（位置信息在内存中，速度极快），否则随着客户端的增加，NameNode会很快成为瓶颈。

## HDFS文件写入过程

HDFS有一个DistributedFileSystem实例，客户端通过调用这个实例的create()方法就可以创建文件。DistributedFileSystem会发送给NameNode一个RPC调用，在文件系统的命名空间创建一个新文件，在创建文件前NameNode会做一些检查，如文件是否存在，客户端是否有创建权限等，若检查通过，NameNode会为创建文件写一条记录到本地磁盘的EditLog，若不通过会向客户端抛出IOException。创建成功之后DistributedFileSystem会返回一个FSDataOutputStream对象，客户端由此开始写入数据。

同读文件过程一样，FSDataOutputStream类转而封装成DFSDataOutputStream对象,这个对象管理着与DataNode和NameNode的I/O，具体过程是：

1. 客户端在向NameNode请求之前先写入文件数据到本地文件系统的一个临时文件
2. 待临时文件达到块大小时开始向NameNode请求DataNode信息
3. NameNode在文件系统中创建文件并返回给客户端一个数据块及其对应DataNode的地址列表（列表中包含副本存放的地址）
4. 客户端通过上一步得到的信息把创建临时文件块flush到列表中的第一个DataNode
5. 当文件关闭，NameNode会提交这次文件创建，此时，文件在文件系统中可见

上面第四步描述的flush过程实际处理过程比较负杂，现在单独描述一下：

1. 首先，第一个DataNode是以数据包(数据包一般4KB)的形式从客户端接收数据的，DataNode在把数据包写入到本地磁盘的同时会向第二个DataNode（作为副本节点）传送数据。
2. 在第二个DataNode把接收到的数据包写入本地磁盘时会向第三个DataNode发送数据包
3. 第三个DataNode开始向本地磁盘写入数据包。此时，数据包以流水线的形式被写入和备份到所有DataNode节点
4. 传送管道中的每个DataNode节点在收到数据后都会向前面那个DataNode发送一个ACK,最终，第一个DataNode会向客户端发回一个ACK
5. 当客户端收到数据块的确认之后，数据块被认为已经持久化到所有节点。然后，客户端会向NameNode发送一个确认
6. 如果管道中的任何一个DataNode失败，管道会被关闭。数据将会继续写到剩余的DataNode中。同时NameNode会被告知待备份状态，NameNode会继续备份数据到新的可用的节点
7. 数据块都会通过计算校验和来检测数据的完整性，校验和以隐藏文件的形式被单独存放在hdfs中，供读取时进行完整性校验

## hdfs文件删除过程

hdfs文件删除过程一般需要如下几步：

1. 一开始删除文件，NameNode只是重命名被删除的文件到/trash目录，因为重命名操作只是元信息的变动，所以整个过程非常快。在/trash中文件会被保留一定间隔的时间（可配置，默认是6小时），在这期间，文件可以很容易的恢复，恢复只需要将文件从/trash移出即可。
2. 当指定的时间到达，NameNode将会把文件从命名空间中删除
3. 标记删除的文件块释放空间，HDFS文件系统显示空间增加

# HDFS缺点

一般来说，一条元信息记录会占用200byte内存空间。假设块大小为64MB，备份数量是3 ，那么一个1GB大小的文件将占用1GB/64MB\*3=48个文件块。如果现在有1000个1MB大小的文件，则会占用1000*3=3000个文件块（多个文件不能放到一个块中）。我们可以发现，如果文件越小，存储同等大小文件所需要的元信息就越多，所以，Hadoop更喜欢大文件。

还有一个问题就是，因为 Map task 的数量是由 splits 来决定的，所以 用 MR 处理大量的小文件时，就会产生过多的 Maptask ，线程管理开销将会增加作业时间。举个例子，处理10000M的文件，若每个split为1M ，那就会有10000个Maptasks，会有很大的线程开销；若每个split为 100M，则只有100个Maptasks，每个Maptask 将会有更多的事情做，而线程的管理开销也将减小很多。

# 参考文献

>[HDFS 原理、架构与特性介绍](https://my.oschina.net/leejun2005/blog/151872)
>[Hadoop核心之HDFS 架构设计](http://blog.csdn.net/suifeng3051/article/details/48548341)
---
title: Hadoop之Yarn原理
date: 2017-04-17 19:43:49
categories: hadoop
tags:
 - hadoop
 - yarn
---

YARN（Yet Another Resource Negotiator）是一个通用的资源管理平台，可为各类计算框架提供资源的管理和调度。

<!-- more -->

# Hadoop MapReduce 框架的问题

随着分布式系统集群的规模和其工作负荷的增长，MapReduce 框架的问题逐渐浮出水面，主要的问题集中如下：

1. JobTracker 是 Map-reduce 的集中处理点，存在单点故障。
2. JobTracker 完成了太多的任务，造成了过多的资源消耗，当 map-reduce job 非常多的时候，会造成很大的内存开销，潜在来说，也增加了 JobTracker fail 的风险，这也是业界普遍总结出老 Hadoop 的 Map-Reduce 只能支持 4000 节点主机的上限。
3. 在 TaskTracker 端，以 map/reduce task 的数目作为资源的表示过于简单，没有考虑到 cpu/ 内存的占用情况，如果两个大内存消耗的 task 被调度到了一块，很容易出现 OOM(Out-Of-Memory)。
4. 在 TaskTracker 端，把资源强制划分为 map task slot 和 reduce task slot, 如果当系统中只有 map task 或者只有 reduce task 的时候，会造成资源的浪费，也就是前面提过的集群资源利用的问题。
5. 源代码层面分析的时候，会发现代码非常的难读，常常因为一个 class 做了太多的事情，代码量达 3000 多行，造成 class 的任务不清晰，增加 bug 修复和版本维护的难度。
6. 从操作的角度来看，现在的 Hadoop MapReduce 框架在有任何重要的或者不重要的变化 ( 例如 bug 修复，性能提升和特性化 ) 时，都会强制进行系统级别的升级更新。更糟的是，它不管用户的喜好，强制让分布式集群系统的每一个用户端同时更新。这些更新会让用户为了验证他们之前的应用程序是不是适用新的 Hadoop 版本而浪费大量时间。

# Yarn介绍

Yarn的核心出发点是为了分离资源管理与作业调度/监控，实现分离的做法是拥有一个全局的资源管理器（ResourceManager，RM），以及每个应用程序对应一个的应用管理器（ApplicationMaster，AM），应用程序由一个作业（Job）或者Job的有向无环图（DAG）组成。

YARN可以将多种计算框架(如离线处理MapReduce、在线处理的Storm、迭代式计算框架Spark、流式处理框架S4等) 部署到一个公共集群中，共享集群的资源。并提供如下功能：

1. 资源的统一管理和调度：
	集群中所有节点的资源(内存、CPU、磁盘、网络等)抽象为Container。计算框架需要资源进行运算任务时需要向YARN申请Container， YARN按照特定的策略对资源进行调度进行Container的分配。
2. 资源隔离：
	YARN使用了轻量级资源隔离机制Cgroups进行资源隔离以避免相互干扰，一旦Container使用的资源量超过事先定义的上限值，就将其杀死。

YARN是对Mapreduce V1重构得到的，有时候也成为MapReduce V2。

YARN可以看成一个云操作系统，由一个ResourceManager和多个NodeManager组成， 它负责管理所有NodeManger上多维度资源， 并以Container(启动一个Container相当于启动一个进程)方式分配给应用程序启动ApplicationMaster(相当于主进程中运行逻辑) 或运行ApplicationMaster切分的各Task(相当于子进程中运行逻辑)。

# YARN体系架构

YARN架构如下图所示：

![YARN架构图](/imgs/Hadoop/yarn/architecture.png)

YARN总体上是Master/Slave结构，主要由ResourceManager、NodeManager、 ApplicationMaster和Container等几个组件构成。

- **ResourceManager(RM)**
	负责对各NodeManager上的资源进行统一管理和调度。将ApplicationMaster分配空闲的Container运行并监控其运行状态。对ApplicationMaster申请的资源请求分配相应的空闲Container。主要由两个组件构成：调度器和应用程序管理器：
    1. **调度器(Scheduler)**：调度器根据容量、队列等限制条件（如每个队列分配一定的资源，最多执行一定数量的作业等），将系统中的资源分配给各个正在运行的应用程序。调度器仅根据各个应用程序的资源需求进行资源分配，而资源分配单位是Container，从而限定每个任务使用的资源量。Scheduler不负责监控或者跟踪应用程序的状态，也不负责任务因为各种原因而需要的重启（由ApplicationMaster负责）。总之，调度器根据应用程序的资源要求，以及集群机器的资源情况，为应用程序分配封装在Container中的资源。
    2. **应用程序管理器(Applications Manager)**：应用程序管理器负责管理整个系统中所有应用程序，包括应用程序提交、与调度器协商资源以启动ApplicationMaster、监控ApplicationMaster运行状态并在失败时重新启动等，跟踪分给的Container的进度、状态也是其职责。

- **NodeManager (NM)**
	NodeManager是每个节点上的资源和任务管理器。它会定时地向ResourceManager汇报本节点上的资源使用情况和各个Container的运行状态；同时会接收并处理来自ApplicationMaster的Container 启动/停止等请求。

- **ApplicationMaster (AM)**
	用户提交的应用程序均包含一个ApplicationMaster，负责应用的监控，跟踪应用执行状态，重启失败任务等。ApplicationMaster是应用框架，它负责向ResourceManager协调资源，并且与NodeManager协同工作完成Task的执行和监控。MapReduce就是原生支持的一种框架，可以在YARN上运行Mapreduce作业。有很多分布式应用都开发了对应的应用程序框架，用于在YARN上运行任务，例如Spark，Storm等。

- **Container**
	Container是YARN中的资源抽象，它封装了某个节点上的多维度资源，如内存、CPU、磁盘、网络等，当ApplicationMaster向ResourceManager申请资源时，ResourceManager为ApplicationMaster返回的资源便是用Container 表示的。 YARN会为每个任务分配一个Container且该任务只能使用该Container中描述的资源。

# YARN应用工作流程

如下图所示用户向YARN中提交一个应用程序后，YARN将分两个阶段运行该应用程序：

1. 启动ApplicationMaster，如下步骤1~3；
2. 由ApplicationMaster创建应用程序为它申请资源并监控它的整个运行过程，直到运行完成，如下步骤4~7。

![YARN应用工作流程图](/imgs/Hadoop/yarn/yarn.png)

1. 用户向YARN中提交应用程序，其中包括ApplicationMaster程序、启动ApplicationMaster的命令、命令参数、用户程序等；事实上，需要准确描述运行ApplicationMaster的unix进程的所有信息。提交工作通常由YarnClient来完成。
2. ResourceManager为该应用程序分配第一个Container，并与对应的NodeManager通信，要求它在这个Container中启动ApplicationMaster；
3. ApplicationMaster首先向ResourceManager注册，这样用户可以直接通过ResourceManager査看应用程序的运行状态，运行状态通过 `AMRMClientAsync.CallbackHandler`的`getProgress()` 方法来传递给ResourceManager。 然后它将为各个任务申请资源，并监控它的运行状态，直到运行结束，即重复步骤4〜7；
4. ApplicationMaster采用轮询的方式通过RPC协议向ResourceManager申请和领取资源；资源的协调通过`AMRMClientAsync`异步完成,相应的处理方法封装在`AMRMClientAsync.CallbackHandler`中。
5. 一旦ApplicationMaster申请到资源后，便与对应的NodeManager通信，要求它启动任务；通常需要指定一个`ContainerLaunchContext`，提供Container启动时需要的信息。
6. NodeManager为任务设置好运行环境(包括环境变量、JAR包、二进制程序等)后，将任务启动命令写到一个脚本中，并通过运行该脚本启动任务；
7. 各个任务通过某个RPC协议向ApplicationMaster汇报自己的状态和进度，以让ApplicationMaster随时掌握各个任务的运行状态，从而可以在任务失败时重新启动任务；ApplicationMaster与NodeManager的通信通过`NMClientAsync object`来完成，容器的所有事件通过`NMClientAsync.CallbackHandler`来处理。例如启动、状态更新、停止等。
8. 应用程序运行完成后，ApplicationMaster向ResourceManager注销并关闭自己。

# YARN资源调度模型

YARN提供了一个资源管理平台能够将集群中的资源统一进行管理。所有节点上的多维度资源都会根据申请抽象为一个个Container。

YARN采用了双层资源调度模型：

- ResourceManager中的资源调度器将资源分配给各个ApplicationMaster：资源分配过程是异步的。资源调度器将资源分配给一个应用程序后，它不会立刻push给对应的ApplicationMaster，而是暂时放到一个缓冲区中，等待ApplicationMaster通过周期性的心跳主动来取；
- ApplicationMaster领取到资源后再进一步分配给它内部的各个任务：不属于YARN平台的范畴，由用户自行实现。

也就是说，ResourceManager分配集群资源的时候，以抽象的Container形式分配给各应用程序，至于应用程序的子任务如何使用这些资源，由应用程序自行决定。

YARN目前采用的资源分配算法有三种。但真实的调度器实现中还对算法做了一定程度的优化。

1. Capacity Scheduler：该调度器用于在共享、多租户（multi-tenant）的集群环境中运行Hadoop应用，对运营尽可能友好的同时最大化吞吐量和效用。
	该调度器保证共享集群的各个组织能够得到容量的保证，同时可以超额使用集群中暂时没有人使用的资源。Capacity Scheduler为了实现这些目标，抽象了queue的概念，queue通常由管理员配置。为了进一步细分容量的使用，调度器支持层级化的queue（hierarchical queues），使得在特定组织内部，可以进一步有效利用集群资源。
    Capacity调度器支持的一些特性如下：
    - 层级队列（Hierarchical Queues）
    - 容量保证
    - 安全性：每个队列都有队列的访问权限控制（ACL）
    - 弹性： 空闲资源可以额外分配给任何需要的队列
    - 多租户
    - 基于资源的调度（resouce-based scheduling): 对资源敏感的应用程序，可以有效地控制资源情况
    - 支持用户（组）到queue的映射：基于用户组提交作业到对应queue。
    - 运营支持：支持运行时配置队列的容量，ACL等。也可以在运行时停止queue阻止进一步往queue提交作业。
2. Fair Scheduler：公平调度FAIR，该算法的思想是尽可能地公平调度，即已分配资源量少的优先级高。也就是说，在考虑如何分配资源时，调度器尽可能使得每个应用程序都能够得到大致相当的资源。默认情况下，公平性只通过内存来衡量，但是可以配置成内存和CPU。 
这种策略使得运行时间短的应用能够尽快结束，而不至于在等待资源时被饿死。另外，也可以为应用程序配置优先级，优先级用于决定资源使用量的占比。


# 参考文献
>[Hadoop 新MapReduce 框架Yarn 详解](https://www.ibm.com/developerworks/cn/opensource/os-cn-hadoop-yarn/)
>[理解Hadoop YARN架构](http://blog.csdn.net/bingduanlbd/article/details/51880019)



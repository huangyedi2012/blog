---
title: Python split保留分隔符
date: 2017-03-31 16:21:37
categories: python
tags:
- python
- split
toc: false
---

python 文本或句子切割，并保留分隔符

<!-- more -->

主要思想，利用正则表达式re.split() 分割，同时利用re.findall() 查找分隔符，而后将二者链接即可。

``` python
# coding: utf-8

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import re


def my_split(str,sep=u"要求\d+|岗位\S+"):  # 分隔符可为多样的正则表达式
    wlist = re.split(sep,str)
    sepword = re.findall(sep,str)
    sepword.insert(0," ") # 开头（或末尾）插入一个空字符串，以保持长度和切割成分相同
    wlist = [ x+y for x,y in zip(wlist,sepword) ] # 顺序可根据需求调换
    return wlist



if __name__ == "__main__":
    inputstr = "岗位：学生： \n要求1.必须好好学习。\n要求2.必须踏实努力。\n要求3.必须求实上进。"
    res = my_split(inputstr)
    print '\n'.join(res)
```

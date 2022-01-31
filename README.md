# how-to-optimize-gemm
RowMajor gemm optimization

## 编译和运行

1. 准备 armv7/aarch64 linux 开发环境，树莓派/rk3399/aws arm server 都可以。
2. clone 代码后，修改`makefile`的`OLD`和`NEW`。首次运行需要改成同一个代码版本，例如
```
OLD    := MMult_4x4_8
NEW   := MMult_4x4_8
```

3. 默认情况下`ARCH := native`。直接编译运行即可
```
make run
```

4. gflops 结果在对应`.m` 文件，用`plot.py`可绘制相关折线图。

## fp32 gemm
此项目基于 [blis-lab](https://github.com/flame/blislab) 文档和[项目](https://github.com/flame/how-to-optimize-gemm)实现，与原作区别在于：

1. 原作为**列主序**`x86 SSE`代码。考虑到移动端卷积优化一般使用`arm`架构芯片，本项目是基于`arm64`版**行主序**`gemm`优化；
2. 原作没有做 k 维拆解，也没有太细致的分块，离 CPU 极限差距不小。本项目目前最新的`MMult_4x4_17.c`最高可达 9.9gflops，相当于 CPU 峰值的 70%；
3. 本项目没有处理边界问题，只考虑 MNK 均为 4 的倍数的情况；`sub_kernel`也只写了最简单的一种汇编。实用需要简单调整一下；
4. 绘图方面扔掉了冗长的 `octave`（arm linux 配置一次环境太麻烦），改用 `python plot`。


本系列优化中文教程在

1. [知乎 GEMM 入门](https://zhuanlan.zhihu.com/p/65436463)
2. [知乎 GEMM caching](https://zhuanlan.zhihu.com/p/69700540) (最后一节增加了理论支撑哦)
3. [ARMv7 4x4kernel 懒人优化小实践](https://zhuanlan.zhihu.com/p/333799799)


## int8 gemm
自 [知乎 GEMM 入门](https://zhuanlan.zhihu.com/p/65436463) 发布后，有不少同学问如何写一个 int8 gemm。本座写好了~~~

[chgemm](https://github.com/tpoisonooo/chgemm) 是个可用的 int8 gemm 库。相对于本教程中的代码，区别在于:
1. 处理了边界问题，不像教程里只考虑尺寸为 4 的倍数的情况;
2. int8 最高达到了 18.6 gflops（相对 fp32 理论极限只有14.3，gemmlowp大约 12-14gflops）;
3. 基于对称量化原理，输入数值范围必须在 \[-127, +127\]，不能出现 -128；

[chgemm](https://github.com/tpoisonooo/chgemm)已合入[ncnn](https://github.com/tencent/ncnn) INT8 卷积。

## cuda 版本

切换到  [cuda](https://github.com/tpoisonooo/how-to-optimize-gemm/tree/cuda) 分支。

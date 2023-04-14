# HavSampling Plugin

**Table Of Contents**

- [Description](#description)
  * [Structure](#structure)
- [Parameters](#parameters)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

The `HavSamplingPlugin` performs the sampling operation of PointNet++ to extract the centre of local regions.

This plugin allows you to do point-based 3D detector inference in TensorRT.

### Structure

The `HavSamplingPlugin` takes 1 inputs; `raw_xyz`.

`raw_xyz`
The input raw points from a point cloud. The shape of this tensor is `[B, N, C]`, where `B` is batch size, `N` is the
number of points in a point cloud frame, and `C` is the number of channels for each point.

The `HavSamplingPlugin` generates the following 1 outputs:

`indices`
The inputs indices of grouped points. The shape of this tensor is `[B, M]`, where `M` is the sizeof of sampled subset point.


## Parameters

The parameters are defined below and consists of the following attributes:

| Type                  | Parameter           | Description
|-----------------------|---------------------|--------------------------------------------------------
| `int`                 | `sample_num`           | the number of points should be sampled.
| `[float,float,float]` | `voxel_size`           | the initial maximum voxel size to quantize points.
| `float`               | `tolerance`           | the convergence radius for adaptive voxel searching.
| `int`                 | `max_iter`           | the maximum iteration time for adaptive voxel searching.

## Additional resources
This is our recent research and relative paper will attach here after review.
## License

For terms and conditions for use, reproduction, and distribution, see
the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html)
documentation.

## Changelog
Apr 2023
This is the first release of this `README.md` file.

## Known issues

There are no known issues in this plugin.

## Limitation
1. set_mask_kernel use atomic add to find next place to fill the output.
But the order can not preserve across different inference even using the same input, since
the execution orders of threads are variant. Use prefix sum to find the index in output.
2. do_while loops need wait for GPU in each iteration. dynamic parallelism can solve this problem.
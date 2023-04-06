# ballQuery Plugin

**Table Of Contents**

- [Description](#description)
    * [Structure](#structure)
- [Parameters](#parameters)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

The `ballQueryPlugin` performs the grouping operation of PointNet++ to extract local regions from raw points.

This plugin allows you to do point-based 3D detector inference in TensorRT.

### Structure

The `ballQueryPlugin` takes 2 inputs; `raw_xyz`, and `query_xyz`.

`raw_xyz`
The input raw points from a point cloud. The shape of this tensor is `[B, N, C]`, where `B` is batch size, `N` is the
number of points in a point cloud frame, and `C` is the number of channels for each point.

`query_xyz`
The group center to query inputs. The shape of this tensor is `[B, M, C]`.

The `ballQueryPlugin` generates the following 1 outputs:

`indices`
The inputs indices of grouped points. The shape of this tensor is `[B, M, K, C]`, where `K` is the maximum number of neighbors. 


## Parameters

The parameters are defined below and consists of the following attributes:

| Type             | Parameter           | Description
|------------------|---------------------|--------------------------------------------------------
| `int`            | `nsample`           | Maximum number of local regions, i.e., M.
| `float`          | `raduis`            | Maximum range for a region center to find neighbors.

## Additional resources

## License

For terms and conditions for use, reproduction, and distribution, see
the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html)
documentation.

## Changelog
Apr 2023
This is the first release of this `README.md` file.

## Known issues

There are no known issues in this plugin.

# FPSSampling Plugin

**Table Of Contents**

- [Description](#description)
  * [Structure](#structure)
- [Parameters](#parameters)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

The `FPSSamplingPlugin` performs the sampling operation of PointNet++ to extract the centre of local regions.

This plugin allows you to do point-based 3D detector inference in TensorRT.

### Structure

The `FPSSamplingPlugin` takes 1 inputs; `raw_xyz`.

`raw_xyz`
The input raw points from a point cloud. The shape of this tensor is `[B, N, C]`, where `B` is batch size, `N` is the
number of points in a point cloud frame, and `C` is the number of channels for each point.

The `FPSSamplingPlugin` generates the following 1 outputs:

`indices`
The inputs indices of grouped points. The shape of this tensor is `[B, M]`, where `M` is the sizeof of sampled subset point.


## Parameters

The parameters are defined below and consists of the following attributes:

| Type             | Parameter           | Description
|------------------|---------------------|--------------------------------------------------------
| `int`            | `nsample`           | the number of points should be sampled.

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

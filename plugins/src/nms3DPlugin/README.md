# nms3D Plugin

**Table Of Contents**

- [Description](#description)
    * [Structure](#structure)
- [Parameters](#parameters)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

The `nms3DPlugin` performs the 3D NMS postprocessing for all 3D detection results.

This plugin allows you to do 3D detector inference in TensorRT.

### Structure

The `nms3DPlugin` takes 2 inputs; `batch_pred_scores`, and `batch_pred_boxes`.

`batch_pred_scores`
The predicted scores of objects. The shape of this tensor is `[B, N, 1]`, where `B` is batch size, `N` is the
number of output boxes.

`batch_pred_boxes`
The predicted boxes of objects. The shape of this tensor is `[B, N, 7+C+1]`, where `B` is batch size, `N` is the
number of output boxes, `7` is `x,y,z,l,w,h,yaw`, and `C` is the additional features of box, and `1` is the predicted
cls.

The `nms3DPlugin` generates the following 3 outputs:

`final_scores`
The final scores after NMS. The shape of this tensor is `[B, M, 1]`, where `M` is the maximum number of output boxes.

`final_boxes`
The final boxes after NMS. The shape of this tensor is `[B, M, 7+C+1]`.

`num_boxes`
This indicates how many boxes are valid in `final_boxes` and `final_scores`, since the NMS results of different batch have different number of output.
The shape of this tensor is `[B, 1]`.

## Parameters
The parameters are defined below and consists of the following attributes:

| Type    | Parameter           | Description
|---------|---------------------|--------------------------------------------------------
| `float` | `score_thresh`           | filter out low confidence prediction.
| `float` | `nms_iou_thresh`           | remove a box if its overlapping with high-quality objects more than this threshold.
| `int`   | `max_num_post_nms`           | the maximum number of output after NMS.

## Additional resources

## License
[
For terms and conditions for use, reproduction, and distribution, see]()
the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html)
documentation.

## Changelog

Apr 2023
This is the first release of this `README.md` file.

## Known issues

There are no known issues in this plugin.

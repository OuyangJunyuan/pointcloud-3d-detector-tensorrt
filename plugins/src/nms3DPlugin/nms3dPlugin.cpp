//
// Created by nrsl on 23-5-1.
//

#include "nms3D.h"
#include "nms3DPlugin.h"


namespace nvinfer1::plugin {
int32_t NMSBEVPlugin::initialize() noexcept { return 0; }

void NMSBEVPlugin::terminate() noexcept {}

int32_t NMSBEVPlugin::enqueue(cudaStream_t stream) noexcept {

    nms3DInference(
            in.boxes.ptr, in.scores.ptr,
            out.final_boxes.ptr, out.final_scores.ptr, out.num_valid.ptr,
            attr.score_threshold, attr.iou_threshold, attr.num_max_nms,
            def.num_batch, def.num_box, def.num_box_feat, def.num_sort_temp,
            ws.valid_indices.ptr, ws.valid_scores.ptr, ws.valid_nums.ptr, ws.valid_ind_start.ptr, ws.valid_ind_end.ptr,
            ws.sort_temp_memory.ptr, ws.sorted_indices.ptr, ws.sorted_scores.ptr,
            stream
    );
    return 0;

}


} // namespace nvinfer1

# memcheck
no memory error here.
```shell
========= CUDA-MEMCHECK
['DisentangledAttention_TRT', 'CustomEmbLayerNormPluginDynamic', 'CustomEmbLayerNormPluginDynamic', 'CustomEmbLayerNormPluginDynamic', 'CustomFCPluginDynamic', 'CustomGeluPluginDynamic', 'GroupNormalizationPlugin', 'RnRes2Br1Br2c_TRT', 'RnRes2Br1Br2c_TRT', 'RnRes2Br2bBr2c_TRT', 'RnRes2Br2bBr2c_TRT', 'RnRes2FullFusion_TRT', 'SingleStepLSTMPlugin', 'CustomSkipLayerNormPluginDynamic', 'CustomSkipLayerNormPluginDynamic', 'CustomSkipLayerNormPluginDynamic', 'CustomSkipLayerNormPluginDynamic', 'CustomQKVToContextPluginDynamic', 'CustomQKVToContextPluginDynamic', 'CustomQKVToContextPluginDynamic', 'DLRM_BOTTOM_MLP_TRT', 'SmallTileGEMM_TRT', 'RNNTEncoderPlugin', 'NMSBEV', 'HAVSampling', 'HAVSamplingQ', 'BallQuery', 'GridBallQuery', 'FPSampling']
load config from file: configs/iassd/iassd_hvcsx1_4x8_80e_kitti_3cls.py
load config from module: configs.iassd.iassd_hvcsx1_4x8_80e_kitti_3cls
(8, 16384, 4)
9975.075960159302
========= ERROR SUMMARY: 0 errors
```
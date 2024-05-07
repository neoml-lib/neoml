"""Runs standard backend tests from ONNX on neoml.Onnx backend
"""
import neoml
import unittest
import onnx.backend.test

pytest_plugins = "onnx.backend.test.report"

backend_test = onnx.backend.test.runner.Runner(neoml.Onnx, __name__)

# Exclude CUDA tests for the sake of less testing time
backend_test.exclude('_cuda')

# OnnxBackendNodeModelTest (tests for single ONNX node)

# NeoOnnx doesn't support any training-related things
backend_test.exclude('test_adagrad_')
backend_test.exclude('test_adam_')
backend_test.exclude('test_nesterov_momentum_')
backend_test.exclude('test_training_')
# Operators not supported by NeoOnnx at all
backend_test.exclude('test_acos_')  # Acos
backend_test.exclude('test_acosh_')  # Acosh
backend_test.exclude('test_and2d_')  # And
backend_test.exclude('test_and3d_')  # And
backend_test.exclude('test_and4d_')  # And
backend_test.exclude('test_and_')  # And
backend_test.exclude('test_argmin_')  # ArgMin
backend_test.exclude('test_asin_')  # Asin
backend_test.exclude('test_asinh_')  # Asinh
backend_test.exclude('test_atan_')  # Atan
backend_test.exclude('test_atanh_')  # Atanh
backend_test.exclude('test_basic_convinteger_')  # ConvInteger
backend_test.exclude('test_bernoulli_')  # Bernoulli, RandomUniformLike
backend_test.exclude('test_bitshift_')  # BitShift
backend_test.exclude('test_bitwise_and_')  # BitwiseAnd
backend_test.exclude('test_bitwise_not_')  # BitwiseNot
backend_test.exclude('test_bitwise_or_')  # BitwiseOr
backend_test.exclude('test_bitwise_xor_')  # BitwiseXor
backend_test.exclude('test_blackmanwindow_')  # BlackmanWindow (or Cos for expanded version)
backend_test.exclude('test_castlike_')  # CastLike
backend_test.exclude('test_ceil_')  # Ceil
backend_test.exclude('test_celu_(?!expanded)')  # Celu
backend_test.exclude('test_center_crop_pad')  # CenterCropPad (or Max for expanded version)
backend_test.exclude('test_col2im_')  # Col2Im
backend_test.exclude('test_compress_')  # Compress
backend_test.exclude('test_convinteger_')  # ConvInteger
backend_test.exclude('test_cos_')  # Cos
backend_test.exclude('test_cosh_')  # Cosh
backend_test.exclude('deform_conv')  # DeformConv
backend_test.exclude('test_depthtospace_crd_mode_')  # DepthToSpace in CRD mode
backend_test.exclude('test_dequantizelinear_')  # DequantizeLinear
backend_test.exclude('test_det_')  # Det
backend_test.exclude('test_dft_')  # DFT
backend_test.exclude('test_dynamicquantizelinear_')  # DynamicQuantizeLinear
backend_test.exclude('test_einsum_')  # EinSum
backend_test.exclude('test_elu_[a-z0-9_]*expanded')  # CastLike
backend_test.exclude('test_eyelike_')  # EyeLike
backend_test.exclude('test_floor_')  # Floor
backend_test.exclude('test_gather_elements_')  # GatherElements
backend_test.exclude('test_gathernd_')  # GatherND
backend_test.exclude('test_greater_[a-z0-9_]*expanded_')  # Or
backend_test.exclude('test_gridsample_') # GridSample
backend_test.exclude('test_group_normalization_')  # GroupNormalization
backend_test.exclude('test_gru_')  # GRU
backend_test.exclude('test_hammingwindow_')  # HammingWindow (or Cos for expanded)
backend_test.exclude('test_hannwindow_')  # HannWindow (or Cos for expanded)
backend_test.exclude('test_hardmax_')  # HardMax
backend_test.exclude('test_hardsigmoid_[a-z0-9_]*expanded')  # Max, Min and CastLike
backend_test.exclude('test_if_')  # If
backend_test.exclude('test_isinf_')  # IsInf
backend_test.exclude('test_isnan_')  # IsNan
backend_test.exclude('test_layer_normalization_')  # LayerNormalization (or Size, Reciprocal for expanded)
backend_test.exclude('test_leakyrelu_[a-z0-9_]*expanded_')  # CastLike
backend_test.exclude('test_less_equal_[a-z0-9_]*expanded_')  # Or
backend_test.exclude('test_loop')  # Loop
backend_test.exclude('test_lppool_')  # LpPool
backend_test.exclude('test_matmulinteger_')  # MatMulInteger
backend_test.exclude('test_max_')  # Max
backend_test.exclude('test_maxunpool_')  # MaxUnpool
backend_test.exclude('test_mean_')  # Mean
backend_test.exclude('test_melweightmatrix_')  # MelWeightMatrix
backend_test.exclude('test_min_')  # Min
backend_test.exclude('test_mish_')  # Mish
backend_test.exclude('test_mod_')  # Mod
backend_test.exclude('test_mvn_(?!expanded)')  # MeanVarianceNormalization
backend_test.exclude('test_nllloss_')  # NegativeLogLikelihoodLoss, GatherElements
backend_test.exclude('test_nonmaxsuppression_')  # NonMaxSuppression
backend_test.exclude('test_optional_get_element_')  # OptionalGetElement
backend_test.exclude('test_optional_has_element_')  # OptionalHasElement
backend_test.exclude('test_or2d_')  # Or
backend_test.exclude('test_or3d_')  # Or
backend_test.exclude('test_or4d_')  # Or
backend_test.exclude('test_or_')  # Or
backend_test.exclude('test_prelu_')  # PRelu
backend_test.exclude('test_qlinearconv_')  # QLinearConv
backend_test.exclude('test_qlinearmatmul_')  # QLinearMatMul
backend_test.exclude('test_quantizelinear_')  # QuantizeLinear
# backend_test.exclude('test_range_[a-z0-9_]*_expanded_')  # Ceil, Loop
backend_test.exclude('test_reciprocal_')  # Reciprocal
backend_test.exclude('test_reduce_l1_')  # ReduceL1
backend_test.exclude('test_reduce_log_sum_')  # ReduceLogSum
backend_test.exclude('test_reduce_log_sum_exp_')  # ReduceLogSumExp
backend_test.exclude('test_reduce_prod_')  # ReduceProd
backend_test.exclude('test_reduce_sum_square_')  # ReduceSumSquare
backend_test.exclude('test_relu_[a-z0-9_]*expanded_')  # Max, CastLike
backend_test.exclude('test_reversesequence_')  # ReverseSequence
backend_test.exclude('_rnn_')  # RNN
backend_test.exclude('test_roialign_')  # RoiAlign
backend_test.exclude('test_round_')  # Round
backend_test.exclude('test_scan9_')  # Scan
backend_test.exclude('test_scan_')  # Scan
backend_test.exclude('test_scatter_')  # Scatter and ScatterElements
backend_test.exclude('test_sce_')  # SoftmaxCrossEntropyLoss, NegativeLogLikelihoodLoss
backend_test.exclude('test_selu_')  # Selu
backend_test.exclude('test_sequence_insert_')  # SequenceInsert
backend_test.exclude('test_sequence_map_')  # SequenceMap
backend_test.exclude('test_shrink_')  # Shrink
backend_test.exclude('test_sign_')  # Sign
backend_test.exclude('test_sin_')  # Sin
backend_test.exclude('test_sinh_')  # Sinh
backend_test.exclude('test_size_')  # Size
backend_test.exclude('test_softplus_')  # Softplus
backend_test.exclude('test_softsign_')  # Softsign
backend_test.exclude('test_spacetodepth_')  # SpaceToDepth
backend_test.exclude('test_split_to_sequence_')  # SplitToSequence
backend_test.exclude('test_stft_')  # STFT
backend_test.exclude('test_strnormalizer_')  # StringNormalizer
backend_test.exclude('test_tan_')  # Tan
backend_test.exclude('test_tfidfvectorizer_')  # TfIdfVectorizer
backend_test.exclude('test_thresholdedrelu_')  # ThresholdedRelu
backend_test.exclude('test_tile_')  # Tile
backend_test.exclude('test_top_k_')  # TopK
backend_test.exclude('test_tril_')  # Trilu
backend_test.exclude('test_triu_')  # Trilu
backend_test.exclude('test_unique_')  # Unique
backend_test.exclude('test_xor2d_')  # Xor
backend_test.exclude('test_xor3d_')  # Xor
backend_test.exclude('test_xor4d_')  # Xor
backend_test.exclude('test_xor_')  # Xor

# Operators partly supported (with comments, why test fails)
# TODO: analyze nodes that are implemented but can't be tested...

#
backend_test.exclude('test_argmax_[a-z0-9_]*example[a-z0-9_]select_last_index')  # NeoOnnx doesn't support last index in ArgMax
backend_test.exclude('test_averagepool_2d_pads_cpu')  # NeoOnnx doesn't support excluded pads for average pooling
backend_test.exclude('test_averagepool_2d_precomputed_pads_cpu')  # NeoOnnx doesn't support excluded pads for average pooling
backend_test.exclude('test_averagepool_[a-z0-9_]*_same_')  # NeoOnnx doesn't support pads for average pooling
backend_test.exclude('test_averagepool_[a-z0-9_]*_ceil_')  # NeoOnnx doesn't support ceiling for average pooling
backend_test.exclude('test_averagepool_[a-z0-9_]*_dilations_')  # NeoOnnx doesn't support dilations for average pooling
backend_test.exclude('test_averagepool_1d_')  # NeoOnnx supports only 2d average pooling
backend_test.exclude('test_averagepool_3d_')  # NeoOnnx supports only 2d average pooling
backend_test.exclude('test_basic_conv_')  # NeoOnnx doesn't support trained filters as input
backend_test.exclude('test_batchnorm_')  # NeoOnnx doesn't support trained coeffs as net input
# NeoOnnx supports only INT32 <-> FLOAT32 cast (which isn't tested by ONNX tests anyway...)
backend_test.exclude('test_cast_')
# NeoOnnx doesn't support clip thresholds as net inputs but we will try to enable expanded tests
backend_test.exclude('test_clip_.*bounds_cpu')
backend_test.exclude('test_clip_.*max_cpu')
backend_test.exclude('test_clip_.*min_cpu')
backend_test.exclude('test_clip(_example)?_cpu')
backend_test.exclude('test_constant_pad_')  # NeoOnnx doesn't support padding sizes as inputs
backend_test.exclude('test_constantofshape_')  # NeoOnnx doesn't support tensor size as input
backend_test.exclude('test_conv_')  # NeoOnnx doesn't support trained filters as input
backend_test.exclude('test_convtranspose_')  # NeoOnnx doesn't support trained filters as input
backend_test.exclude('test_cumsum_')  # NeoOnnx doesn't support axis index as input
backend_test.exclude('test_dropout_default_mask_')  # NeoOnnx doesn't support dropout mask as output
backend_test.exclude('test_dropout_default_ratio_')  # NeoOnnx doesn't support dropout rate as input
backend_test.exclude('test_edge_pad_')  # NeoOnnx doesn't support padding sizes as input
backend_test.exclude('test_equal_string_')  # NeoOnnx doesn't support tensor with strings
backend_test.exclude('test_expand_')  # NeoOnnx doesn't support shape as input
backend_test.exclude('test_gemm_')  # NeoOnnx supports only specific case when it's an FC layer (with constant weights)
backend_test.exclude('test_identity_opt_')  # NeoOnnx doesn't support optional values as inputs
backend_test.exclude('test_identity_sequence_')  # NeoOnnx doesn't support sequences values as inputs
backend_test.exclude('test_instancenorm_')  # NeoOnnx doesn't support scales as input
backend_test.exclude('test_lstm_')  # NeoOnnx doesn't support trained weights as inputs
backend_test.exclude('test_maxpool_1d_')  # NeoOnnx supports only 2d max pooling
backend_test.exclude('test_maxpool_2d_ceil_')  # NeoOnnx doesn't support ceil in maxpool
backend_test.exclude('test_maxpool_2d_dilations_')  # NeoOnnx doesn't support dilations in maxpool
backend_test.exclude('test_maxpool_2d_uint8_')  # NeoOnnx doesn't support maxpool over integer data
backend_test.exclude('test_maxpool_3d_')  # NeoOnnx supports only 2d max pooling
backend_test.exclude('test_maxpool_with_argmax_')  # NeoOnnx doesn't support indices as additional output
backend_test.exclude('test_mvn_expanded_ver18')  # NeoOnnx doesn't suppoprt value_ints atrr in Constant node
backend_test.exclude('test_nonzero_')  # NeoOnnx supports nonzero only over constant tensors
backend_test.exclude('test_onehot_')  # NeoOnnx supports only constant depth in OneHot
backend_test.exclude('test_pow_')  # NeoOnnx doesn't support power of the exponent as input
backend_test.exclude('test_range')  # NeoOnnx supports Range only over constant data
backend_test.exclude('test_reduce_l2_')  # NeoOnnx support ReduceL2 only when axes are fixed within data
backend_test.exclude('test_reduce_max_(?!default_axes_)')  # NeoOnnx support ReduceMax only when axes are default
backend_test.exclude('test_reduce_mean_')  # NeoOnnx support ReduceMean only when axes are fixed within data
backend_test.exclude('test_reduce_min_(?!default_axes_)')  # NeoOnnx support ReduceMin only when axes are default
backend_test.exclude('test_reduce_sum_')  # NeoOnnx support ReduceSum only when axes are fixed within data
backend_test.exclude('test_reflect_pad_')  # NeoOnnx doesn't support padding sizes as input
backend_test.exclude('test_reshape_')  # NeoOnnx doesn't support shape as input
backend_test.exclude('test_resize_')  # NeoOnnx doesn't support sizes or scales as inputs
backend_test.exclude('test_scatternd_add_')  # NeoOnnx doesn't support non-trivial reduction in ScatterND
backend_test.exclude('test_scatternd_max_')  # NeoOnnx doesn't support non-trivial reduction in ScatterND
backend_test.exclude('test_scatternd_min_')  # NeoOnnx doesn't support non-trivial reduction in ScatterND
backend_test.exclude('test_scatternd_multiply_')  # NeoOnnx doesn't support non-trivial reduction in ScatterND
backend_test.exclude('test_slice_')  # NeoOnnx doesn't support sizes, stars, ends or axes as inputs
backend_test.exclude('test_split_.*_uneven_')  # NeoOnnx doesn't support uneven splits
backend_test.exclude('test_split_variable_parts_')  # NeoOnnx doesn't support split sizes as inputs
backend_test.exclude('test_split_zero_size_')  # NeoOnnx doesn't support zero-sized splits
backend_test.exclude('test_squeeze_')  # NeoOnnx doesn't support axes as inputs
backend_test.exclude('test_unsqueeze_')  # NeoOnnx doesn't support axes as inputs
backend_test.exclude('test_upsample_')  # NeoOnnx doesn't support scales as input
backend_test.exclude('test_wrap_pad_')  # NeoOnnx doesn't support padding sizes as input

# OnnxBackendRealModelTest (a bunch of models from the model zoo)
backend_test.exclude('test_bvlc_alexnet_')  # Contains groupped convolution
backend_test.exclude('test_inception_v1_')  # Contains average pooling with padding
backend_test.exclude('test_inception_v2_')  # Contains average pooling with padding
backend_test.exclude('test_shufflenet_')  # Contains groupped convolution

# OnnxBackendSimpleModelTest (some synthetic test models)
# Models containing unsupported operators
backend_test.exclude('test_gradient_of_add_')  # Gradient
backend_test.exclude('test_sequence_model1_')  # SequenceInsert, SequenceEmpty, SequenceAt
backend_test.exclude('test_sequence_model2_')  # SequenceConstruct, SequenceErase, SequenceAt
backend_test.exclude('test_sequence_model3_')  # SequenceConstruct, SequenceInsert, SequenceErase, SequenceAt
backend_test.exclude('test_sequence_model4_')  # ConcatFromSequence, SequenceConstruct
backend_test.exclude('test_sequence_model5_')  # ConcatFromSequence, SequenceConstruct
backend_test.exclude('test_sequence_model6_')  # SplitToSequence, SequenceLength
backend_test.exclude('test_sequence_model7_')  # SplitToSequence, SequenceAt
backend_test.exclude('test_sequence_model8_')  # SplitToSequence, SequenceLength
backend_test.exclude('test_strnorm_model_')  # StringNormalizer

# OnnxBackendPyTorchConvertedModelTest

# backend_test.exclude('_sparse_')  # NeoOnnx doesn't support sparse data
backend_test.exclude('test_AvgPool3d_')  # NeoOnnx supports only 2d avg pooling
backend_test.exclude('test_Conv[a-z0-9_]*group')  # NeoOnnx doesn't support groupped convolution
backend_test.exclude('test_Conv[a-z0-9_]*multiplier')  # NeoOnnx doesn't support groupped convolution
backend_test.exclude('test_Conv3d[a-z0-9_]*dilated')  # NeoOnnx doesn't support dilation in 3d convolution
backend_test.exclude('test_MaxPool1d_')  # NeoOnnx supports only 2d max pooling
backend_test.exclude('test_MaxPool2d[a-z0-9_]*dilation')  # NeoOnnx doesn't support dilation in MaxPool
backend_test.exclude('test_MaxPool3d_')  # NeoOnnx supports only 2d max pooling
backend_test.exclude('test_PReLU')  # Contains PRelu operator
backend_test.exclude('test_SELU')  # Contains Selu operator
backend_test.exclude('test_Softplus')  # Contains Softplus operator
backend_test.exclude('test_Softplus')  # Contains Softplus operator

# OnnxBackendPyTorchOperatorModelTest
backend_test.exclude('test_operator_add_')  # NeoOnnx doesn't support doubles which can't be casted to float
backend_test.exclude('test_operator_addconstant_')  # NeoOnnx doesn't support doubles which can't be casted to float
# NeoOnnx supports only specific case when it's an FC layer (with constant weights)
backend_test.exclude('test_operator_addmm_')
backend_test.exclude('test_operator_max_')  # Contains Max operator
backend_test.exclude('test_operator_maxpool_')  # NeoOnnx supports only 2d max pooling
backend_test.exclude('test_operator_min_')  # Contains Min operator
# NeoOnnx supports only specific case when it's an FC layer (with constant weights)
backend_test.exclude('test_operator_mm_')
backend_test.exclude('test_operator_pow_')  # NeoOnnx doesn't support power of the exponent as input
backend_test.exclude('test_operator_repeat_')  # Contains Tile operator
backend_test.exclude('test_operator_selu_')  # Contains Selu operator

# TODO: ALARM!!! Some failing tests for future research (no explanation for failure yet...)

backend_test.exclude('test_momentum_')  # Some WEIRD stuff happens here... (Default operator set is missing)
backend_test.exclude('test_ai_onnx_ml')  # Default version missing???

# TODO: ALARM!!! Run ALL the failing tests and fix all the asserts (it should be replaced with some exception)...

globals().update(backend_test.enable_report().test_cases)

if __name__ == '__main__':
	unittest.main()

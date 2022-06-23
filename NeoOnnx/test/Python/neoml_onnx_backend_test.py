"""Runs standard backend tests from ONNX on neoml.Onnx backend
"""
import neoml
import unittest
import onnx.backend.test

pytest_plugins = "onnx.backend.test.report"

backend_test = onnx.backend.test.runner.Runner(neoml.Onnx, __name__)

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
backend_test.exclude('test_bitshift_')  # BitShift
backend_test.exclude('test_ceil_')  # Ceil
backend_test.exclude('test_celu_')  # Celu
backend_test.exclude('test_compress_')  # Compress
backend_test.exclude('test_cos_')  # Cos
backend_test.exclude('test_cosh_')  # Cosh
backend_test.exclude('test_cumsum_')  # CumSum
backend_test.exclude('test_depthtospace_')  # DepthToSpace
backend_test.exclude('test_dequantizelinear_')  # DequantizeLinear
backend_test.exclude('test_det_')  # Det
backend_test.exclude('test_dynamicquantizelinear_')  # DynamicQuantizeLinear
backend_test.exclude('test_einsum_')  # EinSum
backend_test.exclude('test_eyelike_')  # EyeLike
backend_test.exclude('test_floor_')  # Floor
backend_test.exclude('test_gathernd_')  # GatherND
backend_test.exclude('test_greater_')  # Greater
backend_test.exclude('test_gru_')  # GRU
backend_test.exclude('test_hardmax_')  # HardMax
backend_test.exclude('test_isinf_')  # IsInf
backend_test.exclude('test_isnan_')  # IsNan
backend_test.exclude('test_less_')  # Less
backend_test.exclude('test_log_')  # Log
backend_test.exclude('test_logsoftmax_')  # LogSoftmax
backend_test.exclude('test_matmulinteger_')  # MatMulInteger
backend_test.exclude('test_max_')  # Max
backend_test.exclude('test_mean_')  # Mean
backend_test.exclude('test_min_')  # Min
backend_test.exclude('test_mod_')  # Mod
backend_test.exclude('test_mvn_')  # MeanVarianceNormalization
backend_test.exclude('test_neg_')  # Neg
backend_test.exclude('test_negative_log_likelihood_loss_')  # NegativeLogLikelihoodLoss
backend_test.exclude('test_nonmaxsuppression_')  # NonMaxSuppression
backend_test.exclude('test_not_')  # Not
backend_test.exclude('test_or2d_')  # Or
backend_test.exclude('test_or3d_')  # Or
backend_test.exclude('test_or4d_')  # Or
backend_test.exclude('test_or_')  # Or
backend_test.exclude('test_qlinearconv_')  # QLinearConv
backend_test.exclude('test_qlinearmatmul_')  # QLinearMatMul
backend_test.exclude('test_quantizelinear_')  # QuantizeLinear
backend_test.exclude('test_reciprocal_')  # Reciprocal
backend_test.exclude('test_reduce_l1_')  # ReduceL1
backend_test.exclude('test_reduce_l2_')  # ReduceL2
backend_test.exclude('test_reduce_log_sum_') # ReduceLogSum
backend_test.exclude('test_reduce_log_sum_exp_') # ReduceLogSumExp
backend_test.exclude('test_reduce_prod_')  # ReduceProd
backend_test.exclude('test_reduce_sum_square_')  # ReduceSumSquare
backend_test.exclude('test_reversesequence_')  # ReverseSequence
backend_test.exclude('_rnn_')  # RNN
backend_test.exclude('test_roialign_')  # RoiAlign
backend_test.exclude('test_round_')  # Round
backend_test.exclude('test_scan9_')  # Scan
backend_test.exclude('test_scan_')  # Scan
backend_test.exclude('test_scatter_')  # Scatter and ScatterElements
backend_test.exclude('test_scatternd_')  # ScatterND
backend_test.exclude('test_selu_')  # Selu
backend_test.exclude('_sequence_len')  # Selu
backend_test.exclude('test_shrink_')  # Shrink
backend_test.exclude('test_sign_')  # Sign
backend_test.exclude('test_sin_')  # Sin
backend_test.exclude('test_sinh_')  # Sinh
backend_test.exclude('test_size_')  # Size
backend_test.exclude('test_softplus_')  # Softplus
backend_test.exclude('test_softsign_')  # Softsign
backend_test.exclude('test_strnormalizer_')  # StringNormalizer
backend_test.exclude('test_tan_')  # Tan
backend_test.exclude('test_tfidfvectorizer_')  # TfIdfVectorizer
backend_test.exclude('test_thresholdedrelu_')  # ThreasholdedRelu
backend_test.exclude('test_tile_')  # Tile
backend_test.exclude('test_top_k_')  # TopK
backend_test.exclude('test_unique_')  # Unique
backend_test.exclude('test_xor2d_')  # Xor
backend_test.exclude('test_xor3d_')  # Xor
backend_test.exclude('test_xor4d_')  # Xor
backend_test.exclude('test_xor_')  # Xor
# Operators partly supported (with comments, why test fails)
# TODO: analyze nodes that are implemented but can't be tested...

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
# Models not supported by NeoOnnx
# TODO: analyze and write it down (with reason!)

# OnnxBackendPyTorchConvertedModelTest

backend_test.exclude('Conv[a-z_]*group')  # Groupped convolution
backend_test.exclude('Conv3d[a-z_]*dilated')  # 3d convolution with dilations
backend_test.exclude('_sparse_')  # Anything related to the sparse data
backend_test.exclude('test_LogSoftmax')  # Contains LogSoftmax operator
backend_test.exclude('test_PReLU')  # Contains PRelu operator
backend_test.exclude('test_SELU')  # Contains Selu operator
backend_test.exclude('test_Softmin')  # Contains Neg operator
# TODO: analyze and write down remaining failed tests (with reason!)

# OnnxBackendPyTorchOperatorModelTest

backend_test.exclude('test_operator_basic_')  # Contains Neg operator
backend_test.exclude('test_operator_max_')  # Contains Max operator
backend_test.exclude('test_operator_min_')  # Contains Min operator
backend_test.exclude('test_operator_params_')  # Contains Neg operator
backend_test.exclude('test_operator_repeat_')  # Contains Tile operator
backend_test.exclude('test_operator_selu_')  # Contains Selu operator
backend_test.exclude('test_operator_symbolic_override_nested_')  # Contains Neg operator
# TODO: analyze and write down remaining failed tests (with reason!)

globals().update(backend_test.enable_report().test_cases)

if __name__ == '__main__':
	unittest.main()

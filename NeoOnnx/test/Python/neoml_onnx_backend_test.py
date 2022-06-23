"""Runs standard backend tests from ONNX on neoml.Onnx backend
"""
import neoml
import unittest
import onnx.backend.test

pytest_plugins = "onnx.backend.test.report"

backend_test = onnx.backend.test.runner.Runner(neoml.Onnx, __name__)

# TODO: here we should disable tests which ain't gonna work in NeoOnnx anyway...

globals().update(backend_test.enable_report().test_cases)

if __name__ == '__main__':
	unittest.main()

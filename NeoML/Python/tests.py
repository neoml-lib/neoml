from unittest import TestCase
import neoml


class MathEngineTestCase(TestCase):
    def test_gpu_math_engine(self):

        check = False
        try:
            print(neoml.MathEngine.GpuMathEngine(666).info)
        except ValueError as err:
            check = True

        self.assertEqual(check, True)

        check = False
        try:
            print(neoml.MathEngine.GpuMathEngine(-666).info)
        except ValueError as err:
            check = True

        self.assertEqual(check, True)

        gpu = neoml.MathEngine.enum_gpu()

        index = 0
        for x in gpu:
            math_engine = neoml.MathEngine.GpuMathEngine(index)
            self.assertTrue(isinstance(math_engine, neoml.MathEngine.GpuMathEngine))
            index += 1

    def test_cpu_math_engine(self):

        math_engine = neoml.MathEngine.CpuMathEngine()
        self.assertTrue(isinstance(math_engine, neoml.MathEngine.CpuMathEngine))
        blob = neoml.Blob.vector(math_engine, 10, "int32")
        self.assertEqual(math_engine.peak_memory_usage, 40)


class BlobTestCase(TestCase):
    def test_blob(self):
        return


class SolverTestCase(TestCase):
    def test_NesterovGradient(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        solver = neoml.Solver.NesterovGradient(math_engine, learning_rate=0.6, l1=0.6, l2=0.6,
                                               moment_decay_rate=0.6, max_gradient_norm=0.6,
                                               second_moment_decay_rate=0.6, epsilon=0.6, ams_grad=True)

        self.assertAlmostEqual(solver.l1, 0.6)
        self.assertAlmostEqual(solver.l2, 0.6)
        self.assertAlmostEqual(solver.learning_rate, 0.6)
        self.assertAlmostEqual(solver.max_gradient_norm, 0.6)
        self.assertAlmostEqual(solver.moment_decay_rate, 0.6)
        self.assertAlmostEqual(solver.second_moment_decay_rate, 0.6)
        self.assertAlmostEqual(solver.epsilon, 0.6)
        self.assertEqual(solver.ams_grad, True)

    def test_AdaptiveGradient(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        solver = neoml.Solver.AdaptiveGradient(math_engine, learning_rate=0.6, l1=0.6, l2=0.6,
                                               moment_decay_rate=0.6, max_gradient_norm=0.6,
                                               second_moment_decay_rate=0.6, epsilon=0.6, ams_grad=True)

        self.assertAlmostEqual(solver.l1, 0.6)
        self.assertAlmostEqual(solver.l2, 0.6)
        self.assertAlmostEqual(solver.learning_rate, 0.6)
        self.assertAlmostEqual(solver.max_gradient_norm, 0.6)
        self.assertAlmostEqual(solver.moment_decay_rate, 0.6)
        self.assertAlmostEqual(solver.second_moment_decay_rate, 0.6)
        self.assertAlmostEqual(solver.epsilon, 0.6)
        self.assertEqual(solver.ams_grad, True)

    def test_SimpleGradient(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        solver = neoml.Solver.SimpleGradient(math_engine, learning_rate=0.6, l1=0.6, l2=0.6,
                                             moment_decay_rate=0.6, max_gradient_norm=0.6)

        self.assertAlmostEqual(solver.l1, 0.6)
        self.assertAlmostEqual(solver.l2, 0.6)
        self.assertAlmostEqual(solver.learning_rate, 0.6)
        self.assertAlmostEqual(solver.max_gradient_norm, 0.6)
        self.assertAlmostEqual(solver.moment_decay_rate, 0.6)


class LayersTestCase(TestCase):
    def test_argmax(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        dnn = neoml.Dnn.Dnn(math_engine)
        source = neoml.Source.Source(dnn, "source")

        argmax = neoml.Argmax.Argmax(source, dimension="channels", name="argmax")

        self.assertEqual(argmax.dimension, "channels")
        argmax.dimension = "batch_length"
        self.assertEqual(argmax.dimension, "batch_length")

    def test_attention_decoder(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        dnn = neoml.Dnn.Dnn(math_engine)
        source1 = neoml.Source.Source(dnn, "source1")
        source2 = neoml.Source.Source(dnn, "source2")

        scores = ["additive", "dot_product"]

        decoder = neoml.AttentionDecoder.AttentionDecoder((source1, source2), "additive", 16, 32, 64, "decoder")

        self.assertEqual(decoder.hidden_layer_size, 16)
        decoder.hidden_layer_size = 17
        self.assertEqual(decoder.hidden_layer_size, 17)

        self.assertEqual(decoder.output_object_size, 32)
        decoder.output_object_size = 33
        self.assertEqual(decoder.output_object_size, 33)

        self.assertEqual(decoder.output_seq_len, 64)
        decoder.output_seq_len = 65
        self.assertEqual(decoder.output_seq_len, 65)

        self.assertEqual(decoder.score, "additive")
        decoder.score = "dot_product"
        self.assertEqual(decoder.score, "dot_product")

    def test_conv(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        dnn = neoml.Dnn.Dnn(math_engine)
        source = neoml.Source.Source(dnn, "source1")
        conv = neoml.Conv.Conv(source, 16, (3, 3), (2, 2), (1, 1), (4, 4), False, "conv")

        self.assertEqual(conv.filter_count, 16)
        conv.filter_count = 17
        self.assertEqual(conv.filter_count, 17)

        self.assertEqual(conv.filter_size, (3, 3))
        conv.filter_size = (6, 6)
        self.assertEqual(conv.filter_size, (6, 6))

        self.assertEqual(conv.stride_size, (2, 2))
        conv.stride_size = (7, 7)
        self.assertEqual(conv.stride_size, (7, 7))

        self.assertEqual(conv.padding_size, (1, 1))
        conv.padding_size = (8, 8)
        self.assertEqual(conv.padding_size, (8, 8))

        self.assertEqual(conv.dilation_size, (4, 4))
        conv.padding_size = (9, 9)
        self.assertEqual(conv.padding_size, (9, 9))


class DnnTestCase(TestCase):
    def test_solver(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        dnn = neoml.Dnn.Dnn(math_engine)

        dnn.solver = neoml.Solver.NesterovGradient(math_engine)
        self.assertTrue(isinstance(dnn.solver, neoml.Solver.NesterovGradient))

        dnn.solver = neoml.Solver.AdaptiveGradient(math_engine)
        self.assertTrue(isinstance(dnn.solver, neoml.Solver.AdaptiveGradient))

        dnn.solver = neoml.Solver.SimpleGradient(math_engine)
        self.assertTrue(isinstance(dnn.solver, neoml.Solver.SimpleGradient))

    def test_initializer(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)

        random = neoml.Random.Random(0)

        dnn = neoml.Dnn.Dnn(math_engine, random)

        dnn.initializer = neoml.Initializer.Xavier(random)
        self.assertTrue(isinstance(dnn.initializer, neoml.Initializer.Xavier))

        dnn.initializer = neoml.Initializer.Uniform()
        self.assertTrue(isinstance(dnn.initializer, neoml.Initializer.Uniform))

    def test_math_engine(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        dnn = neoml.Dnn.Dnn(math_engine)
        self.assertTrue(isinstance(dnn.math_engine, neoml.MathEngine.CpuMathEngine))

    def test_properties(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        dnn = neoml.Dnn.Dnn(math_engine)
        source = neoml.Source.Source(dnn, "source")
        argmax = neoml.Argmax.Argmax(source, name="argmax")
        sink = neoml.Sink.Sink(argmax, "sink")

        self.assertTrue(len(dnn.input_layers), 1)
        self.assertTrue(len(dnn.layers), 3)
        self.assertTrue(len(dnn.output_layers), 1)

from unittest import TestCase
import os
import tempfile
import pickle
import itertools
import numpy as np
from scipy import sparse
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
    def test_pickle(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        a = np.ones((2, 3, 4, 5), dtype=np.int32)
        shape = (2, 3, 1, 4, 1, 1, 5)

        blob = neoml.Blob.asblob(math_engine, a, shape, False)

        dir = tempfile.mkdtemp()

        path = os.path.join(dir, 'blob.pickle')
        binary_file = open(path, mode='wb')
        pickle.dump(blob, binary_file)
        binary_file.close()

        binary_file = open(path, mode='rb')
        loaded_blob = pickle.load(binary_file)
        binary_file.close()

        os.remove(path)
        os.rmdir(dir)

        self.assertEqual(blob.shape, loaded_blob.shape)

    def test_load_store(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        a = np.ones((2, 3, 4, 5), dtype=np.int32)
        shape = (2, 3, 1, 4, 1, 1, 5)

        blob = neoml.Blob.asblob(math_engine, a, shape, False)

        dir = tempfile.mkdtemp()

        path = os.path.join(dir, 'blob.pickle')
        neoml.Blob.store(blob, path)

        loaded_blob = neoml.Blob.load(math_engine, path)

        os.remove(path)
        os.rmdir(dir)

        self.assertEqual(blob.shape, loaded_blob.shape)
 
    def test_copy(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)

        a = np.ones((4, 4, 4, 4), dtype=np.int32)
        shape = (4, 4, 1, 4, 4, 1, 1)
        blob = neoml.Blob.asblob(math_engine, a, shape, False)

        blob2 = blob.copy(math_engine)
        self.assertEqual(blob2.shape, blob.shape)

        a2 = blob2.asarray()

        self.assertEqual(a2.shape, a.shape)

    def test_asblob(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)

        float_array = np.ones((2, 5, 7, 16), dtype=np.float32)
        shape = (1, 2, 1, 5, 7, 1, 16)

        float_blob = neoml.Blob.asblob(math_engine, float_array, shape, False)
        self.assertEqual(float_blob.shape, shape)
        self.assertEqual(float_blob.batch_len, 1)
        self.assertEqual(float_blob.batch_width, 2)
        self.assertEqual(float_blob.list_size, 1)
        self.assertEqual(float_blob.height, 5)
        self.assertEqual(float_blob.width, 7)
        self.assertEqual(float_blob.depth, 1)
        self.assertEqual(float_blob.channels, 16)
        self.assertEqual(float_blob.size, 2 * 5 * 7 * 16)
        self.assertEqual(float_blob.object_count, 2)
        self.assertEqual(float_blob.object_size, 5 * 7 * 16)

        blob_float_array = float_blob.asarray()
        blob_float_array2 = float_blob.asarray(True)
        self.assertEqual(blob_float_array.shape, blob_float_array2.shape)

        float_array[0][1][1][1] = 2.0

        self.assertEqual(float_array[0][1][1][1], blob_float_array[0][1][1][1])
        self.assertEqual(1.0, blob_float_array2[0][1][1][1])

    def test_vector(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        float_blob = neoml.Blob.vector(math_engine, 16, "float32")
        self.assertEqual(float_blob.batch_len, 16)
        self.assertEqual(float_blob.batch_width, 1)
        self.assertEqual(float_blob.list_size, 1)
        self.assertEqual(float_blob.height, 1)
        self.assertEqual(float_blob.width, 1)
        self.assertEqual(float_blob.depth, 1)
        self.assertEqual(float_blob.channels, 1)
        self.assertEqual(float_blob.size, 16)
        self.assertEqual(float_blob.object_count, 16)
        self.assertEqual(float_blob.object_size, 1)

    def test_matrix(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        float_blob = neoml.Blob.matrix(math_engine, 16, 32, "int32")
        self.assertEqual(float_blob.batch_len, 16)
        self.assertEqual(float_blob.batch_width, 32)
        self.assertEqual(float_blob.list_size, 1)
        self.assertEqual(float_blob.height, 1)
        self.assertEqual(float_blob.width, 1)
        self.assertEqual(float_blob.depth, 1)
        self.assertEqual(float_blob.channels, 1)
        self.assertEqual(float_blob.size, 16 * 32)
        self.assertEqual(float_blob.object_count, 16 * 32)
        self.assertEqual(float_blob.object_size, 1)

    def test_tensor(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        shape = (1, 2, 3, 4, 5, 6, 7)
        float_blob = neoml.Blob.tensor(math_engine, shape, "int32")
        self.assertEqual(float_blob.batch_len, 1)
        self.assertEqual(float_blob.batch_width, 2)
        self.assertEqual(float_blob.list_size, 3)
        self.assertEqual(float_blob.height, 4)
        self.assertEqual(float_blob.width, 5)
        self.assertEqual(float_blob.depth, 6)
        self.assertEqual(float_blob.channels, 7)
        self.assertEqual(float_blob.size, 1 * 2 * 3 * 4 * 5 * 6 * 7)
        self.assertEqual(float_blob.object_count, 2 * 3)
        self.assertEqual(float_blob.object_size, 4 * 5 * 6 * 7)

    def test_list_blob(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        float_blob = neoml.Blob.list_blob(math_engine, 2, 3, 4, 5, "int32")
        self.assertEqual(float_blob.batch_len, 2)
        self.assertEqual(float_blob.batch_width, 3)
        self.assertEqual(float_blob.list_size, 4)
        self.assertEqual(float_blob.height, 1)
        self.assertEqual(float_blob.width, 1)
        self.assertEqual(float_blob.depth, 1)
        self.assertEqual(float_blob.channels, 5)
        self.assertEqual(float_blob.size, 2 * 3 * 4 * 5)
        self.assertEqual(float_blob.object_count, 2 * 3 * 4)
        self.assertEqual(float_blob.object_size, 5)

    def test_image2d(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        float_blob = neoml.Blob.image2d(math_engine, 2, 3, 4, 5, 6, "float32")
        self.assertEqual(float_blob.batch_len, 2)
        self.assertEqual(float_blob.batch_width, 3)
        self.assertEqual(float_blob.list_size, 1)
        self.assertEqual(float_blob.height, 4)
        self.assertEqual(float_blob.width, 5)
        self.assertEqual(float_blob.depth, 1)
        self.assertEqual(float_blob.channels, 6)
        self.assertEqual(float_blob.size, 2 * 3 * 4 * 5 * 6)
        self.assertEqual(float_blob.object_count, 2 * 3)
        self.assertEqual(float_blob.object_size, 4 * 5 * 6)

    def test_image3d(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        float_blob = neoml.Blob.image3d(math_engine, 2, 3, 4, 5, 6, 7, "float32")
        self.assertEqual(float_blob.batch_len, 2)
        self.assertEqual(float_blob.batch_width, 3)
        self.assertEqual(float_blob.list_size, 1)
        self.assertEqual(float_blob.height, 4)
        self.assertEqual(float_blob.width, 5)
        self.assertEqual(float_blob.depth, 6)
        self.assertEqual(float_blob.channels, 7)
        self.assertEqual(float_blob.size, 2 * 3 * 4 * 5 * 6 * 7)
        self.assertEqual(float_blob.object_count, 2 * 3)
        self.assertEqual(float_blob.object_size, 4 * 5 * 6 * 7)


class SolverTestCase(TestCase):
    def test_nesterov_gradient(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        solver = neoml.Dnn.NesterovGradient(math_engine, learning_rate=0.6, l1=0.6, l2=0.6,
                                               moment_decay_rate=0.6, max_gradient_norm=0.6,
                                               second_moment_decay_rate=0.6, epsilon=0.6, ams_grad=True)

        self.assertAlmostEqual(solver.l1, 0.6, delta=1e-3)
        self.assertAlmostEqual(solver.l2, 0.6, delta=1e-3)
        self.assertAlmostEqual(solver.learning_rate, 0.6, delta=1e-3)
        self.assertAlmostEqual(solver.max_gradient_norm, 0.6, delta=1e-3)
        self.assertAlmostEqual(solver.moment_decay_rate, 0.6, delta=1e-3)
        self.assertAlmostEqual(solver.second_moment_decay_rate, 0.6, delta=1e-3)
        self.assertAlmostEqual(solver.epsilon, 0.6, delta=1e-3)
        self.assertEqual(solver.ams_grad, True)

    def test_adaptive_gradient(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        solver = neoml.Dnn.AdaptiveGradient(math_engine, learning_rate=0.6, l1=0.6, l2=0.6,
                                               moment_decay_rate=0.6, max_gradient_norm=0.6,
                                               second_moment_decay_rate=0.6, epsilon=0.6, ams_grad=True)

        self.assertAlmostEqual(solver.l1, 0.6, delta=1e-3)
        self.assertAlmostEqual(solver.l2, 0.6, delta=1e-3)
        self.assertAlmostEqual(solver.learning_rate, 0.6, delta=1e-3)
        self.assertAlmostEqual(solver.max_gradient_norm, 0.6, delta=1e-3)
        self.assertAlmostEqual(solver.moment_decay_rate, 0.6, delta=1e-3)
        self.assertAlmostEqual(solver.second_moment_decay_rate, 0.6, delta=1e-3)
        self.assertAlmostEqual(solver.epsilon, 0.6, delta=1e-3)
        self.assertEqual(solver.ams_grad, True)

    def test_simple_gradient(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        solver = neoml.Dnn.SimpleGradient(math_engine, learning_rate=0.6, l1=0.6, l2=0.6,
                                             moment_decay_rate=0.6, max_gradient_norm=0.6)

        self.assertAlmostEqual(solver.l1, 0.6, delta=1e-3)
        self.assertAlmostEqual(solver.l2, 0.6, delta=1e-3)
        self.assertAlmostEqual(solver.learning_rate, 0.6, delta=1e-3)
        self.assertAlmostEqual(solver.max_gradient_norm, 0.6, delta=1e-3)
        self.assertAlmostEqual(solver.moment_decay_rate, 0.6, delta=1e-3)


class LayersTestCase(TestCase):
    def test_lstm(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        dnn = neoml.Dnn.Dnn(math_engine)
        source1 = neoml.Dnn.Source(dnn, "source1")
        lstm = neoml.Dnn.Lstm(source1, 7, 0.6, name="lstm")
        sink1 = neoml.Dnn.Sink((lstm, 0), "sink1")
        sink2 = neoml.Dnn.Sink((lstm, 1), "sink2")
        layer = dnn.layers['lstm']
        self.assertEqual(layer.name, 'lstm')

        input1 = neoml.Blob.asblob(math_engine, np.ones((5, 3, 16), dtype=np.float32), (5, 3, 1, 1, 1, 1, 16))
        inputs = {"source1": input1}

        outputs = dnn.run(inputs)
        out1 = outputs["sink1"].asarray()
        out2 = outputs["sink2"].asarray()

        self.assertEqual(lstm.hidden_size, 7)
        self.assertEqual(layer.hidden_size, 7)

        self.assertEqual(lstm.reverse_sequence, False)
        lstm.reverse_sequence = True
        self.assertEqual(lstm.reverse_sequence, True)
        self.assertEqual(layer.reverse_sequence, True)

        self.assertAlmostEqual(lstm.dropout, 0.6, delta=1e-3)
        lstm.dropout = 0.9
        self.assertAlmostEqual(lstm.dropout, 0.9, delta=1e-3)
        self.assertAlmostEqual(layer.dropout, 0.9, delta=1e-3)

        self.assertEqual(lstm.activation, "sigmoid")
        lstm.activation = "abs"
        self.assertEqual(lstm.activation, "abs")

        self.assertEqual(out1.shape, (5, 3, 7))
        self.assertEqual(out2.shape, (5, 3, 7))

        w_blob = lstm.input_weights
        weights = w_blob.asarray()
        lstm.input_weights = w_blob
        f_blob = lstm.input_free_term
        free_term = f_blob.asarray()
        lstm.input_free_term = f_blob

        w_blob = lstm.recurrent_weights
        weights = w_blob.asarray()
        lstm.recurrent_weights = w_blob
        f_blob = lstm.recurrent_free_term
        free_term = f_blob.asarray()
        lstm.recurrent_free_term = f_blob

        self.assertEqual(weights.shape, (28, 7))
        self.assertEqual(free_term.shape, (28,))

    def test_fully_connected(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        dnn = neoml.Dnn.Dnn(math_engine)
        source1 = neoml.Dnn.Source(dnn, "source1")
        source2 = neoml.Dnn.Source(dnn, "source2")
        fully = neoml.Dnn.FullyConnected((source1, source2), 5, False, "fully")
        sink1 = neoml.Dnn.Sink((fully, 0), "sink1")
        sink2 = neoml.Dnn.Sink((fully, 1), "sink2")
        layer = dnn.layers['fully']
        self.assertEqual(layer.name, 'fully')

        input1 = neoml.Blob.asblob(math_engine, np.ones((12, 16), dtype=np.float32), (12, 1, 1, 1, 1, 1, 16))
        input2 = neoml.Blob.asblob(math_engine, np.ones((10, 16), dtype=np.float32), (10, 1, 1, 1, 1, 1, 16))

        inputs = {"source1": input1, "source2": input2}
        outputs = dnn.run(inputs)
        out1 = outputs["sink1"].asarray()
        out2 = outputs["sink2"].asarray()

        self.assertEqual(fully.element_count, 5)
        self.assertEqual(layer.element_count, 5)

        self.assertEqual(fully.zero_free_term, False)
        fully.zero_free_term = True
        self.assertEqual(fully.zero_free_term, True)
        self.assertEqual(layer.zero_free_term, True)

        self.assertEqual(out1.shape, (12, 5))
        self.assertEqual(out2.shape, (10, 5))

        w_blob = fully.weights
        weights = w_blob.asarray()
        fully.weights = w_blob
        f_blob = fully.free_term
        free_term = f_blob.asarray()
        fully.free_term = f_blob

        self.assertEqual(weights.shape, (5, 16))
        self.assertEqual(free_term.shape, (5,))

    def test_concat_channels(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        dnn = neoml.Dnn.Dnn(math_engine)
        source1 = neoml.Dnn.Source(dnn, "source1")
        source2 = neoml.Dnn.Source(dnn, "source2")
        concat = neoml.Dnn.ConcatChannels((source1, source2), "concat")
        sink = neoml.Dnn.Sink(concat, "sink")
        layer = dnn.layers['concat']
        self.assertEqual(layer.name, 'concat')

        input1 = neoml.Blob.asblob(math_engine, np.ones((16), dtype=np.float32), (1, 1, 1, 1, 1, 1, 16))
        input2 = neoml.Blob.asblob(math_engine, np.ones((16), dtype=np.float32), (1, 1, 1, 1, 1, 1, 16))

        inputs = {"source1": input1, "source2": input2}
        outputs = dnn.run(inputs)
        a = outputs["sink"].asarray()

        self.assertEqual(outputs["sink"].channels, 32)
        self.assertEqual(a.size, 32)

    def test_concat_depth(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        dnn = neoml.Dnn.Dnn(math_engine)
        source1 = neoml.Dnn.Source(dnn, "source1")
        source2 = neoml.Dnn.Source(dnn, "source2")
        concat = neoml.Dnn.ConcatDepth((source1, source2), "concat")
        sink = neoml.Dnn.Sink(concat, "sink")
        layer = dnn.layers['concat']
        self.assertEqual(layer.name, 'concat')

        input1 = neoml.Blob.asblob(math_engine, np.ones((16), dtype=np.float32), (1, 1, 1, 1, 1, 16, 1))
        input2 = neoml.Blob.asblob(math_engine, np.ones((16), dtype=np.float32), (1, 1, 1, 1, 1, 16, 1))

        inputs = {"source1": input1, "source2": input2}
        outputs = dnn.run(inputs)
        a = outputs["sink"].asarray()

        self.assertEqual(outputs["sink"].depth, 32)
        self.assertEqual(a.size, 32)

    def test_concat_width(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        dnn = neoml.Dnn.Dnn(math_engine)
        source1 = neoml.Dnn.Source(dnn, "source1")
        source2 = neoml.Dnn.Source(dnn, "source2")
        concat = neoml.Dnn.ConcatWidth((source1, source2), "concat")
        sink = neoml.Dnn.Sink(concat, "sink")
        layer = dnn.layers['concat']
        self.assertEqual(layer.name, 'concat')

        input1 = neoml.Blob.asblob(math_engine, np.ones((16), dtype=np.float32), (1, 1, 1, 1, 16, 1, 1))
        input2 = neoml.Blob.asblob(math_engine, np.ones((16), dtype=np.float32), (1, 1, 1, 1, 16, 1, 1))

        inputs = {"source1": input1, "source2": input2}
        outputs = dnn.run(inputs)
        a = outputs["sink"].asarray()

        self.assertEqual(outputs["sink"].width, 32)
        self.assertEqual(a.size, 32)

    def test_concat_height(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        dnn = neoml.Dnn.Dnn(math_engine)
        source1 = neoml.Dnn.Source(dnn, "source1")
        source2 = neoml.Dnn.Source(dnn, "source2")
        concat = neoml.Dnn.ConcatHeight((source1, source2), "concat")
        sink = neoml.Dnn.Sink(concat, "sink")
        layer = dnn.layers['concat']
        self.assertEqual(layer.name, 'concat')

        input1 = neoml.Blob.asblob(math_engine, np.ones((16), dtype=np.float32), (1, 1, 1, 16, 1, 1, 1))
        input2 = neoml.Blob.asblob(math_engine, np.ones((16), dtype=np.float32), (1, 1, 1, 16, 1, 1, 1))

        inputs = {"source1": input1, "source2": input2}
        outputs = dnn.run(inputs)
        a = outputs["sink"].asarray()

        self.assertEqual(outputs["sink"].height, 32)
        self.assertEqual(a.size, 32)

    def test_concat_batch_width(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        dnn = neoml.Dnn.Dnn(math_engine)
        source1 = neoml.Dnn.Source(dnn, "source1")
        source2 = neoml.Dnn.Source(dnn, "source2")
        concat = neoml.Dnn.ConcatBatchWidth((source1, source2), "concat")
        sink = neoml.Dnn.Sink(concat, "sink")
        layer = dnn.layers['concat']
        self.assertEqual(layer.name, 'concat')

        input1 = neoml.Blob.asblob(math_engine, np.ones((16), dtype=np.float32), (1, 16, 1, 1, 1, 1, 1))
        input2 = neoml.Blob.asblob(math_engine, np.ones((16), dtype=np.float32), (1, 16, 1, 1, 1, 1, 1))

        inputs = {"source1": input1, "source2": input2}
        outputs = dnn.run(inputs)
        a = outputs["sink"].asarray()

        self.assertEqual(outputs["sink"].batch_width, 32)
        self.assertEqual(a.size, 32)

    def test_concat_object(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        dnn = neoml.Dnn.Dnn(math_engine)
        source1 = neoml.Dnn.Source(dnn, "source1")
        source2 = neoml.Dnn.Source(dnn, "source2")
        concat = neoml.Dnn.ConcatObject((source1, source2), "concat")
        sink = neoml.Dnn.Sink(concat, "sink")
        layer = dnn.layers['concat']
        self.assertEqual(layer.name, 'concat')

        input1 = neoml.Blob.asblob(math_engine, np.ones((2, 3, 4, 5), dtype=np.float32), (1, 1, 1, 2, 3, 4, 5))
        input2 = neoml.Blob.asblob(math_engine, np.ones((16), dtype=np.float32), (1, 1, 1, 16, 1, 1, 1))

        inputs = {"source1": input1, "source2": input2}
        outputs = dnn.run(inputs)
        a = outputs["sink"].asarray()

        self.assertEqual(outputs["sink"].channels, 136)
        self.assertEqual(a.size, 136)

    def test_enum_binarization(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        dnn = neoml.Dnn.Dnn(math_engine)
        source1 = neoml.Dnn.Source(dnn, "source1")
        bin = neoml.Dnn.EnumBinarization(source1, 5, "bin")
        sink = neoml.Dnn.Sink(bin, "sink")
        layer = dnn.layers['bin']
        self.assertEqual(layer.name, 'bin')

        self.assertEqual(bin.enum_size, 5)
        bin.enum_size = 4
        self.assertEqual(bin.enum_size, 4)
        self.assertEqual(layer.enum_size, 4)

        input1 = neoml.Blob.asblob(math_engine, np.ones((4, 3, 3, 3), dtype=np.float32), (4, 1, 1, 3, 3, 3, 1))

        inputs = {"source1": input1}
        outputs = dnn.run(inputs)
        a = outputs["sink"]

        self.assertEqual(a.shape, (4, 1, 1, 3, 3, 3, 4))

    def test_bitset_vectorization(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        dnn = neoml.Dnn.Dnn(math_engine)
        source1 = neoml.Dnn.Source(dnn, "source1")
        bin = neoml.Dnn.BitSetVectorization(source1, 5, "bin")
        sink = neoml.Dnn.Sink(bin, "sink")
        layer = dnn.layers['bin']
        self.assertEqual(layer.name, 'bin')

        self.assertEqual(bin.bit_set_size, 5)
        bin.bit_set_size = 4
        self.assertEqual(bin.bit_set_size, 4)
        self.assertEqual(layer.bit_set_size, 4)

        input1 = neoml.Blob.asblob(math_engine, np.ones((4, 3, 3, 3), dtype=np.int32), (4, 1, 1, 3, 3, 3, 1))

        inputs = {"source1": input1}
        outputs = dnn.run(inputs)
        a = outputs["sink"]

        self.assertEqual(a.shape, (4, 1, 1, 3, 3, 3, 4))

    def test_dotproduct(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        dnn = neoml.Dnn.Dnn(math_engine)
        source1 = neoml.Dnn.Source(dnn, "source1")
        source2 = neoml.Dnn.Source(dnn, "source2")
        dotProduct = neoml.Dnn.DotProduct((source1, source2), "dotProduct")
        sink = neoml.Dnn.Sink(dotProduct, "sink")
        layer = dnn.layers['dotProduct']
        self.assertEqual(layer.name, 'dotProduct')

        input1 = neoml.Blob.asblob(math_engine, np.ones((16), dtype=np.float32), (1, 1, 1, 1, 1, 1, 16))
        input2 = neoml.Blob.asblob(math_engine, np.ones((16), dtype=np.float32), (1, 1, 1, 1, 1, 1, 16))

        inputs = {"source1": input1, "source2": input2}
        outputs = dnn.run(inputs)
        a = outputs["sink"].asarray()

        self.assertEqual(a.size, 1)
        self.assertEqual(a[0], 16)

    def test_dropout(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        dnn = neoml.Dnn.Dnn(math_engine)
        source = neoml.Dnn.Source(dnn, "source")
        dropout = neoml.Dnn.Dropout(source, 0.5, True, True, "dropout")
        sink = neoml.Dnn.Sink(dropout, "sink")
        layer = dnn.layers['dropout']
        self.assertEqual(layer.name, 'dropout')

        input = neoml.Blob.asblob(math_engine, np.ones((2, 3, 5, 4), dtype=np.float32), (2, 3, 1, 5, 1, 1, 4))
        inputs = {"source": input}
        outputs = dnn.run(inputs)
        a = outputs["sink"].asarray()

        self.assertEqual(a.shape, input.asarray().shape)
        self.assertEqual(dropout.rate, 0.5)
        self.assertEqual(dropout.spatial, True)
        self.assertEqual(dropout.batchwise, True)
        self.assertEqual(layer.rate, 0.5)

    def test_accumulative_lookup(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        dnn = neoml.Dnn.Dnn(math_engine)
        source = neoml.Dnn.Source(dnn, "source")
        lookup = neoml.Dnn.AccumulativeLookup(source, 5, 6, "lookup")
        sink = neoml.Dnn.Sink(lookup, "sink")
        layer = dnn.layers['lookup']
        self.assertEqual(layer.name, 'lookup')

        self.assertEqual(lookup.size, 6)
        self.assertEqual(lookup.count, 5)

        input = neoml.Blob.asblob(math_engine, np.ones((2, 5, 3), dtype=np.int32), (2, 1, 1, 5, 1, 1, 3))
        inputs = {"source": input}
        outputs = dnn.run(inputs)
        out = outputs["sink"].asarray()
        self.assertEqual(out.shape, (2, 6))

    def test_multichannel_lookup(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        dnn = neoml.Dnn.Dnn(math_engine)
        source = neoml.Dnn.Source(dnn, "source")
        lookup = neoml.Dnn.MultichannelLookup((source,), [(1, 4)], "lookup")
        sink = neoml.Dnn.Sink(lookup, "sink")
        layer = dnn.layers['lookup']
        self.assertEqual(layer.name, 'lookup')

        self.assertEqual(lookup.dimensions, [(1, 4)])
        lookup.dimensions = [(3, 5)]
        self.assertEqual(layer.dimensions, [(3, 5)])

        input = neoml.Blob.asblob(math_engine, np.ones((2, 5, 3), dtype=np.float32), (2, 1, 1, 5, 1, 1, 3))
        inputs = {"source": input}
        outputs = dnn.run(inputs)
        out = outputs["sink"].asarray()
        self.assertEqual(out.shape, (2, 5, 7))

        blob = lookup.get_embeddings(0)
        lookup.set_embeddings(0, blob)

    def test_tied_embeddings(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        dnn = neoml.Dnn.Dnn(math_engine)
        source = neoml.Dnn.Source(dnn, "source")
        tied = neoml.Dnn.TiedEmbeddings((source,), "embeddings", 0, "tied")
        sink = neoml.Dnn.Sink(tied, "sink")
        layer = dnn.layers['tied']
        self.assertEqual(layer.name, 'tied')

        self.assertEqual(tied.channel, 0)
        tied.channel = 1
        self.assertEqual(tied.channel, 1)
        self.assertEqual(layer.channel, 1)

        self.assertEqual(tied.embeddings_layer_name, "embeddings")
        tied.embeddings_layer_name = "embeddings2"
        self.assertEqual(tied.embeddings_layer_name, "embeddings2")

    def test_accuracy(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        dnn = neoml.Dnn.Dnn(math_engine)
        source1 = neoml.Dnn.Source(dnn, "source1")
        source2 = neoml.Dnn.Source(dnn, "source2")
        accuracy = neoml.Dnn.Accuracy((source1, source2), True, "accuracy")
        sink = neoml.Dnn.Sink(accuracy, "sink")
        layer = dnn.layers['accuracy']
        self.assertEqual(layer.name, 'accuracy')

        self.assertEqual(accuracy.reset, True)
        self.assertEqual(layer.reset, True)

        input1 = neoml.Blob.asblob(math_engine, np.ones((1, 16), dtype=np.float32), (1, 16, 1, 1, 1, 1, 1))
        input2 = neoml.Blob.asblob(math_engine, np.ones((1, 16), dtype=np.float32), (1, 16, 1, 1, 1, 1, 1))

        inputs = {"source1": input1, "source2": input2}
        outputs = dnn.run(inputs)
        a = outputs["sink"].asarray()

        self.assertEqual(a.size, 1)
        self.assertAlmostEqual(a[0], 1.0, delta=1e-3)

    def test_confusion_matrix(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        dnn = neoml.Dnn.Dnn(math_engine)
        source1 = neoml.Dnn.Source(dnn, "source1")
        source2 = neoml.Dnn.Source(dnn, "source2")
        accuracy = neoml.Dnn.ConfusionMatrix((source1, source2), True, "accuracy")
        sink = neoml.Dnn.Sink(accuracy, "sink")
        layer = dnn.layers['accuracy']
        self.assertEqual(layer.name, 'accuracy')

        self.assertEqual(accuracy.reset, True)
        self.assertEqual(layer.reset, True)

        input1 = neoml.Blob.asblob(math_engine, np.ones((16, 2), dtype=np.float32), (1, 16, 1, 1, 1, 1, 2))
        input2 = neoml.Blob.asblob(math_engine, np.ones((16, 2), dtype=np.float32), (1, 16, 1, 1, 1, 1, 2))

        inputs = {"source1": input1, "source2": input2}
        outputs = dnn.run(inputs)
        a = outputs["sink"].asarray()

        self.assertEqual( accuracy.matrix.shape, (2, 2) )

        self.assertEqual(a.size, 4)
        self.assertAlmostEqual(a[0][0], 16.0, delta=1e-3)

    def _test_activation(self, layer, kwargs={}):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        dnn = neoml.Dnn.Dnn(math_engine)
        source = neoml.Dnn.Source(dnn, "source")
        activation = getattr(neoml.Dnn, layer)(source, name="activation", **kwargs)
        sink = neoml.Dnn.Sink(activation, "sink")
        layer = dnn.layers['activation']
        self.assertEqual(layer.name, 'activation')

        input = neoml.Blob.asblob(math_engine, np.ones((1, 16), dtype=np.float32), (1, 16, 1, 1, 1, 1, 1))
        inputs = {"source": input}
        outputs = dnn.run(inputs)
        out = outputs["sink"].asarray()

        for k,v in kwargs.items():
            self.assertAlmostEqual(getattr(activation, k), v, delta=1e-3,
                                   msg='Field {} of {} activation differs'.format(k, layer))
            self.assertEqual(getattr(activation, k), getattr(layer, k))

        return out

    def test_activation_linear(self):
        out = self._test_activation('Linear', dict(multiplier=3.3, free_term=4.4))
        self.assertTrue(np.isclose(out, 7.7).all())

    def test_activation_elu(self):
        out = self._test_activation('ELU', dict(alpha=3.3))
        self.assertTrue(np.isclose(out, 1).all())

    def test_activation_relu(self):
        out = self._test_activation('ReLU', dict(threshold=3.3))
        self.assertTrue(np.isclose(out, 1).all())

    def test_activation_leaky_relu(self):
        out = self._test_activation('LeakyReLU', dict(alpha=3.3))
        self.assertTrue(np.isclose(out, 1).all())

    def test_activation_hswish(self):
        out = self._test_activation('HSwish')
        self.assertTrue(np.isclose(out, 2./3).all())

    def test_activation_gelu(self):
        out = self._test_activation('GELU')
        self.assertTrue(np.isclose(out, 0.84579575).all())

    def test_activation_abs(self):
        out = self._test_activation('Abs')
        self.assertTrue(np.isclose(out, 1).all())

    def test_activation_sigmoid(self):
        out = self._test_activation('Sigmoid')
        self.assertTrue(np.isclose(out, 0.7310586).all())

    def test_activation_tanh(self):
        out = self._test_activation('Tanh')
        self.assertTrue(np.isclose(out, 0.7615942).all())

    def test_activation_hardtanh(self):
        out = self._test_activation('HardTanh')
        self.assertTrue(np.isclose(out, 1).all())

    def test_activation_hardsigmoid(self):
        out = self._test_activation('HardSigmoid', dict(slope=5.5, bias=6.6))
        self.assertTrue(np.isclose(out, 1).all())

    def test_activation_power(self):
        out = self._test_activation('Power', dict(exponent=5.5))
        self.assertTrue(np.isclose(out, 1).all())

    def test_add_object(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        dnn = neoml.Dnn.Dnn(math_engine)
        source1 = neoml.Dnn.Source(dnn, "source1")
        source2 = neoml.Dnn.Source(dnn, "source2")
        add_to_object = neoml.Dnn.AddToObject((source1, source2), "add_to_object")
        sink = neoml.Dnn.Sink(add_to_object, "sink")
        layer = dnn.layers['add_to_object']
        self.assertEqual(layer.name, 'add_to_object')

        input1 = neoml.Blob.asblob(math_engine, np.ones((8, 4, 4), dtype=np.float32), (1, 8, 1, 4, 4, 1, 1))
        input2 = neoml.Blob.asblob(math_engine, np.ones((4, 4), dtype=np.float32), (1, 1, 1, 4, 4, 1, 1))

        inputs = {"source1": input1, "source2": input2}
        outputs = dnn.run(inputs)
        a = outputs["sink"].asarray()

        self.assertEqual(a.size, 128)
        self.assertAlmostEqual(a[1][1][1], 2.0, delta=1e-3)

    def test_argmax(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        dnn = neoml.Dnn.Dnn(math_engine)
        source = neoml.Dnn.Source(dnn, "source")
        argmax = neoml.Dnn.Argmax(source, dimension="channels", name="argmax")
        sink = neoml.Dnn.Sink(argmax, "sink")
        layer = dnn.layers['argmax']
        self.assertEqual(layer.name, 'argmax')

        self.assertEqual(argmax.dimension, "channels")
        argmax.dimension = "batch_length"
        self.assertEqual(argmax.dimension, "batch_length")

        input = neoml.Blob.asblob(math_engine, np.array([1, 2, 3, 1], dtype=np.float32), (4, 1, 1, 1, 1, 1, 1))
        inputs = {"source": input}
        outputs = dnn.run(inputs)
        out = outputs["sink"].asarray()
        self.assertEqual(out, 2)

    def test_attention_decoder(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        dnn = neoml.Dnn.Dnn(math_engine)
        source1 = neoml.Dnn.Source(dnn, "source1")
        source2 = neoml.Dnn.Source(dnn, "source2")
        decoder = neoml.Dnn.AttentionDecoder((source1, source2), "additive", 16, 32, 64, "decoder")
        sink = neoml.Dnn.Sink(decoder, "sink")
        layer = dnn.layers['decoder']
        self.assertEqual(layer.name, 'decoder')

        self.assertEqual(decoder.hidden_layer_size, 16)
        self.assertEqual(decoder.output_object_size, 32)
        self.assertEqual(decoder.output_seq_len, 64)
        

        self.assertEqual(decoder.score, "additive")
        decoder.score = "dot_product"
        self.assertEqual(decoder.score, "dot_product")
        decoder.score = "additive"

        decoder.hidden_layer_size = 1
        self.assertEqual(decoder.hidden_layer_size, 1)
        self.assertEqual(layer.hidden_layer_size, 1)

        decoder.output_object_size = 1
        self.assertEqual(decoder.output_object_size, 1)
        self.assertEqual(layer.output_object_size, 1)

        decoder.output_seq_len = 1
        self.assertEqual(decoder.output_seq_len, 1)
        self.assertEqual(layer.output_seq_len, 1)

        input1 = neoml.Blob.asblob(math_engine, np.ones(1, dtype=np.float32), (1, 1, 1, 1, 1, 1, 1))
        input2 = neoml.Blob.asblob(math_engine, np.ones(1, dtype=np.float32), (1, 1, 1, 1, 1, 1, 1))
        inputs = {"source1": input1, "source2": input2}
        outputs = dnn.run(inputs)
        out = outputs["sink"].asarray()
        self.assertSequenceEqual(out, [1])

    def test_batch_norm(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        dnn = neoml.Dnn.Dnn(math_engine)
        source = neoml.Dnn.Source(dnn, "source")
        batch_norm = neoml.Dnn.BatchNormalization(source, True, True, 0.3, "batch_norm")
        sink = neoml.Dnn.Sink(batch_norm, "sink")
        layer = dnn.layers['batch_norm']
        self.assertEqual(layer.name, 'batch_norm')

        arr = np.ones((5, 3, 2), dtype=np.float32)
        input = neoml.Blob.asblob(math_engine, arr, (5, 1, 3, 2, 1, 1, 1))
        inputs = {"source": input}
        outputs = dnn.run(inputs)
        out = outputs["sink"].asarray()

        self.assertTrue(np.array_equal(arr, out))
        self.assertEqual(batch_norm.channel_based, True)
        self.assertEqual(batch_norm.zero_free_term, True)
        self.assertAlmostEqual(batch_norm.slow_convergence_rate, 0.3, delta=1e-3)
        self.assertEqual(layer.channel_based, True)
        self.assertEqual(layer.zero_free_term, True)
        self.assertAlmostEqual(layer.slow_convergence_rate, 0.3, delta=1e-3)

    def test_matrix_multiplication(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        dnn = neoml.Dnn.Dnn(math_engine)
        source1 = neoml.Dnn.Source(dnn, "source1")
        source2 = neoml.Dnn.Source(dnn, "source2")
        mult = neoml.Dnn.MatrixMultiplication((source1, source2), "mm")
        sink = neoml.Dnn.Sink(mult, "sink")
        layer = dnn.layers['mm']
        self.assertEqual(layer.name, 'mm')

        mult1 = np.array([[1, 2], [3, 4]], dtype=np.float32)
        mult2 = np.array([[1, 2], [3, 4]], dtype=np.float32)
        input1 = neoml.Blob.asblob(math_engine, mult1, (2, 1, 2, 1, 1, 1, 1))
        input2 = neoml.Blob.asblob(math_engine, mult2, (2, 1, 2, 1, 1, 1, 1))
        inputs = {"source1": input1, "source2": input2}
        outputs = dnn.run(inputs)
        out = outputs["sink"].asarray()
        
        self.assertTrue(np.array_equal(out, mult1 * mult2))

    def test_multihead_attention(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        dnn = neoml.Dnn.Dnn(math_engine)
        source1 = neoml.Dnn.Source(dnn, "source1")
        source2 = neoml.Dnn.Source(dnn, "source2")
        source3 = neoml.Dnn.Source(dnn, "source3")
        att = neoml.Dnn.MultiheadAttention((source1, source2, source3), 5, 9, 3, 0.3, "att")
        sink = neoml.Dnn.Sink(att, "sink")
        layer = dnn.layers['att']
        self.assertEqual(layer.name, 'att')

        self.assertEqual(att.head_count, 5)
        att.head_count = 4
        self.assertEqual(att.head_count, 4)

        self.assertEqual(att.hidden_size, 9)
        att.hidden_size = 8
        self.assertEqual(att.hidden_size, 8)

        self.assertEqual(att.output_size, 3)
        att.output_size = 8
        self.assertEqual(att.output_size, 8)

        self.assertEqual(att.use_mask, False)
        att.use_mask = True
        self.assertEqual(att.use_mask, True)
        att.use_mask = False

        self.assertAlmostEqual(att.dropout_rate, 0.3, delta=1e-3)
        att.dropout_rate = 0.4
        self.assertAlmostEqual(att.dropout_rate, 0.4, delta=1e-3)

        self.assertEqual(layer.hidden_size, 8)
        self.assertEqual(layer.output_size, 8)
        self.assertEqual(layer.use_mask, False)
        self.assertAlmostEqual(layer.dropout_rate, 0.4, delta=1e-3)

        input1 = neoml.Blob.asblob(math_engine, np.ones((4, 3, 3), dtype=np.float32), (1, 4, 3, 1, 1, 1, 3))
        input2 = neoml.Blob.asblob(math_engine, np.ones((4, 2, 3), dtype=np.float32), (1, 4, 2, 1, 1, 1, 3))
        input3 = neoml.Blob.asblob(math_engine, np.ones((4, 2, 2), dtype=np.float32), (1, 4, 2, 1, 1, 1, 4))
        inputs = {"source1": input1, "source2": input2, "source3": input3}
        outputs = dnn.run(inputs)
        out = outputs["sink"].asarray()
        self.assertEqual(out.shape, (4, 3, 8))

    def test_image_to_pixel(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        dnn = neoml.Dnn.Dnn(math_engine)
        source1 = neoml.Dnn.Source(dnn, "source1")
        source2 = neoml.Dnn.Source(dnn, "source2")
        conv = neoml.Dnn.ImageToPixel((source1, source2), "conv")
        sink = neoml.Dnn.Sink(conv, "sink")
        layer = dnn.layers['conv']
        self.assertEqual(layer.name, 'conv')

        input1 = neoml.Blob.asblob(math_engine, np.ones((4, 3, 2), dtype=np.float32), (1, 4, 3, 1, 1, 1, 2))
        input2 = neoml.Blob.asblob(math_engine, np.zeros((4, 3), dtype=np.int32), (1, 4, 1, 1, 1, 1, 3))
        inputs = {"source1": input1, "source2": input2}
        outputs = dnn.run(inputs)
        out = outputs["sink"].asarray()
        self.assertEqual(out.shape, (4, 3, 2))

    def test_pixel_to_image(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        dnn = neoml.Dnn.Dnn(math_engine)
        source1 = neoml.Dnn.Source(dnn, "source1")
        source2 = neoml.Dnn.Source(dnn, "source2")
        conv = neoml.Dnn.PixelToImage((source1, source2), 4, 8, "conv")
        sink = neoml.Dnn.Sink(conv, "sink")
        layer = dnn.layers['conv']
        self.assertEqual(layer.name, 'conv')

        self.assertEqual(conv.height, 4)
        conv.height = 5
        self.assertEqual(conv.height, 5)

        self.assertEqual(conv.width, 8)
        conv.width = 9
        self.assertEqual(conv.width, 9)

        self.assertEqual(layer.height, 5)
        self.assertEqual(layer.width, 9)

        input1 = neoml.Blob.asblob(math_engine, np.ones((4, 3, 2), dtype=np.float32), (1, 4, 3, 1, 1, 1, 2))
        input2 = neoml.Blob.asblob(math_engine, np.zeros((4, 3), dtype=np.int32), (1, 4, 1, 1, 1, 1, 3))
        inputs = {"source1": input1, "source2": input2}
        outputs = dnn.run(inputs)
        out = outputs["sink"].asarray()
        self.assertEqual(out.shape, (4, 5, 9, 2))

    def test_image_resize(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        dnn = neoml.Dnn.Dnn(math_engine)
        source1 = neoml.Dnn.Source(dnn, "source1")
        conv = neoml.Dnn.ImageResize(source1, [5, 6, 7, 8], 0.1, "conv")
        sink = neoml.Dnn.Sink(conv, "sink")
        layer = dnn.layers['conv']
        self.assertEqual(layer.name, 'conv')

        self.assertEqual(conv.deltas, [5, 6, 7, 8] )
        conv.deltas = [1, 2, 3, 4]
        self.assertEqual(conv.deltas, [1, 2, 3, 4])

        self.assertAlmostEqual(conv.default_value, 0.1)
        conv.default_value = 0.2
        self.assertAlmostEqual(conv.default_value, 0.2)

        self.assertEqual(layer.deltas, [1, 2, 3, 4])
        self.assertAlmostEqual(layer.default_value, 0.2)

        input1 = neoml.Blob.asblob(math_engine, np.ones((2, 10, 11, 2), dtype=np.float32), (1, 1, 2, 10, 11, 1, 2))
        inputs = {"source1": input1}
        outputs = dnn.run(inputs)
        out = outputs["sink"].asarray()
        self.assertEqual(out.shape, (2, 17, 14, 2))

    def test_crf(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        dnn = neoml.Dnn.Dnn(math_engine)
        source1 = neoml.Dnn.Source(dnn, "source1")
        source2 = neoml.Dnn.Source(dnn, "source2")
        crf = neoml.Dnn.Crf((source1, source2), 5, 3, 0.3, "crf")
        sink1 = neoml.Dnn.Sink((crf, 0), "sink1")
        sink2 = neoml.Dnn.Sink((crf, 1), "sink2")
        sink3 = neoml.Dnn.Sink((crf, 2), "sink3")
        layer = dnn.layers['crf']
        self.assertEqual(layer.name, 'crf')

        self.assertEqual(crf.class_count, 5)
        crf.class_count = 7
        self.assertEqual(crf.class_count, 7)

        self.assertEqual(crf.padding, 3)
        crf.padding = 1
        self.assertEqual(crf.padding, 1)

        self.assertAlmostEqual(crf.dropout_rate, 0.3)
        crf.dropout_rate = 0.2
        self.assertAlmostEqual(crf.dropout_rate, 0.2)

        self.assertEqual(crf.calc_best_prev_class, False)
        crf.calc_best_prev_class = True
        self.assertEqual(crf.calc_best_prev_class, True)

        self.assertEqual(layer.class_count, 7)
        self.assertEqual(layer.padding, 1)
        self.assertAlmostEqual(layer.dropout_rate, 0.2)
        self.assertEqual(layer.calc_best_prev_class, True)

        hidden_weights = crf.hidden_weights
        crf.hidden_weights = hidden_weights
        
        free_terms = crf.free_terms
        crf.free_terms = free_terms

        transitions = crf.transitions
        crf.transitions = transitions

        input1 = neoml.Blob.asblob(math_engine, np.ones((5, 7), dtype=np.float32), (1, 1, 5, 1, 1, 1, 7))
        input2 = neoml.Blob.asblob(math_engine, np.ones((5, ), dtype=np.int32), (1, 1, 5, 1, 1, 1, 1))
        inputs = {"source1": input1, "source2": input2}
        outputs = dnn.run(inputs)
        out1 = outputs["sink1"].asarray()
        out2 = outputs["sink2"].asarray()
        out3 = outputs["sink3"].asarray()
        self.assertEqual(out1.shape, (7,))
        self.assertEqual(out2.shape, (7,))
        self.assertEqual(out3.shape, (1,))

    def test_crf_loss(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        dnn = neoml.Dnn.Dnn(math_engine)
        source1 = neoml.Dnn.Source(dnn, "source1")
        source2 = neoml.Dnn.Source(dnn, "source2")
        source3 = neoml.Dnn.Source(dnn, "source3")
        crfLoss = neoml.Dnn.CrfLoss((source1, source2, source3), 0.4, "loss")
        layer = dnn.layers['loss']
        self.assertEqual(layer.name, 'loss')

        self.assertAlmostEqual(crfLoss.loss_weight, 0.4, delta=1e-3)
        crfLoss.loss_weight = 0.6
        self.assertAlmostEqual(crfLoss.loss_weight, 0.6, delta=1e-3)
        self.assertAlmostEqual(layer.loss_weight, 0.6, delta=1e-3)

        crfLoss.max_gradient = 0.6
        self.assertAlmostEqual(crfLoss.max_gradient, 0.6, delta=1e-3)
        self.assertAlmostEqual(layer.max_gradient, 0.6, delta=1e-3)

        self.assertAlmostEqual(crfLoss.last_loss, 0, delta=1e-3)

        input1 = neoml.Blob.asblob(math_engine, np.ones((3, 5), dtype=np.int32), (3, 1, 5, 1, 1, 1, 1))
        input2 = neoml.Blob.asblob(math_engine, np.ones((3, 5), dtype=np.float32), (3, 1, 5, 1, 1, 1, 1))
        input3 = neoml.Blob.asblob(math_engine, np.ones((3, 5), dtype=np.float32), (3, 1, 5, 1, 1, 1, 1))
        inputs = {"source1": input1, "source2": input2, "source3": input3}
        dnn.run(inputs)

        self.assertAlmostEqual(crfLoss.last_loss, -2, delta=1e-3)

    def test_crf_best_sequence(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        dnn = neoml.Dnn.Dnn(math_engine)
        source1 = neoml.Dnn.Source(dnn, "source1")
        source2 = neoml.Dnn.Source(dnn, "source2")
        best = neoml.Dnn.BestSequence((source1, source2), "best")
        sink = neoml.Dnn.Sink(best, "sink")
        layer = dnn.layers['best']
        self.assertEqual(layer.name, 'best')

        input1 = neoml.Blob.asblob(math_engine, np.zeros((3, 5), dtype=np.int32), (3, 1, 5, 1, 1, 1, 1))
        input2 = neoml.Blob.asblob(math_engine, np.ones((3, 5), dtype=np.float32), (3, 1, 5, 1, 1, 1, 1))
        inputs = {"source1": input1, "source2": input2}
        outputs = dnn.run(inputs)
        out = outputs["sink"].asarray()
        self.assertTrue(np.equal(out, [0., 0., 0.]).all())

    def test_ctc_loss(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        dnn = neoml.Dnn.Dnn(math_engine)
        source1 = neoml.Dnn.Source(dnn, "source1")
        source2 = neoml.Dnn.Source(dnn, "source2")
        ctcLoss = neoml.Dnn.CtcLoss((source1, source2), 6, False, 0.4, "loss")
        layer = dnn.layers['loss']
        self.assertEqual(layer.name, 'loss')

        self.assertEqual(ctcLoss.blank, 6)
        ctcLoss.blank = 5
        self.assertEqual(ctcLoss.blank, 5)
        self.assertEqual(layer.blank, 5)

        self.assertAlmostEqual(ctcLoss.loss_weight, 0.4, delta=1e-3)
        ctcLoss.loss_weight = 0.6
        self.assertAlmostEqual(ctcLoss.loss_weight, 0.6, delta=1e-3)
        self.assertAlmostEqual(layer.loss_weight, 0.6, delta=1e-3)

        ctcLoss.max_gradient = 0.6
        self.assertAlmostEqual(ctcLoss.max_gradient, 0.6, delta=1e-3)

        self.assertAlmostEqual(ctcLoss.last_loss, 0, delta=1e-3)

        self.assertEqual(ctcLoss.skip, False)
        ctcLoss.skip = True
        self.assertEqual(ctcLoss.skip, True)

        input1 = neoml.Blob.asblob(math_engine, np.ones((64, 4, 5), dtype=np.float32), (3, 4, 1, 1, 1, 1, 5))
        input2 = neoml.Blob.asblob(math_engine, np.ones((2, 4), dtype=np.int32), (2, 4, 1, 1, 1, 1, 1))
        inputs = {"source1": input1, "source2": input2}
        dnn.run(inputs)
        self.assertAlmostEqual(ctcLoss.last_loss, 4.8283, delta=1e-4)
        self.assertAlmostEqual(layer.last_loss, 4.8283, delta=1e-4)

    def test_ctc_decoding(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        dnn = neoml.Dnn.Dnn(math_engine)
        source1 = neoml.Dnn.Source(dnn, "source1")
        source2 = neoml.Dnn.Source(dnn, "source2")
        ctc = neoml.Dnn.CtcDecoding((source1, source2), 5, 0.4, 0.5, "ctc")
        layer = dnn.layers['ctc']
        self.assertEqual(layer.name, 'ctc')

        self.assertEqual(ctc.blank, 5)
        ctc.blank = 6
        self.assertEqual(ctc.blank, 6)

        self.assertAlmostEqual(ctc.blank_threshold, 0.4, delta=1e-3)
        ctc.blank_threshold = 0.6
        self.assertAlmostEqual(ctc.blank_threshold, 0.6, delta=1e-3)

        self.assertAlmostEqual(ctc.arc_threshold, 0.5, delta=1e-3)
        ctc.arc_threshold = 0.7
        self.assertAlmostEqual(ctc.arc_threshold, 0.7, delta=1e-3)

        self.assertEqual(ctc.sequence_length, 0)
        self.assertEqual(ctc.batch_width, 0)
        self.assertEqual(ctc.label_count, 0)
        self.assertAlmostEqual(layer.blank_threshold, 0.6, delta=1e-3)
        self.assertAlmostEqual(layer.arc_threshold, 0.7, delta=1e-3)

        ctc.get_best_sequence(0)

        input1 = neoml.Blob.asblob(math_engine, np.ones((3, 4, 5), dtype=np.float32), (3, 4, 1, 1, 1, 1, 5))
        input2 = neoml.Blob.asblob(math_engine, np.ones((4, ), dtype=np.int32), (1, 4, 1, 1, 1, 1, 1))
        inputs = {"source1": input1, "source2": input2}
        dnn.run(inputs)

    def test_gru(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        dnn = neoml.Dnn.Dnn(math_engine)
        source1 = neoml.Dnn.Source(dnn, "source1")
        gru = neoml.Dnn.Gru((source1,), 5, "gru")
        sink = neoml.Dnn.Sink(gru, "sink")
        layer = dnn.layers['gru']
        self.assertEqual(layer.name, 'gru')

        self.assertEqual(gru.hidden_size, 5)
        gru.hidden_size = 6
        self.assertEqual(gru.hidden_size, 6)
        self.assertEqual(layer.hidden_size, 6)

        main_weights = gru.main_weights
        gru.main_weights = main_weights

        main_free_term = gru.main_free_term
        gru.main_free_term = main_free_term

        gate_weights = gru.gate_weights
        gru.gate_weights = gate_weights

        gate_free_term = gru.gate_free_term
        gru.gate_free_term = gate_free_term

        input1 = neoml.Blob.asblob(math_engine, np.ones((3, 2, 3), dtype=np.float32), (3, 2, 1, 1, 1, 1, 3))
        inputs = {"source1": input1}
        outputs = dnn.run(inputs)
        out = outputs["sink"].asarray()
        self.assertEqual(out.shape, (3, 2, 6))

    def _test_eltwise(self, layer, check_f):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        dnn = neoml.Dnn.Dnn(math_engine)
        source1 = neoml.Dnn.Source(dnn, "source1")
        source2 = neoml.Dnn.Source(dnn, "source2")
        eltwise = getattr(neoml.Dnn, layer)((source1, source2), "eltwise")
        sink = neoml.Dnn.Sink(eltwise, "sink")
        layer = dnn.layers['eltwise']
        self.assertEqual(layer.name, 'eltwise')

        input1 = neoml.Blob.asblob(math_engine, 3 * np.ones((2, 3, 4), dtype=np.float32), (2, 3, 4, 1, 1, 1, 1))
        input2 = neoml.Blob.asblob(math_engine, 2 * np.ones((2, 3, 4), dtype=np.float32), (2, 3, 4, 1, 1, 1, 1))
        inputs = {"source1": input1, "source2": input2}
        outputs = dnn.run(inputs)
        out = outputs["sink"].asarray()

        self.assertTrue(check_f(out))

    def test_eltwise_sum(self):
        self._test_eltwise('EltwiseSum', lambda x: (x == 5).all())

    def test_eltwise_mul(self):
        self._test_eltwise('EltwiseMul', lambda x: (x == 6).all())

    def test_eltwise_negmul(self):
        self._test_eltwise('EltwiseNegMul', lambda x: (x == -4).all())

    def test_eltwise_max(self):
        self._test_eltwise('EltwiseMax', lambda x: (x == 3).all())

    def test_conv(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        dnn = neoml.Dnn.Dnn(math_engine)
        source1 = neoml.Dnn.Source(dnn, "source1")
        source2 = neoml.Dnn.Source(dnn, "source2")
        conv = neoml.Dnn.Conv((source1, source2), 16, (6, 6), (7, 7), (8, 8), (9, 9), False, "conv")
        sink1 = neoml.Dnn.Sink((conv, 0), "sink1")
        sink2 = neoml.Dnn.Sink((conv, 1), "sink2")
        layer = dnn.layers['conv']
        self.assertEqual(layer.name, 'conv')

        self.assertEqual(conv.filter_count, 16)
        conv.filter_count = 17
        self.assertEqual(conv.filter_count, 17)
        self.assertEqual(layer.filter_count, 17)

        self.assertEqual(conv.filter_size, (6, 6))
        self.assertEqual(layer.filter_size, (6, 6))
        conv.filter_size = (3, 3)
        self.assertEqual(conv.filter_size, (3, 3))

        self.assertEqual(conv.stride_size, (7, 7))
        conv.stride_size = (2, 2)
        self.assertEqual(conv.stride_size, (2, 2))
        self.assertEqual(layer.stride_size, (2, 2))

        self.assertEqual(conv.padding_size, (8, 8))
        conv.padding_size = (1, 1)
        self.assertEqual(conv.padding_size, (1, 1))

        self.assertEqual(conv.dilation_size, (9, 9))
        conv.dilation_size = (1, 1)
        self.assertEqual(conv.dilation_size, (1, 1))

        input1 = neoml.Blob.asblob(math_engine, np.ones((2, 3, 4, 5, 5), dtype=np.float32), (2, 3, 4, 5, 5, 1, 1))
        input2 = neoml.Blob.asblob(math_engine, np.ones((2, 3, 4, 5, 5), dtype=np.float32), (2, 3, 4, 5, 5, 1, 1))
        inputs = {"source1": input1, "source2": input2}
        outputs = dnn.run(inputs)
        out1 = outputs["sink1"].asarray()
        out2 = outputs["sink2"].asarray()
        self.assertEqual(out1.shape, (2, 3, 4, 3, 3, 17))
        self.assertEqual(out2.shape, (2, 3, 4, 3, 3, 17))

    def test_transposed_conv(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        dnn = neoml.Dnn.Dnn(math_engine)
        source1 = neoml.Dnn.Source(dnn, "source1")
        source2 = neoml.Dnn.Source(dnn, "source2")
        conv = neoml.Dnn.TransposedConv((source1, source2), 16, (6, 6), (7, 7), (8, 8), (9, 9), False, "conv")
        sink1 = neoml.Dnn.Sink((conv, 0), "sink1")
        sink2 = neoml.Dnn.Sink((conv, 1), "sink2")
        layer = dnn.layers['conv']
        self.assertEqual(layer.name, 'conv')

        self.assertEqual(conv.filter_count, 16)
        conv.filter_count = 17
        self.assertEqual(conv.filter_count, 17)
        self.assertEqual(layer.filter_count, 17)

        self.assertEqual(conv.filter_size, (6, 6))
        self.assertEqual(layer.filter_size, (6, 6))
        conv.filter_size = (3, 3)
        self.assertEqual(conv.filter_size, (3, 3))

        self.assertEqual(conv.stride_size, (7, 7))
        conv.stride_size = (2, 2)
        self.assertEqual(conv.stride_size, (2, 2))

        self.assertEqual(conv.padding_size, (8, 8))
        conv.padding_size = (1, 1)
        self.assertEqual(conv.padding_size, (1, 1))

        self.assertEqual(conv.dilation_size, (9, 9))
        conv.dilation_size = (1, 1)
        self.assertEqual(conv.dilation_size, (1, 1))

        input1 = neoml.Blob.asblob(math_engine, np.ones((2, 3, 4, 5, 5), dtype=np.float32), (2, 3, 4, 5, 5, 1, 1))
        input2 = neoml.Blob.asblob(math_engine, np.ones((2, 3, 4, 5, 5), dtype=np.float32), (2, 3, 4, 5, 5, 1, 1))
        inputs = {"source1": input1, "source2": input2}
        outputs = dnn.run(inputs)
        out1 = outputs["sink1"].asarray()
        out2 = outputs["sink2"].asarray()
        self.assertEqual(out1.shape, (2, 3, 4, 9, 9, 17))
        self.assertEqual(out2.shape, (2, 3, 4, 9, 9, 17))

    def test_channelwise_conv(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        dnn = neoml.Dnn.Dnn(math_engine)
        source1 = neoml.Dnn.Source(dnn, "source1")
        source2 = neoml.Dnn.Source(dnn, "source2")
        conv = neoml.Dnn.ChannelwiseConv((source1, source2), 16, (6, 6), (7, 7), (8, 8), False, "conv")
        sink1 = neoml.Dnn.Sink((conv, 0), "sink1")
        sink2 = neoml.Dnn.Sink((conv, 1), "sink2")
        layer = dnn.layers['conv']
        self.assertEqual(layer.name, 'conv')

        self.assertEqual(conv.filter_count, 16)
        conv.filter_count = 17
        self.assertEqual(conv.filter_count, 17)
        self.assertEqual(layer.filter_count, 17)

        self.assertEqual(conv.filter_size, (6, 6))
        self.assertEqual(layer.filter_size, (6, 6))
        conv.filter_size = (3, 3)
        self.assertEqual(conv.filter_size, (3, 3))

        self.assertEqual(conv.stride_size, (7, 7))
        conv.stride_size = (2, 2)
        self.assertEqual(conv.stride_size, (2, 2))

        self.assertEqual(conv.padding_size, (8, 8))
        conv.padding_size = (1, 1)
        self.assertEqual(conv.padding_size, (1, 1))

        input1 = neoml.Blob.asblob(math_engine, np.ones((2, 3, 4, 5, 5, 7), dtype=np.float32), (2, 3, 4, 5, 5, 1, 7))
        input2 = neoml.Blob.asblob(math_engine, np.ones((2, 3, 4, 5, 5, 7), dtype=np.float32), (2, 3, 4, 5, 5, 1, 7))
        inputs = {"source1": input1, "source2": input2}
        outputs = dnn.run(inputs)
        out1 = outputs["sink1"].asarray()
        out2 = outputs["sink2"].asarray()
        self.assertEqual(out1.shape, (2, 3, 4, 3, 3, 7))
        self.assertEqual(out2.shape, (2, 3, 4, 3, 3, 7))

    def test_time_conv(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        dnn = neoml.Dnn.Dnn(math_engine)
        source1 = neoml.Dnn.Source(dnn, "source1")
        source2 = neoml.Dnn.Source(dnn, "source2")
        conv = neoml.Dnn.TimeConv((source1, source2), 16, 6, 7, 8, 19, 9, "conv")
        sink1 = neoml.Dnn.Sink((conv, 0), "sink1")
        sink2 = neoml.Dnn.Sink((conv, 1), "sink2")
        layer = dnn.layers['conv']
        self.assertEqual(layer.name, 'conv')

        self.assertEqual(conv.filter_count, 16)
        conv.filter_count = 8
        self.assertEqual(conv.filter_count, 8)
        self.assertEqual(layer.filter_count, 8)

        self.assertEqual(conv.filter_size, 6)
        self.assertEqual(layer.filter_size, 6)
        conv.filter_size = 3
        self.assertEqual(conv.filter_size, 3)

        self.assertEqual(conv.padding_front, 7)
        conv.padding_front = 2
        self.assertEqual(conv.padding_front, 2)

        self.assertEqual(conv.padding_back, 8)
        conv.padding_back = 2
        self.assertEqual(conv.padding_back, 2)
        self.assertEqual(layer.padding_back, 2)

        self.assertEqual(conv.stride, 9)
        conv.stride = 3
        self.assertEqual(conv.stride, 3)

        self.assertEqual(conv.dilation, 19)
        self.assertEqual(layer.dilation, 19)
        conv.dilation = 5
        self.assertEqual(conv.dilation, 5)

        input1 = neoml.Blob.asblob(math_engine, np.ones((9, 3, 3, 3), dtype=np.float32), (9, 3, 1, 3, 3, 1, 1))
        input2 = neoml.Blob.asblob(math_engine, np.ones((9, 3, 3, 3), dtype=np.float32), (9, 3, 1, 3, 3, 1, 1))
        inputs = {"source1": input1, "source2": input2}
        outputs = dnn.run(inputs)
        out1 = outputs["sink1"].asarray()
        out2 = outputs["sink2"].asarray()
        self.assertEqual(out1.shape, (3, 8))
        self.assertEqual(out2.shape, (3, 8))

    def test_conv3d(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        dnn = neoml.Dnn.Dnn(math_engine)
        source1 = neoml.Dnn.Source(dnn, "source1")
        source2 = neoml.Dnn.Source(dnn, "source2")
        conv = neoml.Dnn.Conv3D((source1, source2), 16, (8, 6, 2), (9, 7, 2), (8, 8, 4), False, "conv")
        sink1 = neoml.Dnn.Sink((conv, 0), "sink1")
        sink2 = neoml.Dnn.Sink((conv, 1), "sink2")
        layer = dnn.layers['conv']
        self.assertEqual(layer.name, 'conv')

        self.assertEqual(conv.filter_count, 16)
        self.assertEqual(layer.filter_count, 16)
        conv.filter_count = 7
        self.assertEqual(conv.filter_count, 7)

        self.assertEqual(conv.filter_size, (8, 6, 2))
        conv.filter_size = (4, 3, 2)
        self.assertEqual(conv.filter_size, (4, 3, 2))
        self.assertEqual(layer.filter_size, (4, 3, 2))

        self.assertEqual(conv.stride_size, (9, 7, 2))
        conv.stride_size = (2, 2, 3)
        self.assertEqual(conv.stride_size, (2, 2, 3))

        self.assertEqual(conv.padding_size, (8, 8, 4))
        conv.padding_size = (2, 1, 1)
        self.assertEqual(conv.padding_size, (2, 1, 1))

        input1 = neoml.Blob.asblob(math_engine, np.ones((9, 3, 8, 6, 4), dtype=np.float32), (9, 3, 1, 8, 6, 4, 1))
        input2 = neoml.Blob.asblob(math_engine, np.ones((9, 3, 8, 6, 4), dtype=np.float32), (9, 3, 1, 8, 6, 4, 1))
        inputs = {"source1": input1, "source2": input2}
        outputs = dnn.run(inputs)
        out1 = outputs["sink1"].asarray()
        out2 = outputs["sink2"].asarray()
        self.assertEqual(out1.shape, (9, 3, 5, 3, 2, 7))
        self.assertEqual(out2.shape, (9, 3, 5, 3, 2, 7))

    def test_transposedconv3d(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        dnn = neoml.Dnn.Dnn(math_engine)
        source1 = neoml.Dnn.Source(dnn, "source1")
        source2 = neoml.Dnn.Source(dnn, "source2")
        conv = neoml.Dnn.TransposedConv3D((source1, source2), 16, (8, 6, 2), (9, 7, 2), (8, 8, 4), False, "conv")
        sink1 = neoml.Dnn.Sink((conv, 0), "sink1")
        sink2 = neoml.Dnn.Sink((conv, 1), "sink2")
        layer = dnn.layers['conv']
        self.assertEqual(layer.name, 'conv')

        self.assertEqual(conv.filter_count, 16)
        self.assertEqual(layer.filter_count, 16)
        conv.filter_count = 7
        self.assertEqual(conv.filter_count, 7)

        self.assertEqual(conv.filter_size, (8, 6, 2))
        conv.filter_size = (4, 3, 2)
        self.assertEqual(conv.filter_size, (4, 3, 2))
        self.assertEqual(layer.filter_size, (4, 3, 2))

        self.assertEqual(conv.stride_size, (9, 7, 2))
        conv.stride_size = (2, 2, 3)
        self.assertEqual(conv.stride_size, (2, 2, 3))

        self.assertEqual(conv.padding_size, (8, 8, 4))
        conv.padding_size = (2, 1, 1)
        self.assertEqual(conv.padding_size, (2, 1, 1))

        input1 = neoml.Blob.asblob(math_engine, np.ones((9, 3, 8, 6, 4), dtype=np.float32), (9, 3, 1, 8, 6, 4, 1))
        input2 = neoml.Blob.asblob(math_engine, np.ones((9, 3, 8, 6, 4), dtype=np.float32), (9, 3, 1, 8, 6, 4, 1))
        inputs = {"source1": input1, "source2": input2}
        outputs = dnn.run(inputs)
        out1 = outputs["sink1"].asarray()
        out2 = outputs["sink2"].asarray()
        self.assertEqual(out1.shape, (9, 3, 14, 11, 9, 7))
        self.assertEqual(out2.shape, (9, 3, 14, 11, 9, 7))

class PoolingTestCase(TestCase):
    def _test_pooling(self, layer, init_params={}, changed_params={},
                      input_shape=(2, 1, 2, 3, 5, 4, 2)):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        dnn = neoml.Dnn.Dnn(math_engine)
        source = neoml.Dnn.Source(dnn, "source")
        pooling = getattr(neoml.Dnn, layer)(source, name="pooling", **init_params)
        sink = neoml.Dnn.Sink(pooling, "sink")
        layer = dnn.layers['pooling']
        self.assertEqual(layer.name, 'pooling')

        for k,v in init_params.items():
            self.assertAlmostEqual(getattr(pooling, k), v, delta=1e-3,
                                   msg='Initial param {} of {} differs'.format(k, layer))
            self.assertEqual(getattr(pooling, k), getattr(layer, k))

        for k,v in changed_params.items():
            setattr(pooling, k, v)
            self.assertAlmostEqual(getattr(pooling, k), v, delta=1e-3,
                                   msg='Changed param {} of {} differs'.format(k, layer))
            self.assertEqual(getattr(pooling, k), getattr(layer, k))

        input = neoml.Blob.asblob(math_engine, np.ones(input_shape, dtype=np.float32), input_shape)
        inputs = {"source": input}
        outputs = dnn.run(inputs)
        out = outputs["sink"].asarray()
        return out

    def test_max_pooling(self):
        out = self._test_pooling('MaxPooling',
                                 dict(filter_size=(5, 5), stride_size=(3, 4)),
                                 dict(filter_size=(2, 2), stride_size=(2, 1)))
        self.assertTrue(np.equal(out, np.ones((2, 2, 4, 4, 2), dtype=np.float32)).all())

    def test_mean_pooling(self):
        out = self._test_pooling('MeanPooling',
                                 dict(filter_size=(5, 5), stride_size=(3, 4)),
                                 dict(filter_size=(2, 2), stride_size=(2, 1)))
        self.assertTrue(np.equal(out, np.ones((2, 2, 4, 4, 2), dtype=np.float32)).all())

    def test_max_pooling3d(self):
        out = self._test_pooling('MaxPooling3D',
                                 dict(filter_size=(5, 5, 5), stride_size=(3, 4, 5)),
                                 dict(filter_size=(2, 2, 2), stride_size=(2, 1, 1)))
        self.assertTrue(np.equal(out, np.ones((2, 2, 4, 3, 2), dtype=np.float32)).all())
        

    def test_mean_pooling3d(self):
        out = self._test_pooling('MeanPooling3D',
                                 dict(filter_size=(5, 5, 5), stride_size=(3, 4, 5)),
                                 dict(filter_size=(2, 2, 2), stride_size=(2, 1, 1)))
        self.assertTrue(np.equal(out, np.ones((2, 2, 4, 3, 2), dtype=np.float32)).all())

    def test_global_max_pooling(self):
        out = self._test_pooling('GlobalMaxPooling',
                                 dict(max_count=5),
                                 dict(max_count=6))
        self.assertTrue(np.equal(out, np.ones((2, 2, 6, 2), dtype=np.float32)).all())

    def test_global_mean_pooling(self):
        out = self._test_pooling('GlobalMeanPooling')
        self.assertTrue(np.equal(out, np.ones((2, 2, 2), dtype=np.float32)).all())

    def test_max_over_time_pooling(self):
        out = self._test_pooling('MaxOverTimePooling',
                                 dict(filter_len=3, stride_len=5),
                                 dict(filter_len=2, stride_len=1))
        self.assertTrue(np.equal(out, np.ones((2, 3, 5, 4, 2), dtype=np.float32)).all())

    def test_projection_pooling(self):
        out = self._test_pooling('ProjectionPooling',
                                 dict(dimension="width", original_size=True),
                                 dict(dimension="channels", original_size=False),
                                 input_shape=(1, 2, 3, 5, 4, 1, 2))
        self.assertTrue(np.equal(out, np.ones((2, 3, 5, 4), dtype=np.float32)).all())

    def test_object_norm(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        dnn = neoml.Dnn.Dnn(math_engine)
        source1 = neoml.Dnn.Source(dnn, "source1")
        norm = neoml.Dnn.ObjectNormalization(source1, 0.01, "norm")
        sink = neoml.Dnn.Sink(norm, "sink")
        layer = dnn.layers['norm']
        self.assertEqual(layer.name, 'norm')

        input1 = neoml.Blob.asblob(math_engine, np.ones((2, 3, 4, 5, 6), dtype=np.float32), (2, 3, 1, 4, 5, 1, 6))

        inputs = {"source1": input1}
        outputs = dnn.run(inputs)
        a = outputs["sink"]

        self.assertAlmostEqual(norm.epsilon, 0.01, delta=1e-3)
        self.assertEqual(norm.epsilon, layer.epsilon)
        norm.epsilon = 0.1

        blob = neoml.Blob.asblob(math_engine, np.ones((4, 5, 6), dtype=np.float32), (1, 1, 1, 1, 1, 1, 120))

        scale = norm.scale
        norm.scale = blob
        bias = norm.bias
        norm.bias = blob

        self.assertEqual(scale.shape, bias.shape)
        self.assertEqual(a.shape, input1.shape)

    def test_positional_embedding(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        dnn = neoml.Dnn.Dnn(math_engine)
        source1 = neoml.Dnn.Source(dnn, "source1")
        pos = neoml.Dnn.PositionalEmbedding(source1, "transformers", "pos")
        sink = neoml.Dnn.Sink(pos, "sink")
        layer = dnn.layers['pos']
        self.assertEqual(layer.name, 'pos')

        input1 = neoml.Blob.asblob(math_engine, np.ones((3, 4, 6), dtype=np.float32), (1, 3, 4, 1, 1, 1, 6))

        inputs = {"source1": input1}
        outputs = dnn.run(inputs)
        a = outputs["sink"]

        self.assertEqual(pos.type, "transformers")
        self.assertEqual(layer.type, "transformers")
        pos.type = "learnable_addition"

        blob = neoml.Blob.asblob(math_engine, np.ones((4, 5, 6), dtype=np.float32), (1, 1, 1, 1, 1, 1, 120))

        addends = pos.addends
        pos.addends = blob
 
        self.assertEqual(a.shape, input1.shape)

    def test_precision_recall(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        dnn = neoml.Dnn.Dnn(math_engine)
        source1 = neoml.Dnn.Source(dnn, "source1")
        source2 = neoml.Dnn.Source(dnn, "source2")
        pre = neoml.Dnn.PrecisionRecall((source1, source2), False, "pre")
        sink = neoml.Dnn.Sink(pre, "sink")
        layer = dnn.layers['pre']
        self.assertEqual(layer.name, 'pre')
        self.assertEqual(pre.reset, False)
        self.assertEqual(layer.reset, False)
        pre.reset = True

        input1 = neoml.Blob.asblob(math_engine, np.ones((2, 3, 4), dtype=np.float32), (2, 3, 4, 1, 1, 1, 1))
        input2 = neoml.Blob.asblob(math_engine, np.ones((2, 3, 4), dtype=np.float32), (2, 3, 4, 1, 1, 1, 1))

        inputs = {"source1": input1, "source2": input2}
        outputs = dnn.run(inputs)
        a = outputs["sink"]

        self.assertEqual(pre.result, [24, 24, 0, 0])
        self.assertEqual(layer.result, [24, 24, 0, 0])
        self.assertEqual(a.size, 4)

    def test_qrnn(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        dnn = neoml.Dnn.Dnn(math_engine)
        source1 = neoml.Dnn.Source(dnn, "source1")
        source2 = neoml.Dnn.Source(dnn, "source2")
        qrnn = neoml.Dnn.Qrnn((source1, source2), 'fo', 7, 4, 2, (1, 1), "sigmoid", 0.6, "direct", "qrnn")
        filter = neoml.Blob.asblob(math_engine, np.ones((21, 5, 6), dtype=np.float32), (1, 21, 1, 4, 1, 1, 6))
        qrnn.filter = filter
        free_term = neoml.Blob.asblob(math_engine, np.ones((21,), dtype=np.float32), (1, 21, 1, 1, 1, 1, 1))
        qrnn.free_term = free_term
        layer = dnn.layers['qrnn']
        self.assertEqual(layer.name, 'qrnn')

        sink = neoml.Dnn.Sink(qrnn, "sink")

        input1 = neoml.Blob.asblob(math_engine, np.ones((2, 3, 6), dtype=np.float32), (2, 3, 1, 1, 1, 1, 6))
        input2 = neoml.Blob.asblob(math_engine, np.ones((3, 7), dtype=np.float32), (1, 3, 1, 1, 1, 1, 7))

        inputs = {"source1": input1, "source2": input2}
        outputs = dnn.run(inputs)
        a = outputs["sink"]

        self.assertEqual(qrnn.hidden_size, 7)
        self.assertEqual(layer.hidden_size, 7)
        self.assertEqual(qrnn.window_size, 4)
        self.assertEqual(qrnn.stride, 2)
        self.assertEqual(qrnn.padding_front, 1)
        self.assertEqual(layer.padding_front, 1)
        self.assertEqual(qrnn.padding_back, 1)
        self.assertEqual(qrnn.activation, "sigmoid")
        self.assertAlmostEqual(qrnn.dropout, 0.6, delta=1e-3)
        self.assertEqual(qrnn.recurrent_mode, "direct")

        self.assertEqual(a.shape, (1, 3, 1, 1, 1, 1, 7 ))

    def test_reorg(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        dnn = neoml.Dnn.Dnn(math_engine)
        source = neoml.Dnn.Source(dnn, "source1")
        reorg = neoml.Dnn.Reorg(source, 3, "reorg")
        sink = neoml.Dnn.Sink(reorg, "sink")
        layer = dnn.layers['reorg']
        self.assertEqual(layer.name, 'reorg')

        layer.stride = 2
        self.assertEqual(reorg.stride, 2)
        self.assertEqual(layer.stride, 2)

        input1 = neoml.Blob.asblob(math_engine, np.ones((1, 3, 1, 8, 8, 1, 4), dtype=np.float32), (1, 3, 1, 8, 8, 1, 4))
        inputs = {"source1": input1}
        outputs = dnn.run(inputs)
        out = outputs["sink"]
        self.assertEqual(out.shape, (1, 3, 1, 4, 4, 1, 16))

    def test_repeat_count(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        dnn = neoml.Dnn.Dnn(math_engine)
        source = neoml.Dnn.Source(dnn, "source1")
        repeat_count = neoml.Dnn.RepeatSequence(source, 5, "layer")
        sink = neoml.Dnn.Sink(repeat_count, "sink")
        layer = dnn.layers['layer']
        self.assertEqual(layer.name, 'layer')

        self.assertEqual(repeat_count.repeat_count, 5)
        self.assertEqual(layer.repeat_count, 5)
        layer.stride = 6

        input1 = neoml.Blob.asblob(math_engine, np.ones((2, 1, 3, 2, 1, 1, 2), dtype=np.float32), (2, 1, 3, 2, 1, 1, 2))
        inputs = {"source1": input1}
        outputs = dnn.run(inputs)
        out = outputs["sink"].asarray()
        self.assertTrue(np.equal(out, np.ones((10, 3, 2, 2), dtype=np.float32)).all())

    def test_softmax(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        dnn = neoml.Dnn.Dnn(math_engine)
        source = neoml.Dnn.Source(dnn, "source1")
        softmax = neoml.Dnn.Softmax(source, "list_size", "layer")
        sink = neoml.Dnn.Sink(softmax, "sink")
        layer = dnn.layers['layer']
        self.assertEqual(layer.name, 'layer')

        self.assertEqual(softmax.area, "list_size")
        self.assertEqual(layer.area, "list_size")
        layer.stride = "channels"

        input1 = neoml.Blob.asblob(math_engine, np.ones((2, 10), dtype=np.float32), (2, 1, 10, 1, 1, 1, 1))
        inputs = {"source1": input1}
        outputs = dnn.run(inputs)
        out = outputs["sink"].asarray()
        self.assertTrue(np.equal(out, 0.1 * np.ones((2, 10), dtype=np.float32)).all())

    def test_split(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        dnn = neoml.Dnn.Dnn(math_engine)
        source = neoml.Dnn.Source(dnn, "source1")

        split_types = ("SplitBatchWidth", "SplitHeight", "SplitWidth", "SplitDepth", "SplitChannels")
        for i, split_name in enumerate(split_types):
            split = getattr(neoml.Dnn, split_name)(source, (2, 3), split_name)
            sink = neoml.Dnn.Sink((split, 0), "sink{}".format(2 * i))
            sink = neoml.Dnn.Sink((split, 1), "sink{}".format(2 * i + 1))

        arr = np.ones((5, 5, 5, 5, 5), dtype=np.float32)
        input1 = neoml.Blob.asblob(math_engine, arr, (1, 5, 1, 5, 5, 5, 5))
        inputs = {"source1": input1}
        outputs = dnn.run(inputs)

        for i, split_name in enumerate(split_types):
            expected1, expected2, _ = np.split(arr, (2, 3), i)
            layer = dnn.layers[split_name]
            self.assertEqual(layer.name, split_name)
            out1 = outputs["sink{}".format(2 * i)].asarray()
            out2 = outputs["sink{}".format(2 * i + 1)].asarray()
            self.assertTrue(np.equal(out1, expected1).all())
            self.assertTrue(np.equal(out2, expected2).all())

    def test_sub_sequence(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        dnn = neoml.Dnn.Dnn(math_engine)
        source = neoml.Dnn.Source(dnn, "source1")
        subsequence = neoml.Dnn.SubSequence(source, 1, 3, "layer")
        reverse = neoml.Dnn.ReverseSequence(source, "layer2")
        sink = neoml.Dnn.Sink(subsequence, "sink")
        sink2 = neoml.Dnn.Sink(reverse, "sink2")
        layer1 = dnn.layers['layer']
        self.assertEqual(layer1.name, 'layer')
        layer2 = dnn.layers['layer2']
        self.assertEqual(layer2.name, 'layer2')

        self.assertEqual(layer1.start_pos, 1)
        subsequence.start_pos = 2
        self.assertEqual(subsequence.start_pos, 2)

        self.assertEqual(subsequence.length, 3)
        subsequence.length = 2
        self.assertEqual(layer1.length, 2)

        input1 = neoml.Blob.asblob(math_engine, np.ones((5, 3, 4), dtype=np.float32), (5, 1, 3, 1, 4, 1, 1))
        inputs = {"source1": input1}
        outputs = dnn.run(inputs)
        out1 = outputs["sink"].asarray()
        self.assertEqual(out1.shape, (2, 3, 4))
        out2 = outputs["sink2"].asarray()
        self.assertEqual(out2.shape, (5, 3, 4))

    def test_upsampling2d(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        dnn = neoml.Dnn.Dnn(math_engine)
        source = neoml.Dnn.Source(dnn, "source1")
        upsampling = neoml.Dnn.Upsampling2D(source, 4, 5, "layer")
        sink = neoml.Dnn.Sink(upsampling, "sink")
        layer = dnn.layers['layer']
        self.assertEqual(layer.name, 'layer')

        self.assertEqual(layer.height_copy_count, 4)
        layer.height_copy_count = 2
        self.assertEqual(upsampling.height_copy_count, 2)

        self.assertEqual(upsampling.width_copy_count, 5)
        upsampling.width_copy_count = 3
        self.assertEqual(layer.width_copy_count, 3)

        input1 = neoml.Blob.asblob(math_engine, np.ones((2, 3, 2), dtype=np.float32), (1, 2, 3, 1, 2, 1, 1))
        inputs = {"source1": input1}
        outputs = dnn.run(inputs)
        out = outputs["sink"].asarray()
        self.assertTrue(np.equal(out, np.ones((2, 3, 2, 6), dtype=np.float32)).all())

    def test_transform(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        dnn = neoml.Dnn.Dnn(math_engine)
        source = neoml.Dnn.Source(dnn, "source1")
        trans = [("set", 3), ("set", 1), ("set", 5), ("set", 4), ("divide", 4), ("multiply", 2), ("remainder", 4)]
        transform = neoml.Dnn.Transform(source, trans, "layer")
        sink = neoml.Dnn.Sink(transform, "sink")
        layer = dnn.layers['layer']
        self.assertEqual(layer.name, 'layer')

        self.assertEqual(transform.transforms, trans)
        self.assertEqual(layer.transforms, trans)
        layer.transforms = trans

        input1 = neoml.Blob.asblob(math_engine, np.ones((2, 3, 4, 5), dtype=np.float32), (1, 2, 3, 1, 4, 1, 5))
        inputs = {"source1": input1}
        outputs = dnn.run(inputs)
        out = outputs["sink"].asarray()
        self.assertEqual(out.shape, (3, 5, 4, 2))

    def test_transpose(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        dnn = neoml.Dnn.Dnn(math_engine)
        source = neoml.Dnn.Source(dnn, "source1")
        transpose = neoml.Dnn.Transpose(source, "height", "width", "layer")
        sink = neoml.Dnn.Sink(transpose, "sink")
        layer = dnn.layers['layer']
        self.assertEqual(layer.name, 'layer')

        self.assertEqual(transpose.first_dim, "height")
        layer.first_dim = "depth"
        self.assertEqual(layer.first_dim, "depth")

        self.assertEqual(transpose.second_dim, "width")
        layer.second_dim = "channels"
        self.assertEqual(layer.second_dim, "channels")

        input1 = neoml.Blob.asblob(math_engine, np.ones((2, 3, 4, 5), dtype=np.float32), (1, 2, 1, 2, 1, 5, 5))
        inputs = {"source1": input1}
        outputs = dnn.run(inputs)
        out = outputs["sink"].asarray()
        self.assertTrue(np.equal(out, np.ones((2, 2, 5, 5), dtype=np.float32)).all())

    def test_sequence_sum(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        dnn = neoml.Dnn.Dnn(math_engine)
        source = neoml.Dnn.Source(dnn, "source1")
        sequence_sum = neoml.Dnn.SequenceSum(source, "layer")
        sink = neoml.Dnn.Sink(sequence_sum, "sink")
        layer = dnn.layers['layer']
        self.assertEqual(layer.name, 'layer')

        input1 = neoml.Blob.asblob(math_engine, np.ones((5, 2, 2), dtype=np.float32), (5, 1, 1, 2, 1, 1, 2))
        inputs = {"source1": input1}
        outputs = dnn.run(inputs)
        out = outputs["sink"].asarray()
        self.assertTrue(np.equal(out, [[5., 5.], [5., 5.]]).all())

    def test_irnn(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        dnn = neoml.Dnn.Dnn(math_engine)

        batch_length = 12
        batch_width = 6
        channels_in = 5
        hidden_size = 10
        identity_scale = 1e-1
        input_weight_std = 1e-4
        name = "irnn_test_name"

        source = neoml.Dnn.Source(dnn, "source")
        irnn = neoml.Dnn.Irnn(source, hidden_size, identity_scale, input_weight_std, True, name)
        sink = neoml.Dnn.Sink(irnn, "sink")
        layer = dnn.layers[name]
        self.assertEqual(layer.name, name)

        input1 = neoml.Blob.asblob(math_engine, np.ones((batch_length, batch_width, channels_in), dtype=np.float32),
            (batch_length, batch_width, 1, 1, 1, 1, channels_in))

        inputs = {"source": input1}
        outputs = dnn.run(inputs)
        a = outputs["sink"]

        self.assertEqual(irnn.hidden_size, hidden_size)
        self.assertEqual(layer.hidden_size, hidden_size)
        self.assertAlmostEqual(irnn.identity_scale, identity_scale, delta=1e-5)
        self.assertAlmostEqual(irnn.input_weight_std, input_weight_std, delta=1e-5)
        self.assertEqual(a.shape, (batch_length, batch_width, 1, 1, 1, 1, hidden_size))


class MulLossCalculator(neoml.Dnn.CustomLossCalculatorBase):
    def calc(self, data, labels):
        return neoml.AutoDiff.mul(data - labels, data - labels)


class BinaryCrossEntropyLossCalculator(neoml.Dnn.CustomLossCalculatorBase):
    def calc(self, data, labels):
        return neoml.AutoDiff.binary_cross_entropy(data, labels, True)


class LossTestCase(TestCase):
    def _test_loss(self, layer, kwargs={},
                   n_classes=2,
                   labels_type=np.float32,
                   last_loss=0.):
        shape = (2, 3, 1, 1, 1, 1, 1 if n_classes == 2 else n_classes)
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        dnn = neoml.Dnn.Dnn(math_engine)
        source1 = neoml.Dnn.Source(dnn, "source1")
        source2 = neoml.Dnn.Source(dnn, "source2")
        source3 = neoml.Dnn.Source(dnn, "source3")
        loss = getattr(neoml.Dnn, layer)((source1, source2, source3), name="loss", **kwargs)
        layer = dnn.layers['loss']
        self.assertEqual(layer.name, 'loss')

        input1 = neoml.Blob.asblob(math_engine, np.ones(shape, dtype=np.float32), shape)
        input2 = neoml.Blob.asblob(math_engine, np.ones(shape, dtype=labels_type), shape)
        input3 = neoml.Blob.asblob(math_engine, np.ones(shape, dtype=np.float32), shape)

        inputs = {"source1": input1, "source2": input2, "source3": input3}
        dnn.run(inputs)

        for k,v in kwargs.items():
            self.assertAlmostEqual(getattr(loss, k), v, delta=1e-3,
                                   msg='Field {} of {} differs'.format(k, layer))
            self.assertEqual(getattr(loss, k), getattr(layer, k))
        self.assertAlmostEqual(loss.last_loss, last_loss, delta=1e-3)
        self.assertAlmostEqual(layer.last_loss, last_loss, delta=1e-3)

    def _test_custom_loss(self, loss_calculator, result_loss):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        dnn = neoml.Dnn.Dnn(math_engine)
        shape = (2, 3, 1, 1, 1, 1, 1)
        source1 = neoml.Dnn.Source(dnn, "source1")
        source2 = neoml.Dnn.Source(dnn, "source2")
        source3 = neoml.Dnn.Source(dnn, "source3")
        loss = neoml.Dnn.CustomLoss((source1, source2, source3), name="loss", loss_weight=7.7,
                                    loss_calculator=loss_calculator)

        input1 = neoml.Blob.asblob(math_engine, np.ones(shape, dtype=np.float32), shape)
        input2 = neoml.Blob.asblob(math_engine, np.ones(shape, dtype=np.float32), shape)
        input3 = neoml.Blob.asblob(math_engine, np.ones(shape, dtype=np.float32), shape)

        inputs = {"source1": input1, "source2": input2, "source3": input3}

        dir = tempfile.mkdtemp()

        path = os.path.join(dir, 'custom_loss_dnn.arc')
        dnn.store_checkpoint(path)

        dnn_loaded = neoml.Dnn.Dnn(math_engine)
        dnn_loaded.load_checkpoint(path)

        os.remove(path)
        os.rmdir(dir)

        dnn_loaded.run(inputs)

        layer = dnn_loaded.layers['loss']
        self.assertEqual(layer.name, 'loss')

        self.assertAlmostEqual(layer.last_loss, result_loss, delta=1e-3)

    def test_custom_loss(self):
        import neoml.AutoDiff as ad
        for loss_calculator, result_loss in [
            (BinaryCrossEntropyLossCalculator(), 0.313261),
            (MulLossCalculator(), 0),
        ]:
            self._test_custom_loss(loss_calculator, result_loss)

    def test_autodiff_functions(self):
        import neoml.AutoDiff as ad
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        shape = (2, 3, 1, 1, 1, 2, 3)
        const0 = ad.const(math_engine, shape, 0)
        const2 = ad.const(math_engine, shape, 2)
        ones = np.ones(shape, dtype=np.float32)
        const_ones = ad.const(math_engine, shape, ones)
        blob = neoml.Blob.asblob(math_engine, ones, shape)
        
        self.assertTrue( np.equal( ad.add(const2, blob).asarray(), 3 * ones ).all() )
        self.assertTrue( np.equal( ad.add(2, blob).asarray(), 3 * ones ).all() )
        self.assertTrue( np.equal( (const2 + 3).asarray(), 5 * ones ).all() )
        self.assertTrue( np.equal( ad.sub(const2, blob).asarray(), ones ).all() )
        self.assertTrue( np.equal( ad.sub(const2, 0).asarray(), 2 * ones ).all() )
        self.assertTrue( np.equal( (3 - blob).asarray(), 2 * ones ).all() )
        self.assertTrue( np.equal( ad.mul(const2, 2).asarray(), 4 * ones ).all() )
        self.assertTrue( np.equal( ad.mul(2, blob).asarray(), 2 * ones ).all() )
        self.assertTrue( np.equal( (const0 * const2).asarray(), 0 * ones ).all() )
        self.assertTrue( np.equal( ad.div(2, const2).asarray(), ones ).all() )
        self.assertTrue( np.equal( ad.div(const2, 2).asarray(), ones ).all() )
        self.assertTrue( np.equal( (const2 / const_ones).asarray(), 2 * ones ).all() )
        self.assertTrue( np.equal( ad.max(const_ones, 2).asarray(), 2 * ones ).all() )
        self.assertEqual( ad.sum(blob).asarray(), 36 )
        self.assertTrue( np.equal( ad.neg(blob).asarray(), -ones ).all() )
        self.assertTrue( np.equal( (-blob).asarray(), -ones ).all() )
        self.assertTrue( np.equal( ad.abs(-blob).asarray(), ones ).all() )
        self.assertTrue( np.equal( ad.log(const_ones).asarray(), 0 * ones ).all() )
        self.assertTrue( np.equal( ad.exp(const0).asarray(), ones ).all() )
        self.assertTrue( np.equal( ad.clip(const2, 3, 4).asarray(), 3 * ones ).all() )
        self.assertTrue( np.equal( ad.top_k(const2, 3).asarray(), [2, 2, 2] ).all() )
        self.assertTrue( np.equal( ad.binary_cross_entropy(const0, const0, False).asarray(), 0 * ones ).all() )

    def test_cross_entropy_loss(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        dnn = neoml.Dnn.Dnn(math_engine)
        source1 = neoml.Dnn.Source(dnn, "source1")
        source2 = neoml.Dnn.Source(dnn, "source2")
        source3 = neoml.Dnn.Source(dnn, "source3")

        loss = neoml.Dnn.CrossEntropyLoss((source1, source2, source3), True, 7.7, "loss")
        layer = dnn.layers['loss']
        self.assertEqual(layer.name, 'loss')

        input1 = neoml.Blob.asblob(math_engine, np.ones((2, 3, 4, 2), dtype=np.float32), (2, 3, 4, 1, 1, 1, 2))
        input2 = neoml.Blob.asblob(math_engine, np.ones((2, 3, 4), dtype=np.int32), (2, 3, 4, 1, 1, 1, 1))
        input3 = neoml.Blob.asblob(math_engine, np.ones((2, 3, 4), dtype=np.float32), (2, 3, 4, 1, 1, 1, 1))

        inputs = {"source1": input1, "source2": input2, "source3": input3}
        dnn.run(inputs)

        self.assertEqual(loss.apply_softmax, True)
        self.assertAlmostEqual(loss.loss_weight, 7.7, delta=1e-3)
        self.assertAlmostEqual(loss.last_loss, 0.6931, delta=1e-3)
        self.assertEqual(loss.loss_weight, layer.loss_weight)
        self.assertEqual(loss.last_loss, layer.last_loss)

    def test_binary_cross_entropy_loss(self):
        self._test_loss('BinaryCrossEntropyLoss', 
                        dict(positive_weight=6.6, loss_weight=7.7),
                        last_loss=2.0675)

    def test_euclidean_loss(self):
        self._test_loss('EuclideanLoss', dict(loss_weight=7.7), last_loss=0.)

    def test_hinge_loss(self):
        self._test_loss('HingeLoss', dict(loss_weight=7.7), last_loss=0.)

    def test_squared_hinge_loss(self):
        self._test_loss('SquaredHingeLoss', dict(loss_weight=7.7), last_loss=0.)

    def test_focal_loss(self):
        self._test_loss('FocalLoss', 
                        dict(force=6.6, loss_weight=7.7),
                        n_classes=5,
                        last_loss=0.)

    def test_binary_focal_loss(self):
        self._test_loss('BinaryFocalLoss', dict(force=6.6, loss_weight=7.7),
                        last_loss=0.)

    def test_center_loss(self):
        self._test_loss('CenterLoss',
                        dict(rate=6.6, loss_weight=7.7, class_count=3),
                        last_loss=1.,
                        labels_type=np.int32)

    def test_multihinge_loss(self):
        self._test_loss('MultiHingeLoss', dict(loss_weight=7.7), last_loss=0.)

    def test_multisquaredhinge_loss(self):
        self._test_loss('MultiSquaredHingeLoss', dict(loss_weight=7.7), last_loss=0.)

class DnnTestCase(TestCase):
    def test_load_store(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        dnn = neoml.Dnn.Dnn(math_engine)
        dnn.solver = neoml.Dnn.AdaptiveGradient(math_engine)
        source = neoml.Dnn.Source(dnn, "source")
        argmax = neoml.Dnn.Argmax(source, name="argmax")
        sink = neoml.Dnn.Sink(argmax, "sink")

        self.assertTrue(len(dnn.input_layers), 1)
        self.assertTrue(len(dnn.layers), 3)
        self.assertTrue(len(dnn.output_layers), 1)

        dir = tempfile.mkdtemp()

        path = os.path.join(dir, 'dnn.arc')
        dnn.store_checkpoint(path)

        dnn_loaded = neoml.Dnn.Dnn(math_engine)
        dnn_loaded.load_checkpoint(path)

        os.remove(path)
        os.rmdir(dir)

        self.assertTrue(isinstance(dnn_loaded.solver, neoml.Dnn.AdaptiveGradient))
        self.assertTrue(len(dnn_loaded.input_layers), 1)
        self.assertTrue(len(dnn_loaded.layers), 3)
        self.assertTrue(len(dnn_loaded.output_layers), 1)

    def test_solver(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        dnn = neoml.Dnn.Dnn(math_engine)

        dnn.solver = neoml.Dnn.NesterovGradient(math_engine)
        self.assertTrue(isinstance(dnn.solver, neoml.Dnn.NesterovGradient))

        dnn.solver = neoml.Dnn.AdaptiveGradient(math_engine)
        self.assertTrue(isinstance(dnn.solver, neoml.Dnn.AdaptiveGradient))

        dnn.solver = neoml.Dnn.SimpleGradient(math_engine)
        self.assertTrue(isinstance(dnn.solver, neoml.Dnn.SimpleGradient))

    def test_initializer(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)

        random = neoml.Random.Random(0)

        dnn = neoml.Dnn.Dnn(math_engine, random)

        dnn.initializer = neoml.Dnn.Xavier(random)
        self.assertTrue(isinstance(dnn.initializer, neoml.Dnn.Xavier))

        dnn.initializer = neoml.Dnn.Uniform()
        self.assertTrue(isinstance(dnn.initializer, neoml.Dnn.Uniform))

    def test_math_engine(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        dnn = neoml.Dnn.Dnn(math_engine)
        self.assertTrue(isinstance(dnn.math_engine, neoml.MathEngine.CpuMathEngine))

    def test_properties(self):
        math_engine = neoml.MathEngine.CpuMathEngine(1)
        dnn = neoml.Dnn.Dnn(math_engine)
        source = neoml.Dnn.Source(dnn, "source")
        argmax = neoml.Dnn.Argmax(source, name="argmax")
        sink = neoml.Dnn.Sink(argmax, "sink")

        self.assertTrue(len(dnn.input_layers), 1)
        self.assertTrue(len(dnn.layers), 3)
        self.assertTrue(len(dnn.output_layers), 1)

class TraditionalTestCase(TestCase):
    def test_differential_evolution(self):
        from neoml.DifferentialEvolution import IntTraits, DoubleTraits, DifferentialEvolution
        def func(vec):
            return sum([x**2 for x in vec])

        for dim, param_traits, max_gen_count, result_traits, population in (
            (1, None, None, None, 50),
            (10, [IntTraits()] * 5 + [DoubleTraits()] * 5, 10, DoubleTraits(), 100),
        ):
            diff_evo = DifferentialEvolution(func, [-5] * dim, [5] * dim,
                param_traits=param_traits, result_traits=result_traits,
                max_generation_count=max_gen_count, population=population)
            diff_evo.build_next_generation()
            diff_evo.run()
            self.assertEqual(diff_evo.build_next_generation(), True)
            res_population = np.array(diff_evo.population)
            self.assertEqual(res_population.shape, (population, dim))
            eval_population = np.array(diff_evo.population_function_values)
            self.assertEqual(eval_population.shape, (population,))
            optimal_vector = np.array(diff_evo.optimal_vector)
            self.assertEqual(optimal_vector.shape, (dim,))

    def _test_classification_model(self, model, params, is_binary=False):
        X_dense = np.eye(20, 5, dtype=np.float32)
        X_dense_list = X_dense.tolist()
        X_sparse = sparse.csr_matrix(X_dense)
        val = 1 if is_binary else 3
        y = val * np.ones(20, dtype=np.int32)
        if not is_binary: # every class should be represented in dataset
            for i in range(3):
                y[i] = i
        weight = np.ones(20, dtype=np.float32)
        for X in (X_dense, X_dense_list, X_sparse):
            classifier = model(**params).train(X, y, weight)
            pred = classifier.classify(X[-3:])
            print(pred, np.argmax(pred))
            self.assertTrue(np.equal(np.argmax(pred), [val, val, val]).all())

    def _test_regression_model(self, model, params):
        X_dense = np.eye(20, 5, dtype=np.float32)
        X_dense_list = X_dense.tolist()
        X_sparse = sparse.csr_matrix(X_dense)
        y = np.ones(20, dtype=np.int32)
        weight = np.ones(20, dtype=np.float32)
        for X in (X_dense, X_dense_list, X_sparse):
            regressor = model(**params).train(X, y, weight)
            pred = regressor.predict(X[0:3])
            self.assertEqual(pred.shape, (3,))

    def test_gradient_boosting_classification(self):
        for loss, builder_type, thread_count, is_binary in itertools.product(
                ('binomial', 'exponential', 'squared_hinge', 'l2'),
                ('full', 'hist', 'multi_full'), (1, 4), (False, True)):
            self._test_classification_model(neoml.GradientBoost.GradientBoostClassifier,
                dict(loss=loss, iteration_count=10, builder_type=builder_type, thread_count=thread_count),
                is_binary=is_binary)

    def test_gradient_boosting_regression(self):
        for builder_type, thread_count in itertools.product(('full', 'hist'), (1, 4)):
            self._test_regression_model(neoml.GradientBoost.GradientBoostRegressor,
                dict(iteration_count=10, builder_type=builder_type, thread_count=thread_count))

    def test_decision_tree_classification(self):
        for criterion, is_binary in itertools.product(('gini', 'information_gain'), (False, True)):
            self._test_classification_model(neoml.DecisionTree.DecisionTreeClassifier,
                dict(criterion=criterion), is_binary=is_binary)
        for multiclass_mode in ('single_tree', 'one_vs_all', 'one_vs_one'):
            self._test_classification_model(neoml.DecisionTree.DecisionTreeClassifier, dict(multiclass_mode=multiclass_mode))

    def test_svm_classification(self):
        for kernel, thread_count, is_binary in itertools.product(('linear', 'poly', 'rbf', 'sigmoid'),
                                                                 (1, 4), (False, True)):
            self._test_classification_model(neoml.SVM.SvmClassifier,
                dict(kernel=kernel, thread_count=thread_count), is_binary=is_binary)
        for multiclass_mode in ('one_vs_all', 'one_vs_one'):
            print('svm ', multiclass_mode)
            self._test_classification_model(neoml.SVM.SvmClassifier, dict(multiclass_mode=multiclass_mode))

    def test_linear_classification(self):
        for loss, thread_count, is_binary in itertools.product(('binomial', 'squared_hinge', 'smoothed_hinge'),
                                                               (1, 4), (False, True)):
            self._test_classification_model(neoml.Linear.LinearClassifier,
                dict(loss=loss, thread_count=thread_count), is_binary=is_binary)
        for multiclass_mode in ('one_vs_all', 'one_vs_one'):
            self._test_classification_model(neoml.Linear.LinearClassifier, dict(multiclass_mode=multiclass_mode))

    def test_linear_regression(self):
        for thread_count in (1, 4):
            self._test_regression_model(neoml.Linear.LinearRegressor,
                dict(thread_count=thread_count))

    def test_cross_validation_score(self):
        from neoml.CrossValidation import cross_validation_score
        X_dense = np.eye(20, 5, dtype=np.float32)
        X_dense_list = X_dense.tolist()
        X_sparse = sparse.csr_matrix(X_dense)
        y = np.ones(20, dtype=np.int32)
        weight = np.ones(20, dtype=np.float32)
        for X in (X_dense, X_dense_list, X_sparse):
            for classifier, score in itertools.product(
                    ( neoml.Linear.LinearClassifier(),
                      neoml.GradientBoost.GradientBoostClassifier(),
                      neoml.SVM.SvmClassifier(),
                      neoml.DecisionTree.DecisionTreeClassifier()
                    ), ('accuracy', 'f1')):
                cv_score = cross_validation_score(classifier, X, y, weight, score, 5)
                self.assertEqual(cv_score.shape, (5,))

    def test_load_store(self):
        dir = tempfile.mkdtemp()
        for model_init, model_result in (
                (neoml.DecisionTree.DecisionTreeClassifier, neoml.DecisionTree.DecisionTreeClassificationModel),
                (neoml.GradientBoost.GradientBoostRegressor, neoml.GradientBoost.GradientBoostRegressionModel)): 
            
            path = os.path.join(dir, 'test')
            pickled = model_init().train([[1]], [1])
            with open(path, 'wb') as file:
                pickle.dump(pickled, file)
            with open(path, 'rb') as file:
                loaded = pickle.load(file)
            self.assertEqual(type(loaded), model_result)
            os.remove(path)
        os.rmdir(dir)

class ClusteringTestCase(TestCase):
    def _test_clusterize(self, method, params={}):
        X_dense = np.eye(20, 5, dtype=np.float32)
        X_dense_list = X_dense.tolist()
        X_sparse = sparse.csr_matrix(X_dense)
        weight = np.ones(20, dtype=np.float32)
        method = getattr(neoml.Clustering, method)(**params)
        for X in (X_dense, X_dense_list, X_sparse):
            clusters = method.clusterize(X, weight)
            self.assertEqual(clusters[0].shape, (20,))
            self.assertEqual(clusters[1].shape[1], 5)
            self.assertEqual(clusters[2].shape[1], 5)

    def test_first_come(self):
        self._test_clusterize('FirstCome', dict(threshold=0.01))

    def test_hierarchical(self):
        self._test_clusterize('Hierarchical', dict(max_cluster_distance=2, min_cluster_count=6))

    def test_iso_data(self):
        self._test_clusterize('IsoData', dict(init_cluster_count=6, max_cluster_count=10, 
            min_cluster_size=1, max_iteration_count=10, min_cluster_distance=0.1, 
            max_cluster_diameter=2, mean_diameter_coef=2))

    def test_kmeans(self):
        self._test_clusterize('KMeans', dict(max_iteration_count=100, cluster_count=6, init='k++'))

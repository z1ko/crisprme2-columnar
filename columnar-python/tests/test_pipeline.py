import columnar_python as col
import numpy as np
import pytest

# =============================================================================
# Pipeline creation
# =============================================================================

class TestPipelineCreation:

    def test_create_pipeline(self):
        def noop(inp, out):
            pass
        pipeline = col.create_pipeline(2, 128, transform=noop)
        assert pipeline is not None

    def test_submit_and_receive(self):
        def identity(inp, out):
            pass
        pipeline = col.create_pipeline(2, 128, transform=identity)
        pipeline.submit()
        batch = pipeline.receive()
        assert batch is not None


# =============================================================================
# Column views — simple fields
# =============================================================================

class TestSimpleColumns:

    @pytest.fixture()
    def batch(self):
        """Submit and receive one alignment batch."""
        def transform(inp, out):
            ids = np.asarray(inp["id"], copy=False)
            occ = np.asarray(out["occurence"], copy=False)
            occ[:] = ids[:] + 1
        pipeline = col.create_pipeline(2, 64, transform=transform)
        pipeline.submit()
        return pipeline.receive()

    def test_column_returns_memoryview(self, batch):
        view = batch["occurence"]
        assert isinstance(view, col.PyColumnView)

    def test_column_as_numpy(self, batch):
        arr = np.asarray(batch["occurence"], copy=False)
        assert isinstance(arr, np.ndarray)

    def test_column_shape_is_1d(self, batch):
        arr = np.asarray(batch["occurence"], copy=False)
        assert arr.ndim == 1

    def test_column_length(self, batch):
        arr = np.asarray(batch["occurence"], copy=False)
        assert len(arr) == 64

    def test_transform_writes_correct_values(self, batch):
        arr = np.asarray(batch["occurence"], copy=False)
        ids = np.arange(64, dtype=np.uint32)
        np.testing.assert_array_equal(arr, ids + 1)

    def test_unknown_column_raises_key_error(self, batch):
        with pytest.raises(KeyError, match="unknown column"):
            batch["nonexistent"]

    def test_multiple_columns_independent(self, batch):
        occ = np.asarray(batch["occurence"], copy=False)
        strand = np.asarray(batch["strand"], copy=False)
        # strand was not written, should be zero
        assert np.all(strand == 0)
        # occurence was written
        assert occ[0] == 1


# =============================================================================
# Column views — group fields
# =============================================================================

class TestGroupColumns:

    @pytest.fixture()
    def batch(self):
        def transform(inp, out):
            scores = np.asarray(out["score"], copy=False)
            for i in range(scores.shape[0]):
                scores[i][:] = float(i) + 0.5
        pipeline = col.create_pipeline(2, 32, transform=transform)
        pipeline.submit()
        return pipeline.receive()

    def test_group_shape_is_2d(self, batch):
        scores = np.asarray(batch["score"], copy=False)
        assert scores.ndim == 2

    def test_group_shape_dimensions(self, batch):
        scores = np.asarray(batch["score"], copy=False)
        # shape should be (group_size=4, rows=32)
        assert scores.shape == (4, 32)

    def test_group_values_correct(self, batch):
        scores = np.asarray(batch["score"], copy=False)
        for i in range(4):
            expected = float(i) + 0.5
            np.testing.assert_allclose(scores[i], expected)

    def test_group_sub_columns_independent(self, batch):
        scores = np.asarray(batch["score"], copy=False)
        # each sub-column has a different value
        assert not np.array_equal(scores[0], scores[1])


# =============================================================================
# Column views — array fields (row-major)
# =============================================================================

class TestArrayColumns:

    @pytest.fixture()
    def batch(self):
        def transform(inp, out):
            rseq = np.asarray(out["rseq"], copy=False)
            for row in range(rseq.shape[0]):
                rseq[row][0] = row % 256
        pipeline = col.create_pipeline(2, 16, transform=transform)
        pipeline.submit()
        return pipeline.receive()

    def test_array_shape_is_2d(self, batch):
        rseq = np.asarray(batch["rseq"], copy=False)
        assert rseq.ndim == 2

    def test_array_shape_dimensions(self, batch):
        rseq = np.asarray(batch["rseq"], copy=False)
        # shape should be (rows=16, array_len=32)
        assert rseq.shape == (16, 32)

    def test_array_values_written(self, batch):
        rseq = np.asarray(batch["rseq"], copy=False)
        for row in range(16):
            assert rseq[row][0] == row % 256


# =============================================================================
# Zero-copy verification
# =============================================================================

class TestZeroCopy:

    def test_numpy_shares_memory(self):
        def transform(inp, out):
            occ = np.asarray(out["occurence"], copy=False)
            occ[:] = 42
        pipeline = col.create_pipeline(2, 32, transform=transform)
        pipeline.submit()
        batch = pipeline.receive()

        a = np.asarray(batch["occurence"], copy=False)
        b = np.asarray(batch["occurence"], copy=False)
        # both views should point to the same memory
        assert np.shares_memory(a, b)

    def test_dtype_u32(self):
        def transform(inp, out):
            pass
        pipeline = col.create_pipeline(2, 8, transform=transform)
        pipeline.submit()
        batch = pipeline.receive()

        arr = np.asarray(batch["occurence"], copy=False)
        assert arr.dtype == np.uint32

    def test_dtype_f32(self):
        def transform(inp, out):
            pass
        pipeline = col.create_pipeline(2, 8, transform=transform)
        pipeline.submit()
        batch = pipeline.receive()

        arr = np.asarray(batch["score"], copy=False)
        assert arr.dtype == np.float32

    def test_dtype_u8(self):
        def transform(inp, out):
            pass
        pipeline = col.create_pipeline(2, 8, transform=transform)
        pipeline.submit()
        batch = pipeline.receive()

        arr = np.asarray(batch["strand"], copy=False)
        assert arr.dtype == np.uint8

import columnar_python as col
import numpy as np

def transform_batch(input: col.SequenceBatch, output: col.AlignmentBatch):
    ids = np.asarray(input.ids())
    occurences = np.asarray(output.occurences())
    occurences[:] = ids[:] + 100
    pass

pipeline = col.create_pipeline(
    transform = transform_batch,
)
print(pipeline)

pipeline.submit()
a = pipeline.receive()
print(np.asarray(a.occurences()))
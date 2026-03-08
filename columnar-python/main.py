import columnar_python as col
import numpy as np

def transform_batch(input: col.PySequenceBatch, output: col.PyAlignmentBatch):
    print(input)
    print(output)

    ids = np.asarray(input["id"])
    occurences = np.asarray(output["occurence"])
    occurences[:] = ids[:] + 100

    scores = [np.asarray(s) for s in output["score"]]
    for i, score in enumerate(scores):
        score[:] = i + 0.5
        print(np.asarray(score))

pipeline = col.create_pipeline(
    transform = transform_batch,
)
print(pipeline)

pipeline.submit()
a = pipeline.receive()
print(np.asarray(a["occurence"]))
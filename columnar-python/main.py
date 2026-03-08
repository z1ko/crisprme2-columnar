import columnar_python as col
import numpy as np

def transform_batch(input: col.PySequenceBatch, output: col.PyAlignmentBatch):
    print(input)
    print(output)

    # returns a np array of shape (rows,)
    ids = np.asarray(input["id"], copy = False)

    # can modify elements of the array without problems
    occurences = np.asarray(output["occurence"], copy = False)
    occurences[:] = ids[:] + 100

    # 'score' is a group of columns, returns np array of shape (group_size, rows)
    scores = np.asarray(output["score"], copy = False)
    print(scores.shape)

    for i in range(0, scores.shape[0]):
        scores[i][:] = i + 0.5
        print(scores[i])

    # 'rseq' is a single column with an array for row, returns np array of shape (rows, array_len)
    rseq = np.asarray(output["rseq"], copy = False)
    print(rseq.shape)


pipeline = col.create_pipeline(2, 1000000,
    transform = transform_batch,
)
print(pipeline)

pipeline.submit()
a = pipeline.receive()
print(np.asarray(a["occurence"]))
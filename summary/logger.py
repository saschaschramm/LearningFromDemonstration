import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from summary import matplot
import os

tag = "test"

class SummaryWriter:

    def __init__(self, logdir):
        path = os.path.join("logs", logdir)
        self.filewriter = tf.summary.FileWriter(logdir=path, flush_secs=120)

    def write(self, value, step):
        summary = tf.Summary(value=[
            tf.Summary.Value(tag=tag, simple_value=value)
        ])

        self.filewriter.add_summary(summary, global_step=step)
        self.filewriter.flush()

class SummaryReader:

    def __init__(self, logdir):
        self.filename = logdir
        self.path = os.path.join("logs", logdir)

    def read(self):
        file_list = os.listdir(self.path)
        file_list.sort(reverse=True)
        event_acc = EventAccumulator(os.path.join(self.path, file_list[0]))
        event_acc.Reload()
        scalars = event_acc.Scalars(tag=tag)

        x = []
        y = []
        for scalar in scalars:
            x.append(scalar[1])
            y.append(scalar[2])
        return x, y

    def plot(self, left_limit, right_limit, bottom_limit, top_limit, save=False):
        x, y = self.read()

        matplot.plot(x, y,
                     left_limit=left_limit,
                     right_limit=right_limit,
                     bottom_limit=bottom_limit,
                     top_limit=top_limit,
                     save=save,
                     filename=self.filename)
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as pyplot
import os

def plot(x, y, left_limit, right_limit, bottom_limit, top_limit, save, filename):
    pyplot.plot(x, y)
    pyplot.xlabel("Steps")
    pyplot.ylabel("Score")
    pyplot.tight_layout()
    pyplot.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    axes = pyplot.gca()
    axes.set_xlim(left=left_limit, right=right_limit)
    axes.set_ylim(bottom=bottom_limit, top=top_limit)

    if save:
        path = os.path.join(".", "logs", "summary_{}.png".format(filename))
        pyplot.savefig(path)
    else:
        pyplot.show()
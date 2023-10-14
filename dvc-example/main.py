"""Main workflow file."""

import package1

from .evaluate import Evaluate
from .featurization import Featurize
from .prepare import Prepare
from .train import Train

if __name__ == "__main__":
    with package1.Project() as project:
        prepare = Prepare()
        featurize = Featurize(prepared=prepare.outs)
        train = Train(features=featurize.features)
        eval = Evaluate(model=train.model, features=featurize.features)

    project.build()

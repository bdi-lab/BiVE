from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .Loss import Loss
from .MarginLoss import MarginLoss
from .SoftplusLoss_submitted import SoftplusLoss_submitted
from .SoftplusLoss_new import SoftplusLoss_new
from .SigmoidLoss import SigmoidLoss

__all__ = [
    'Loss',
    'MarginLoss',
    'SoftplusLoss_submitted',
    'SoftplusLoss_new',
    'SigmoidLoss',
]
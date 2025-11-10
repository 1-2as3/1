# Copyright (c) OpenMMLab. All rights reserved.
from .backbones import *  # noqa: F401,F403
from .data_preprocessors import *  # noqa: F401,F403
from .dense_heads import *  # noqa: F401,F403
from .detectors import *  # noqa: F401,F403
from .layers import *  # noqa: F401,F403
from .losses import *  # noqa: F401,F403
from .necks import *  # noqa: F401,F403
from .roi_heads import *  # noqa: F401,F403
from .seg_heads import *  # noqa: F401,F403
from .task_modules import *  # noqa: F401,F403
from .test_time_augs import *  # noqa: F401,F403

# Ensure custom macl/dhn/msp modules are imported so their registries execute
from . import macldhnmsp  # noqa: F401
# Ensure custom alignment utilities and heads are imported to register
from .roi_heads.aligned_roi_head import AlignedRoIHead  # noqa: F401
from .utils.domain_aligner import DomainAligner  # noqa: F401

# Safely extend module __all__ if it exists and is a list, otherwise create it.
_existing_all = globals().get('__all__')
if isinstance(_existing_all, list):
	_existing_all.extend(['macldhnmsp', 'AlignedRoIHead', 'DomainAligner'])
	__all__ = _existing_all
else:
	__all__ = ['macldhnmsp', 'AlignedRoIHead', 'DomainAligner']

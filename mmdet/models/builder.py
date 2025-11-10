"""Lightweight builder shim for importing custom registration stubs.

This file intentionally only imports the three registration stubs so
they are discoverable when `mmdet.models.builder` is imported.
"""

from .macldhnmsp.macl_head import MACLHead
from .macldhnmsp.msp_module import MSPReweight
from .macldhnmsp.dhn_sampler import DHNSampler

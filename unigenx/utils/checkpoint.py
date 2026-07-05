# -*- coding: utf-8 -*-
"""Tolerant checkpoint loading.

A released checkpoint stores its hyper-parameters under ``["args"]``, pickled
with the config class that was used when the checkpoint was trained. That class
may belong to a training framework that is not shipped with this package, so a
plain :func:`torch.load` would raise ``ModuleNotFoundError`` while unpickling
``args``.

Only the *attribute values* of ``args`` are ever read (inference rebuilds the
architecture from them via ``arg_utils.from_args``), so any class we cannot
import is replaced by a permissive placeholder that simply carries whatever
attributes the pickle held. Classes that *are* importable -- notably PyTorch's
own tensor-rebuild helpers used for the model weights -- are left untouched, so
the weights load normally.
"""
import pickle

import torch


class _PlaceholderArgs:
    """Permissive stand-in for a pickled object whose original class is not
    importable here. Restores whatever attribute state the pickle carried;
    attributes that were never stored read back as ``None``."""

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, *args, **kwargs):
        if len(args) == 1:
            self._value_ = args[0]

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)

    def __getattr__(self, name):
        return None


class _TolerantUnpickler(pickle.Unpickler):
    """Unpickler that substitutes :class:`_PlaceholderArgs` for any class whose
    module cannot be imported in this environment, while resolving every
    importable class (e.g. ``torch._utils._rebuild_tensor_v2``) normally."""

    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, ImportError, AttributeError):
            return type(name, (_PlaceholderArgs,), {})


class _tolerant_pickle:
    """Minimal ``pickle``-module stand-in for ``torch.load(pickle_module=...)``."""

    Unpickler = _TolerantUnpickler

    @staticmethod
    def load(file, **kwargs):
        return _TolerantUnpickler(file).load()


def load_checkpoint(path, map_location="cpu"):
    """``torch.load`` a checkpoint, tolerating a saved ``args`` object whose
    original class is not part of this package.

    The saved ``args`` is returned as a lightweight attribute holder when its
    class is unavailable; the model weights load exactly as with ``torch.load``.
    """
    return torch.load(
        str(path),
        map_location=map_location,
        weights_only=False,
        pickle_module=_tolerant_pickle,
    )

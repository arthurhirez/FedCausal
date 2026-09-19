"""Reproducible hashing for RNG keys.

Why this module exists
----------------------
``add_measurement_noise`` keyed its per-sensor RNG on ``hash(sensor) % 2**31``.
Python salts ``hash()`` on ``str`` per process (PEP 456), so every rerun of the
same world drew a DIFFERENT noise realisation. The true ``value`` column was
identical and ``observed`` was not, which is the worst shape the bug could
take: worlds are content-addressed, so "same ``sim_hash``" was taken to mean
"same client data", and ``client_datasets`` sits downstream of ``observed``.

The fix is not just "don't use ``hash()`` here" -- it is "there is one function
that turns a label into an RNG key, and everything calls it". Three call sites
had grown three slightly different spellings of the same idea (one masking with
``& 0x7FFFFFFF``, one with ``% 2**31``, one not masking at all, two joining
tuple components with ``|`` and one not). None of those was wrong, but a rule
with three spellings is a rule that drifts back apart.

Not to be confused with
-----------------------
:func:`canonical_hash` (below, re-exported by ``experiments.spec``), which
answers a different question:
IDENTITY of a configuration, over arbitrarily nested dicts, normalised so that
``0`` and ``0.0`` agree, and truncated to a short hex string for use as a cache
key. This module answers "give me a stable integer to seed a generator with".
Keep them separate: widening either one to cover the other's job would mean
changing cache keys to fix an RNG, or vice versa.
"""
from __future__ import annotations

import hashlib
import json
import zlib


def stable_hash(obj) -> int:
    """A reproducible, non-negative hash in ``[0, 2**31)``.

    Uses ``repr(obj)`` rather than ``obj`` itself: for the ``str`` and
    ``tuple``-of-``str`` keys this is used with (sensor names, district-pair
    tuples), ``repr`` is a stable, process-independent string, so the crc32
    of its UTF-8 encoding is stable across processes and across machines --
    exactly the property ``hash()`` does not have. Callers that previously
    wrote ``hash(x) % 2**31`` should call ``stable_hash(x)`` directly; the
    modulus is already applied here.

    Two properties worth stating because callers depend on them:

    * ``repr`` distinguishes types that ``str`` conflates. ``stable_hash(1)``
      and ``stable_hash("1")`` differ, as do ``("A", "B")`` and ``"('A', 'B')"``
      -- so a tuple key does not need a hand-rolled separator, and cannot
      collide with a string that happens to contain one.
    * It is NOT cryptographic and NOT collision-free. It is a seed, not an
      identity; use ``canonical_hash`` when the answer must identify a thing.
    """
    return zlib.crc32(repr(obj).encode()) % 2**31


# --------------------------------------------------------------------------
# identity hashing -- the OTHER job, kept as a separate function on purpose
# --------------------------------------------------------------------------
# Moved here verbatim from ``experiments.spec`` so that ``networks.partitions``
# and ``placement.store`` can content-address their caches without importing
# the experiments layer (which itself imports ``networks``). Same algorithm,
# same output: every existing ``sim_hash`` / ``run_hash`` is unchanged.
def _canon(obj):
    """Normalize to hash-stable primitives: sort-insensitive dicts, lists,
    ints-for-integral-floats (0.0 == 0), numpy scalars -> python."""
    if isinstance(obj, dict):
        return {str(k): _canon(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_canon(v) for v in obj]
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        return int(obj) if obj.is_integer() else obj
    if hasattr(obj, "item"):  # numpy scalar
        return _canon(obj.item())
    return obj


def canonical_hash(obj, n: int = 12) -> str:
    """Short hex identity of an arbitrarily nested configuration."""
    payload = json.dumps(_canon(obj), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:n]


def text_sha(text: str, n: int = 16) -> str:
    """Identity of a text file's CONTENT, newline-normalised.

    Callers read with universal newlines (``Path.read_text`` / a Kedro
    ``TextDataset``), so a CRLF ``.inp`` and its LF copy hash the same, and
    the engine (which reads the file) and a node (which gets it from the
    catalog) always agree.
    """
    return hashlib.sha256(text.replace("\r\n", "\n").encode()).hexdigest()[:n]

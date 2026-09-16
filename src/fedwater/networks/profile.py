"""Network-scoped parameters: the keys whose correct value is a fact about the
network rather than about the scenario.

The problem
-----------
``conf/base/parameters.yml`` is shared by every network, but some of its keys
are only meaningful against one. ``anchor_scale = 0.05`` is right for Graeme
(5557 L/s of base demand against a ~1100 L/s frontier) and absurd for KY7
(67 L/s against ~90). ``income_landuse_mapping`` is POSITIONAL over districts,
so a five-entry list is not merely wrong on a four-district network, it is a
different scenario. ``drift.tgt_district`` and ``drift.seed_node`` name things
that may not exist. Each of these was previously a silent wrong answer after a
network switch.

The rule
--------
``conf/base/parameters.yml`` leaves such a key ``null``. ``null`` means "this
is network-scoped -- fill me from the bundle". The value comes from
``profile.yml``'s ``network_params`` block, which mirrors the parameter tree.

The resolver only ever fills nulls. That single property is what makes it safe
to run in two places at once:

* :func:`fedwater.experiments.spec.resolve_world` applies it to the base
  parameters before it builds a world's override blocks, so the resolved value
  lands in ``conf/local/parameters.yml`` AND in the world's content hash --
  editing a profile therefore invalidates that network's worlds, as it must.
* ``network_prep.resolve_network_parameters`` applies it inside a plain
  ``kedro run``, where there is no engine.

Because filling a null is idempotent and a study that sets the key explicitly
leaves nothing to fill, the two cannot disagree and the order does not matter.
No sentinel, no "was this already applied" flag.

What is NOT here
----------------
``sensors`` and ``drift_seed_nodes`` are also network facts but already had
their own top-level blocks in ``profile.yml``; they are read directly by
``sensing`` and by :func:`seed_node_for`. Only keys that shadow an entry in
``parameters.yml`` go through ``network_params``.
"""
from __future__ import annotations

import copy

import pandas as pd

# (block, key, ...) paths that conf/base/parameters.yml may leave null.
# Deliberately a short, closed list: a profile must not be able to quietly
# rewrite arbitrary parameters, and least of all the `fl` block, which belongs
# to the RUN identity rather than the world's.
NETWORK_SCOPED: tuple[tuple[str, ...], ...] = (
    ("hydraulics", "anchor_scale"),
    ("hydraulics", "expected_headloss"),
    ("scenario", "income_landuse_mapping"),
    ("scenario", "drift", "tgt_district"),
    ("validation", "pressure_band_mca"),
)

# Nulls that resolve from somewhere other than `network_params`, so the
# resolver must not complain about them.
_RESOLVED_ELSEWHERE = (
    ("scenario", "drift", "seed_node"),   # profile.drift_seed_nodes, then auto
    ("validation", "peak_factor_order"),  # optional assertion, null = report only
)


def _get(tree: dict, path: tuple[str, ...]):
    node = tree
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def _set(tree: dict, path: tuple[str, ...], value) -> None:
    node = tree
    for key in path[:-1]:
        node = node.setdefault(key, {})
    node[path[-1]] = value


def network_params(profile: dict) -> dict:
    return copy.deepcopy((profile or {}).get("network_params") or {})


def resolve_params(params: dict, profile: dict) -> tuple[dict, pd.DataFrame]:
    """Fill every null :data:`NETWORK_SCOPED` key from the profile.

    Returns the resolved parameters and a one-row-per-key provenance table, so
    a run's artifacts record which values came from the bundle rather than
    from ``conf/``. Raises if a key is null in both places -- a network whose
    profile forgets ``anchor_scale`` must fail loudly at option-pinning time,
    not produce a scenario at some accidental operating point.
    """
    out = copy.deepcopy(params)
    supplied = network_params(profile)
    name = (profile or {}).get("name", "?")

    rows = []
    for path in NETWORK_SCOPED:
        current = _get(out, path)
        if current is not None:
            rows.append({"key": ".".join(path), "source": "parameters",
                         "value": str(current)})
            continue
        value = _get(supplied, path)
        if value is None:
            raise ValueError(
                f"[network={name}] parameter '{'.'.join(path)}' is null in "
                "conf/base/parameters.yml, which marks it network-scoped, and "
                f"data/01_raw/{name}/profile.yml supplies no value for it "
                "under `network_params`. Set it in the profile (preferred: it "
                "is a fact about the network) or pin it in parameters.")
        _set(out, path, copy.deepcopy(value))
        rows.append({"key": ".".join(path), "source": "profile",
                     "value": str(value)})

    for path in _RESOLVED_ELSEWHERE:
        rows.append({"key": ".".join(path), "source": "deferred",
                     "value": str(_get(out, path))})

    report = pd.DataFrame(rows)
    report.insert(0, "network", name)
    return out, report


def seed_node_for(profile: dict, district: str) -> str | None:
    """The profile's drift seed node for a district, if it declares one.

    A seed node id is network-specific, so this override exists to keep a
    hardcoded id out of ``conf/base/parameters.yml``. ``None`` hands the
    decision to the auto-picker.
    """
    override = ((profile or {}).get("drift_seed_nodes") or {}).get(district)
    return str(override) if override else None


def resolve_seed_node(explicit, profile: dict, district: str, auto) -> str:
    """The one place the drift seed's precedence is written down.

    Order, highest first:

    1. ``scenario.drift.seed_node`` -- an EXPLICIT id. This is what the
       experiments engine writes into ``conf/local/parameters.yml`` for every
       world, so a study always wins.
    2. ``profile.yml``'s ``drift_seed_nodes[district]`` -- the network's own
       declared origin, used by a plain ``kedro run``.
    3. ``auto()`` -- the district's largest-base-demand junction.

    Both callers go through here. ``build_drift_schedule`` passes the
    model-based picker in ``networks.partition``; ``experiments.spec`` passes
    the ``.inp``-text one, which keeps spec expansion free of a wntr load. The
    two pickers agree on every district of every bundle (verified), but the
    PRECEDENCE above must not be duplicated -- that divergence is exactly how
    a plain run and an engine run came to disagree before: the profile
    override was read in ``spec`` only, so ``kedro run`` silently auto-picked
    a different origin node than the study it was supposed to reproduce.
    """
    if explicit not in (None, "", "None"):
        return str(explicit)
    override = seed_node_for(profile, district)
    if override:
        return override
    return str(auto())

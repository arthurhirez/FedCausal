"""Sensor placement with quantified district mixture.

A gauge's mixture -- how much of each district's demand signature reaches it
-- is identified by an EXCITATION STACK: K probe worlds at the target world's
operating point, each converting one district's land use. The stack is the
label source; the placement is chosen from it, never from the target world.

Modules
-------
``excite``          probe-world parameters, horizon plan, in-process run of the
                    project's own simulation pipeline
``store``           content-addressed stacks: cache-or-build, analysis,
                    persistence, reload (``ProbeStack``)
``signal_probe``    candidates, weekly profiles, the resemblance battery
``mixture_probe``   gains, mixture, tiers, elasticity, dependence, response,
                    settling, estimator checks
``exact``           exact, threshold-free quantities of a stack (paired against
                    the spliced no-drift city): labels, E1 composition and its
                    change under drift, E2 storage share, topology, gates;
                    cached per physics under ``<stack>/exact/<hash>/``
``epanet_loop``     EPANET toolkit step loop (mediator/tank state, run
                    integrity) and the mediator-blocked twin (M1)
``classification``  blend shape, channel diversity, report-only arms
``select``          the flow rule on the exact layer (``rule: exact``): branch +
                    transit slots, candidates, labels written per drifting district
``select_legacy``   the pre-exact pure / mixed ladder (``rule: legacy``), kept to
                    reproduce earlier worlds and compare
``readouts``        cross-world battery read-outs (leak, carriers, stability)
``topology``        .inp geometry for the network figures
``figures``         static figures (matplotlib)
``browsers``        interactive composites (ipywidgets)

Ported from the placement POC (``sensoring/``). The POC's ``bridge``,
``worlds``, ``bundles`` and ``landuse_world`` are gone: this package lives
inside ``fedwater`` and imports the pipeline directly, probe worlds run the
pipeline itself, and there is no per-network table of anchor scales.
"""

**A Causal Inference Layer for PyMC**

Ben Vincent

PyMC provides the building blocks for Bayesian causal inference — `pm.do()` for graph surgery, `pm.observe()` for conditioning — but wiring a correct multi-equation generative model by hand is expert-level work. I've been building a package that bridges this gap: a lavaan-inspired formula DSL that compiles structural equations into generative PyMC models, automating topological wiring, mixed likelihood families, correlated residuals, panel structure, and nonlinear transforms with estimable parameters. From a single spec string, practitioners get DAG introspection, MCMC estimation, g-computation with full posterior uncertainty, causal identification diagnostics, and sensitivity analysis. This is not a toy — I'll walk through a range of examples from mediation analysis to full marketing mix models, all powered by PyMC, and share the project ahead of its open-source release.

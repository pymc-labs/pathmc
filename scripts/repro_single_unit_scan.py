#   Copyright 2025 - 2026 The PyMC Labs Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""Ready-to-file upstream reproducer for pathmc #85; no pathmc import needed.

PyMC 6.3.2 / PyTensor 3.3.6: the default invocation fails in Scan.make_node's
outer/inner sequence broadcast check while shared data become constants.
``--declare-shapes`` demonstrates the local boundary workaround; native/JAX
logp and gradients must agree. This is separate from PyMC #8337.
"""

from __future__ import annotations

import argparse

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt


def main() -> None:
    """Reproduce shared-to-constant scan conversion, optionally with shapes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declare-shapes", action="store_true")
    args = parser.parse_args()
    import jax
    from pymc.sampling.jax import get_jaxified_logp

    jax.config.update("jax_enable_x64", True)
    with pm.Model() as model:
        x = pm.Data("x", np.ones((8, 1)))
        initial = pm.Data("initial", np.zeros(1))
        beta = pm.Normal("beta")
        sequence = pt.specify_shape(x, (8, 1)) if args.declare_shapes else x
        carry = pt.specify_shape(initial, (1,)) if args.declare_shapes else initial
        state = pytensor.scan(
            lambda x_t, previous, b: b * previous + x_t,
            sequences=[sequence],
            outputs_info=[carry],
            non_sequences=[beta],
            strict=True,
            return_updates=False,
        )
        pm.Normal("y", state, 1, observed=np.ones((8, 1)))
    point = {"beta": np.array(0.3)}
    native = model.compile_logp()(point)
    print(
        f"PyMC {pm.__version__}, PyTensor {pytensor.__version__}, native logp {native}"
    )
    function = get_jaxified_logp(model)
    np.testing.assert_allclose(function([point["beta"]]), native, atol=1e-10)
    np.testing.assert_allclose(
        jax.grad(function)([point["beta"]])[0],
        model.compile_dlogp()(point)[0],
        atol=1e-10,
    )
    print("JAX logp and gradient match native")


if __name__ == "__main__":
    main()

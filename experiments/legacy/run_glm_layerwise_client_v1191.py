# SPDX-License-Identifier: Apache-2.0
"""Run the existing native replay client with corrected GLM layout flags."""

# Standard
import sys

# Local
from glm_serial_only_v1193 import cancel_periodic
from glm_layerwise_configuration_v1191 import corrective_only_flags
import run_corrective_only_cell_v1144 as client


def main() -> None:
    """Use the unchanged warmup/replay/write-barrier APIs with fixed verification."""
    cancel_periodic(sys.argv[1:])
    client.corrective_only_flags = corrective_only_flags
    client.main()


if __name__ == '__main__':
    main()

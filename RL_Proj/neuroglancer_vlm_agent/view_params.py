from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any


DEFAULT_VIEW_PARAMETERS = {
    "crossSectionScale": 5.033991259,
    "projectionOrientation": [
        -0.4784731864929199,
        0.5569255352020264,
        -0.5086390376091003,
        0.449648380279541,
    ],
    "projectionScale": 13976.00586,
}


def get_view_parameters() -> dict[str, Any]:
    return {
        "crossSectionScale": float(DEFAULT_VIEW_PARAMETERS["crossSectionScale"]),
        "projectionOrientation": [
            float(value) for value in DEFAULT_VIEW_PARAMETERS["projectionOrientation"]
        ],
        "projectionScale": float(DEFAULT_VIEW_PARAMETERS["projectionScale"]),
    }


def apply_view_parameters(state: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    params = get_view_parameters()
    state.update(params)
    if "perspectiveOrientation" in state:
        state["perspectiveOrientation"] = list(params["projectionOrientation"])
    return state

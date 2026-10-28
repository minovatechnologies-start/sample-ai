from __future__ import annotations

import os

APPROVED_MODEL_LICENSES = {
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "CC0-1.0",
    "ISC",
    "MIT",
    "Unlicense",
}


def require_model_license_review() -> str:
    model_id = os.getenv("MODEL_ID", "").strip()
    license_id = os.getenv("MODEL_LICENSE", "").strip()
    reviewed = os.getenv("MODEL_LICENSE_REVIEWED", "").strip().lower() == "true"
    if not model_id:
        raise ValueError("MODEL_ID must name the exact reviewed model.")
    if not license_id or license_id not in APPROVED_MODEL_LICENSES:
        raise ValueError(
            "MODEL_LICENSE must be an approved SPDX ID; review the exact model card and access terms first."
        )
    if not reviewed:
        raise ValueError("Set MODEL_LICENSE_REVIEWED=true only after manually approving the exact model terms.")
    return model_id

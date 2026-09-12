from __future__ import annotations


def constraints_for(feature_names: list, mapping: dict) -> list:
    """Build sklearn monotonic_cst from name prefixes in config."""
    out = []
    for name in feature_names:
        sign = 0
        for key, val in mapping.items():
            if key.lower() in name.lower():
                sign = int(val)
                break
        out.append(sign)
    return out

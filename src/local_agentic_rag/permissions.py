from __future__ import annotations


def is_accessible(
    *,
    access_scope: str,
    access_principals: list[str],
    active_principals: list[str],
    permissions_enabled: bool,
) -> bool:
    if not permissions_enabled:
        return True
    if access_scope == "public":
        return True
    if "*" in access_principals:
        return True
    return bool(set(access_principals).intersection(active_principals))

from local_agentic_rag.permissions import is_accessible


def test_permission_filter_blocks_restricted_chunks() -> None:
    assert not is_accessible(
        access_scope="restricted",
        access_principals=["owners"],
        active_principals=["staff"],
        permissions_enabled=True,
    )
    assert is_accessible(
        access_scope="restricted",
        access_principals=["owners"],
        active_principals=["owners"],
        permissions_enabled=True,
    )

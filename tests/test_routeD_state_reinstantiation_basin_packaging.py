from scripts.package_routeD_state_reinstantiation_basin_gate2_5 import reports_exact


def test_gate2_5_packaging_requires_full_report_equality():
    left = {"schema_version": "x", "rows": [{"value": 1.0}]}
    right = {"schema_version": "x", "rows": [{"value": 1.0}]}
    changed = {"schema_version": "x", "rows": [{"value": 1.000001}]}
    assert reports_exact(left, right)
    assert not reports_exact(left, changed)

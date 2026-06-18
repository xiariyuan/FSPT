# P1 Decision

**日期**: 2026-06-18  
**Decision**: `WEAK_GO`

## Teacher Audit Results

| Teacher | Median re-entry | <4px | <8px |
|---|---:|---:|---:|
| CoTracker3 offline | **3.71px** | **53.8%** | **80.2%** |
| CoTracker3 online | 3.86px | 51.3% | 77.0% |
| Track-On2 | 3.91px | 51.0% | 73.6% |

## Oracle Teacher Selection

| Metric | Fixed Best (CT-offline) | Oracle Selection | Gain |
|---|---:|---:|---:|
| AJ_RD | 0.3870 | 0.4117 | **+0.0247 (+2.5pp)** |
| Median re-entry | 3.71px | 1.70px | -2.01px |

## Go/Stop 判定

| Criteria | 状态 |
|---|---|
| Oracle AJ_RD gain >= 5pp | ⚠️ **+2.5pp** → WEAK_GO |

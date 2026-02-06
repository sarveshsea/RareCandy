# Deploy-Safe P0/P1 Fix Sequence

## P0 (must pass before any canary)
1. Runtime safety and contract enforcement
- Enforce long-only policy in risk (`ENTRY_SHORT` rejected in long-only mode).
- Block invalid entry contracts (missing stop loss / take profit).
- Count daily trade cap only on filled executions, not approvals.
- Keep deployment pause as entries-only block before risk side effects.

2. Data and signal integrity
- Require enough history for trend regime (`trend_slow + buffer` bars).
- Clamp strategy confidence to schema range `[0, 1]`.

3. Calibration + gate determinism
- Real-export calibration only for deployment decisions.
- Predeploy checks require status + manifest contracts and fresh timestamps.
- Predeploy syntax checks are shell-portable (no GNU-only `mapfile` dependency).

## P1 (immediately after P0)
1. Cost model alignment
- Runtime and calibration share the same configurable trading-cost defaults:
  - `TRADING_FEE_RATE`
  - `TRADING_SLIPPAGE_RATE`
  - `TRADING_COST_PER_SIDE`

2. Runtime protection follow-through
- Add lightweight stop/take-profit guard exits for active spot-long positions.
- Emit audit telemetry for guard exits and missing guard state.

3. Deployment pipeline hygiene
- Keep generated artifacts excluded from Git and Docker build context.
- Keep runbooks using in-container calibration and manifest checks.

## Promotion Gates
- `window_trade_count >= 250`
- `ev_ci_low_95 > 0`
- `ece <= 0.03`
- `pause_deployment == false`

## CI checks
- `scripts/run_tests_ci.sh`
- `scripts/predeploy_check.sh`

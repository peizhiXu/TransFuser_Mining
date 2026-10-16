# Current-BEV baseline

Branch: `current-bev-baseline`. This is a separately trained baseline.

RGB + LiDAR → current BEV → anchor/status queries and cross-attention →
256 candidate features → offset head + score head → score argmax → decoded
trajectory (`anchor + offset`) → existing HD465 controller.

No world model, future residual reader, AdaLN, reward head, or current-agent
auxiliary head is instantiated. No semantic-map, reward, or Reward Top-K loss
is used. Dataset loading requires only recorded expert future poses, not
future scene targets, dense-route assets, or candidate reward caches.

The two retained objectives are unchanged from the existing implementation:
oracle-matched L1 offset regression (weight 1.0) and soft candidate classification
(weight 0.1). Classification targets are `softmax(-anchor_expert_distance)` over
flattened 8×3 poses. Inference chooses `argmax(scores)` without expert labels.

Use `wote_mining/WoTE_train.py --planner-mode current_only` with the existing
prepared `split_v2` root and the usual training arguments. The branch defaults
to this mode. Use a new output directory and train from scratch; full-method
checkpoints cannot be resumed into this architecture. Keep `args.json` beside
the checkpoint when using `wote_mining/WoTE_evaluate.sh`. The closed-loop agent
restores the mode from `args.json`; world-model pass settings do not affect the
baseline, whose telemetry reports zero world passes and `score_head` selection.

For existing full-method checkpoints, missing planner-mode metadata means
`future_reward`. Full training can be explicitly selected with
`--planner-mode future_reward`.

This baseline versus the full method measures the combined benefit of future
modeling, modulation, reward evaluation and related supervision. It does not
isolate the mining boundary-margin term.

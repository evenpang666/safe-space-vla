# PI05 Latent Safety Decoder Design

## Goal

Build a trainable safety module that reads PI05 VLM prefix latent tokens and predicts the robot's future link-point coordinates after the current action chunk would execute. Collision safety is decided only by deterministic geometry: predicted link points are tested against obstacle OBBs or occupied voxels. The model does not include a collision-classification head.

## Scope

This design covers the first implementation slice:

- Extract or load PI05 VLM `prefix_tokens`.
- Train a decoder that maps those tokens to fixed-topology future robot link points.
- Generate supervision from real VLA action chunks by running FK and sampling the same fixed link-point topology.
- Reuse the existing OBB / occupied-grid collision utilities for safety decisions.

It does not cover online intervention policy, action replanning, or end-to-end PI05 fine-tuning.

## Data Flow

```text
PI05 observation
  -> PI05 VLM prefix encoder
  -> prefix_tokens [B, N, D]
  -> SafetyLatentEncoder
  -> LinkPointDecoder
  -> pred_link_points [B, T, L, P, 3]
  -> flatten to [B, T*L*P, 3]
  -> geometric collision check against obstacle OBBs / occupied grid
  -> collision or safe
```

Where:

- `T` is the future horizon in FK samples, not necessarily the raw action-chunk length.
- `L` is the number of robot link/body groups used by the skeleton.
- `P` is the number of sampled points per link at each future time.
- Every output point has a stable semantic index `(time_idx, link_idx, point_idx)`.

## Training Targets

For each training sample, the VLA's real action chunk is treated as ground truth motion intent:

```text
start robot state + real action_chunk
  -> integrate joint path
  -> FK
  -> fixed-topology link skeleton samples
  -> target_link_points [T, L, P, 3]
```

The first version should use the same MuJoCo geom-envelope skeleton convention already implemented in `scripts/libero_joint_swept_pointcloud.py`: capsules / cylinders use center axes, boxes / ellipsoids use their longest center axis, and meshes use the compiled mesh AABB longest axis. This keeps the learned target aligned with the current visualization and collision pipeline.

The saved dataset record should contain at least:

- `prefix_tokens`: PI05 VLM prefix latent tokens, shape `[N, D]`.
- `target_link_points`: FK target points, shape `[T, L, P, 3]`.
- `action_chunk`: real VLA action chunk used to create the target.
- `start_state`: robot state used before the chunk.
- `obstacle_box_centers`, `obstacle_box_axes`, `obstacle_box_half_sizes`: optional OBB geometry for offline evaluation.
- Metadata: task id/name, init-state id, skeleton source, link names, horizon, points per link.

## Model Interface

The safety module has one required input and one required output:

```python
pred_link_points = model(prefix_tokens)
```

Expected tensor shapes:

```text
prefix_tokens:     [B, N, D]
pred_link_points: [B, T, L, P, 3]
```

Recommended first architecture:

- `SafetyLatentEncoder`: masked token pooling or a small transformer over `prefix_tokens`.
- `LinkPointDecoder`: MLP or transformer decoder that emits fixed `T*L*P*3` coordinates.
- Coordinates are in world frame meters, matching the OBB and point-cloud coordinate system.

The model should not output a safety label. It only predicts geometry.

## Loss

Use a coordinate reconstruction loss:

```text
loss = Huber(pred_link_points, target_link_points)
```

Optional later additions are geometric regularizers, not classification:

- Temporal smoothness between adjacent `T` steps.
- Link-axis consistency so points on the same link stay ordered.
- Larger penalty for points near obstacles if OBBs are present during training.

## Safety Decision

At inference:

```text
pred_points = pred_link_points.reshape(-1, 3)
collision_result = detect_swept_obstacle_collision(pred_points, safe_space, collision_margin)
```

Decision rule:

- If any predicted point overlaps an obstacle OBB / occupied voxel, output `collision`.
- Otherwise output `safe`.

The first implementation should reuse the existing functions:

- `points_inside_oriented_boxes`
- `occupied_grid_collision_mask`
- `detect_swept_obstacle_collision`

This keeps the safety signal deterministic and explainable.

## Evaluation

Evaluate two levels separately:

- Geometry prediction: mean / median / 95th percentile point error in meters, plus per-link errors.
- Safety decision: compare geometry-derived collision decisions from predicted points against geometry-derived collision decisions from FK target points.

False negatives are the critical failure mode: cases where FK target points collide but predicted points are judged safe. Report false-negative rate with the chosen collision margin.

## Implementation Slices

1. Dataset exporter:
   - Takes PI05 observations and real VLA action chunks.
   - Saves `prefix_tokens` and FK-generated `target_link_points`.

2. Model and training script:
   - Loads dataset records.
   - Trains `prefix_tokens -> target_link_points` with Huber loss.
   - Saves checkpoint and normalization metadata.

3. Inference script:
   - Loads PI05 prefix tokens, checkpoint, and safespace OBB data.
   - Predicts future link points.
   - Calls the existing geometric collision checker.
   - Saves predicted points and a collision/safe result.

4. Visualization:
   - Reuse the existing frontview projection / video path for predicted link points.
   - Add title or metadata showing geometry-derived `collision` / `safe`.

## Open Assumptions

- PI05 prefix token extraction can be added without changing PI05 action generation behavior.
- The first version targets LIBERO / Franka geometry and coordinate frames.
- Fixed-topology link points are sufficient for conservative safety checking when combined with a positive collision margin.

"""Classical computer-vision building detector.

Scores each segmentation region on independent, individually inspectable
pieces of evidence, then merges neighbouring accepted regions - large roofs
usually break into several facets at watershed level and belong to one
structure.

Assumptions, stated plainly because they bound where this detector works:
  * near-nadir view, so a roof projects roughly onto its own footprint;
  * roofs are brighter and less chromatic than vegetation and bare ground;
  * a single dominant illumination direction across the image;
  * buildings are wider than the roads and paths around them.

Where those do not hold, expect misses and false positives - hence the
confidence values and the manual-correction workflow.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..logging_setup import get_logger
from ..model import BuildingDetection
from .base import BuildingDetector, DetectionContext, DetectorResult
from .registry import register_detector

log = get_logger(__name__)


def _ramp(value: float, low: float, high: float) -> float:
    """Linear 0..1 ramp; saturates outside [low, high]."""
    if high <= low:
        return 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


class ClassicalDetector(BuildingDetector):
    name = "classical"

    def detect(self, context: DetectionContext) -> DetectorResult:
        from ..stages import regionfeatures

        params = self.params
        weights = dict(params.get("weights", {}))
        min_area = int(params.get("min_region_area_px", 400))
        min_thickness = float(params.get("min_thickness_px", 7.0))
        accept = float(params.get("accept_score", 0.50))

        features = regionfeatures.compute(
            context.segmentation, context.cues, context.shadow_adjacency, min_area
        )
        log.info("scoring %d candidate regions", len(features))

        # A weak shadow estimate should not dominate the score, but neither
        # should it be switched off entirely: attenuate rather than gate.
        raw_trust = float(getattr(context.shadow_direction, "confidence", 0.0))
        shadow_trust = 0.45 + 0.55 * raw_trust
        bright_anchor = context.cues.bright_reference
        bright_high = context.cues.bright_high

        scored: dict[int, tuple[float, dict]] = {}
        for feature in features:
            cue_values = {
                "brightness": _ramp(feature.mean_lightness, bright_anchor, bright_high),
                "rectangularity": _ramp(feature.rectangularity, 0.30, 0.75),
                "solidity": _ramp(feature.solidity, 0.40, 0.88),
                "shadow_adjacency": _ramp(feature.shadow_adjacency, 0.05, 0.35) * shadow_trust,
                "texture": _ramp(feature.mean_texture, 8.0, 32.0),
                "thickness": _ramp(feature.thickness_px, min_thickness, min_thickness * 3.5),
            }
            # Weighted *mean* of the positive cues: the score stays comparable to
            # the accept threshold no matter how the weights are retuned.
            positive_weight = sum(weights.get(k, 0.0) for k in cue_values)
            support = (
                sum(weights.get(k, 0.0) * v for k, v in cue_values.items()) / positive_weight
                if positive_weight > 0 else 0.0
            )

            # Long thin ribbons are roads, kerbs and tree lines, not footprints.
            # Real buildings can be elongated, so the penalty starts late and
            # ramps rather than cutting off.
            elongation_penalty = weights.get("elongation_penalty", 0.45) * _ramp(
                feature.elongation,
                float(params.get("elongation_free_ratio", 4.5)),
                float(params.get("elongation_max_ratio", 11.0)),
            )
            penalty = (
                elongation_penalty
                + weights.get("vegetation_penalty", 0.85) * _ramp(feature.vegetation_fraction, 0.12, 0.45)
                + weights.get("shadow_penalty", 0.90) * _ramp(feature.shadow_fraction, 0.25, 0.60)
                + weights.get("soil_penalty", 0.55) * _ramp(feature.soil_fraction, 0.15, 0.50)
            )
            score = support - penalty

            # A region narrower than the paths around it is not a footprint.
            if feature.thickness_px < min_thickness:
                score -= 0.35

            evidence = {k: round(v, 3) for k, v in cue_values.items()}
            evidence.update({
                "support": round(support, 3),
                "penalty": round(penalty, 3),
                "elongation": round(feature.elongation, 2),
                "elongationPenalty": round(elongation_penalty, 3),
                "vegetationFraction": round(feature.vegetation_fraction, 3),
                "shadowFraction": round(feature.shadow_fraction, 3),
                "soilFraction": round(feature.soil_fraction, 3),
                "thicknessPx": round(feature.thickness_px, 2),
                "score": round(score, 4),
            })
            scored[feature.label] = (float(np.clip(score, 0.0, 1.0)), evidence)

        by_label = {f.label: f for f in features}
        accepted = {lab for lab, (score, _) in scored.items() if score >= accept}
        log.info("%d of %d regions accepted at threshold %.2f", len(accepted), len(scored), accept)

        instance_mask, groups = self._merge_regions(
            context.segmentation.labels, accepted, by_label, params
        )

        detections = self._contours_to_detections(instance_mask, groups, scored, by_label)
        log.info("classical detector produced %d building candidates", len(detections))

        score_layer = np.zeros(context.segmentation.labels.shape, np.float32)
        for label, (score, _) in scored.items():
            score_layer[context.segmentation.labels == label] = score

        return DetectorResult(
            detections=detections,
            debug_layers={
                "region_score": score_layer,
                "accepted_mask": (instance_mask > 0).astype(np.float32),
            },
            notes=[
                f"shadow-direction confidence {raw_trust:.2f}"
                + ("" if raw_trust >= 0.3 else " (low - shadow cue down-weighted)")
            ],
        )

    # -- merging ---------------------------------------------------------

    def _merge_regions(self, labels, accepted, by_label, params):
        """Join touching accepted regions of similar lightness into one instance."""
        tolerance = float(params.get("merge_lightness_tolerance", 26.0))
        dilate_px = int(params.get("merge_dilate_px", 3)) | 1

        mask = np.isin(labels, list(accepted)).astype(np.uint8) if accepted else np.zeros(labels.shape, np.uint8)
        if not accepted:
            return np.zeros(labels.shape, np.int32), {}

        # Watershed leaves a -1 ridge between neighbours; closing bridges it so
        # facets of one roof become a single connected component.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
        bridged = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        count, components = cv2.connectedComponents(bridged, 8)
        groups: dict[int, list[int]] = {}
        for label in accepted:
            region = labels == label
            if not region.any():
                continue
            component = int(np.bincount(components[region].ravel()).argmax())
            if component == 0:
                continue
            groups.setdefault(component, []).append(label)

        # Split a component back apart when its members disagree strongly on
        # lightness - that usually means two different surfaces got bridged.
        instance_mask = np.zeros(labels.shape, np.int32)
        next_id = 1
        final_groups: dict[int, list[int]] = {}
        for members in groups.values():
            lightnesses = np.array([by_label[m].mean_lightness for m in members])
            reference = float(np.median(lightnesses))
            keep = [m for m in members if abs(by_label[m].mean_lightness - reference) <= tolerance]
            outliers = [m for m in members if m not in keep]
            for bucket in ([keep] + [[o] for o in outliers]):
                if not bucket:
                    continue
                instance_mask[np.isin(labels, bucket)] = next_id
                final_groups[next_id] = bucket
                next_id += 1
        return instance_mask, final_groups

    # -- contour extraction ----------------------------------------------

    def _contours_to_detections(self, instance_mask, groups, scored, by_label):
        detections: list[BuildingDetection] = []
        for instance_id, members in sorted(groups.items()):
            mask = (instance_mask == instance_id).astype(np.uint8)
            # Close the watershed ridge so the outline is a single contour.
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1,
            )
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            area = float(cv2.contourArea(contour))
            if area < 1.0:
                continue

            areas = np.array([by_label[m].area_px for m in members], dtype=np.float64)
            scores = np.array([scored[m][0] for m in members], dtype=np.float64)
            confidence = float((scores * areas).sum() / max(areas.sum(), 1.0))

            (_, _), (rw, rh), angle = cv2.minAreaRect(contour)
            orientation = (-angle if rw >= rh else -(angle + 90.0)) % 180.0
            x, y, w, h = cv2.boundingRect(contour)

            # Keep the strongest member's evidence; it explains the acceptance.
            lead = max(members, key=lambda m: scored[m][0])
            evidence = dict(scored[lead][1])
            evidence["mergedRegions"] = len(members)

            detections.append(
                BuildingDetection(
                    id=f"D{instance_id:03d}",
                    contour=[(float(p[0][0]), float(p[0][1])) for p in contour],
                    bounding_box=(int(x), int(y), int(w), int(h)),
                    confidence=round(confidence, 4),
                    orientation_deg=round(float(orientation), 3),
                    area_px=area,
                    detector=self.name,
                    evidence=evidence,
                )
            )
        return detections


register_detector("classical", ClassicalDetector)

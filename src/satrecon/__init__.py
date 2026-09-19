"""satrecon - approximate satellite/aerial image to 3D scene pipeline.

Every quantity this package produces from imagery alone is an *estimate*.
Nothing here measures a real structure; it measures pixels, and converts
pixels to metres only when the user supplies a scale.
"""

__version__ = "0.1.0"

# Bumped whenever a change alters numerical output for identical input.
# Recorded in every scene file so runs can be compared across versions.
MODEL_VERSION = "phase2-2026.09"

PHASE = 2

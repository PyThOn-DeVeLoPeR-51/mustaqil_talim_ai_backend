"""Backward-compatible facade for the modular etalon evaluator.

New code should import ``evaluate_etalon`` from ``app.ai.etalon.pipeline``.
Existing helper imports from this module remain supported.
"""

from app.ai.etalon.config import *
from app.ai.etalon.io import *
from app.ai.etalon.frame import *
from app.ai.etalon.placement import *
from app.ai.etalon.line_types import *
from app.ai.etalon.dimensions import *
from app.ai.etalon.projections import *
from app.ai.etalon.projection_sections import *
from app.ai.etalon.visible_view import *
from app.ai.etalon.visible_section import *
from app.ai.etalon.cleanliness import *
from app.ai.etalon.reporting import *
from app.ai.etalon.pipeline import *

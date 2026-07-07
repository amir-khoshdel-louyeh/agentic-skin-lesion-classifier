"""Verdict envelope schema (proposal.tmp Section 7).

A verdict counts ONLY with a valid execution receipt. `ran=False`
verdicts are recorded as "no evidence".
"""

from typing import Literal

from pydantic import BaseModel, Field, StrictBool, StrictFloat, StrictInt, StrictStr


HAM10000_CLASSES: tuple[str, ...] = (
    "Actinic keratoses",
    "Basal cell carcinoma",
    "Benign keratosis",
    "Dermatofibroma",
    "Melanoma",
    "Melanocytic nevi",
    "Vascular lesions",
)

UncertaintyFlag = Literal["borderline", "no_evidence", "tool_failed", "disagreement"]


class VerdictEnvelope(BaseModel, frozen=True):
    ran: StrictBool
    command: StrictStr = Field(min_length=1)
    exit_code: StrictInt | None = None
    predicted_class: StrictStr | None = None
    confidence: StrictFloat | None = Field(default=None, ge=0.0, le=1.0)
    reasoning: StrictStr = ""
    uncertainty_flags: list[UncertaintyFlag] = Field(default_factory=list)

    def has_receipt(self) -> bool:
        return self.ran and self.exit_code == 0

    def is_countable(self) -> bool:
        return (
            self.has_receipt()
            and self.predicted_class in HAM10000_CLASSES
            and self.confidence is not None
        )

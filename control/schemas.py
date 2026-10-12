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
    # Every tool line the session actually ran (plan.tmp §2.3). `command`
    # repeats the first one for backward compatibility; `commands` exists
    # because a session may run 1..n tools and the referee must receipt
    # ALL of them — otherwise a second, disagreeing model disappears.
    commands: list[StrictStr] = Field(default_factory=list)
    exit_code: StrictInt | None = None
    predicted_class: StrictStr | None = None
    confidence: StrictFloat | None = Field(default=None, ge=0.0, le=1.0)
    reasoning: StrictStr = ""
    # Control-side flags only: copied verbatim from the re-executed tool
    # JSON. These are the ONLY flags the referee may count.
    uncertainty_flags: list[UncertaintyFlag] = Field(default_factory=list)
    # Agent-declared flags, kept as an advisory reliability signal (display
    # only, never a vote). Keeping them in one field is what stopped an
    # invented "borderline" from flipping the referee.
    agent_flags: list[StrictStr] = Field(default_factory=list)

    def command_list(self) -> list[str]:
        """Distinct, non-empty commands this session claims to have run."""
        ordered: list[str] = []
        for cmd in [self.command, *(self.commands or [])]:
            text = (cmd or "").strip()
            if text and text not in ordered:
                ordered.append(text)
        return ordered

    def has_receipt(self) -> bool:
        return self.ran and self.exit_code == 0

    def is_countable(self) -> bool:
        return (
            self.has_receipt()
            and self.predicted_class in HAM10000_CLASSES
            and self.confidence is not None
        )

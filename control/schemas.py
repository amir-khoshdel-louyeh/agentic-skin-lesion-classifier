"""Verdict envelope schema (proposal.tmp Section 7).

A verdict counts ONLY with a valid execution receipt. `ran=False`
verdicts are recorded as "no evidence".
"""

from typing import Literal

from pydantic import (BaseModel, ConfigDict, Field, StrictBool, StrictFloat,
                      StrictInt, StrictStr, model_validator)


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


class ToolResult(BaseModel):
    """Strict, non-coercing schema for JSON emitted by local tools.

    Unknown keys are preserved (tools have different evidence fields), but
    the shared fields are validated without string/boolean coercion. A
    successful diagnostic result must carry a known class and confidence;
    malformed numbers never reach the referee as evidence.
    """

    model_config = ConfigDict(extra="allow", frozen=True)

    status: Literal["success", "error", "warning"]
    message: StrictStr | None = None
    tool: StrictStr | None = None
    disease_name: StrictStr | None = None
    confidence_score: StrictFloat | None = Field(default=None, ge=0.0, le=1.0)
    predicted_class_index: StrictInt | None = Field(default=None, ge=0, le=6)
    uncertainty_flags: list[StrictStr] = Field(default_factory=list)
    quality_pass: StrictBool | None = None
    metrics: dict[str, StrictInt | StrictFloat] | None = None
    risk_band: StrictStr | None = None
    abcde_score: StrictFloat | None = None
    hair_pixels_inpainted: StrictInt | None = Field(default=None, ge=0)
    quality_pass_before: StrictBool | None = None
    quality_pass_after: StrictBool | None = None

    @model_validator(mode="after")
    def validate_evidence(self):
        if self.status == "error" and not (self.message or "").strip():
            raise ValueError("error tool result requires a message")
        if ((self.disease_name is None)
                != (self.confidence_score is None)):
            raise ValueError(
                "disease_name and confidence_score must appear together"
            )
        if (self.disease_name is not None
                and self.disease_name not in HAM10000_CLASSES):
            raise ValueError(f"unknown disease_name: {self.disease_name}")
        return self


class VerdictEnvelope(BaseModel, frozen=True):
    ran: StrictBool
    command: StrictStr = Field(min_length=1)
    # Every tool line the session actually ran (plan.tmp §2.3). `command`
    # repeats the first one for backward compatibility; `commands` exists
    # because a session may run 1..n tools and the referee must receipt
    # ALL of them — otherwise a second, disagreeing model disappears.
    commands: list[StrictStr] = Field(default_factory=list)
    exit_code: StrictInt | None = None
    # Content-addressed receipt ledger. These are written only after a
    # control-side execution; agent-supplied envelopes cannot set them.
    receipt_id: StrictStr | None = Field(
        default=None, min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$"
    )
    command_sha256: StrictStr | None = Field(
        default=None, min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$"
    )
    image_sha256: StrictStr | None = Field(
        default=None, min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$"
    )
    stdout_sha256: StrictStr | None = Field(
        default=None, min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$"
    )
    tool: StrictStr | None = Field(default=None, min_length=1)
    duration_ms: StrictInt | None = Field(default=None, ge=0)
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

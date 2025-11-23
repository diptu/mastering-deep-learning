from enum import Enum

from pydantic import BaseModel, confloat, conint, validator


class GenderEnum(str, Enum):
    male = "Male"
    female = "Female"


class GeographyEnum(str, Enum):
    france = "France"
    germany = "Germany"
    spain = "Spain"


class ChurnRequest(BaseModel):
    Age: confloat(ge=0)  # Age cannot be negative
    Balance: confloat(ge=0)  # Balance cannot be negative
    IsActiveMember: conint(ge=0, le=1)  # Must be 0 or 1
    Geography: GeographyEnum
    Gender: GenderEnum

    # Optional extra check (redundant, for clarity)
    @validator("IsActiveMember")
    def check_isactive(cls, v):
        if v not in (0, 1):
            raise ValueError("IsActiveMember must be 0 or 1")
        return v

from enum import Enum

class ExportErrorType(Enum):
    INVALID_INPUT_TYPE = 1
    INVALID_OUTPUT_TYPE = 2
    VIOLATION_OF_SPEC = 3
    NOT_SUPPORTED = 4
    MISSING_PROPERTY = 5
    UNINITIALIZED = 6

def internal_assert(pred: bool, assert_msg: str) -> None:
    if not pred:
        raise InternalError(assert_msg)

class InternalError(Exception):
    pass

class ExportError(Exception):
    def __init__(self, error_code: ExportErrorType, message: str) -> None:
        super().__init__(f"[{error_code}]: {message}")

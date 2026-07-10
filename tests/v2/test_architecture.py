import ast
from pathlib import Path

from src.fishing_v2.domain.action_intent import ActionIntent
from src.fishing_v2.domain.observations import PromptObservationKind
from src.fishing_v2.domain.runtime_state import RuntimeState


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORE_ROOTS = (
    PROJECT_ROOT / "src" / "fishing_v2" / "domain",
    PROJECT_ROOT / "src" / "fishing_v2" / "fusion",
    PROJECT_ROOT / "src" / "fishing_v2" / "runtime",
)
FORBIDDEN = ("src.detectors", "src.state_detector", "src.state_smoother", "src.ml")


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    result: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            result.append(node.module)
    return result


def test_v2_core_has_no_legacy_or_ml_imports() -> None:
    for root in CORE_ROOTS:
        for path in root.glob("*.py"):
            assert not any(name.startswith(FORBIDDEN) for name in _imports(path)), path


def test_only_legacy_adapters_import_specialized_detectors() -> None:
    offenders = []
    root = PROJECT_ROOT / "src" / "fishing_v2"
    for path in root.rglob("*.py"):
        if any(name.startswith("src.detectors") for name in _imports(path)):
            if "legacy_adapters" not in path.parts:
                offenders.append(path)
    assert offenders == []


def test_domain_enums_are_distinct_types() -> None:
    assert PromptObservationKind.READY_PROMPT != RuntimeState.READY
    assert RuntimeState.READY != ActionIntent.START_HOOK
    assert type(PromptObservationKind.READY_PROMPT) is not type(RuntimeState.READY)


def test_prompt_unknown_does_not_equal_idle() -> None:
    assert PromptObservationKind.UNKNOWN.value != RuntimeState.IDLE.value


def test_other_prompt_is_not_no_prompt() -> None:
    assert PromptObservationKind.OTHER_PROMPT is not PromptObservationKind.NO_PROMPT


def test_prompt_classifier_v1_is_not_imported_by_v2() -> None:
    root = PROJECT_ROOT / "src" / "fishing_v2"
    imported = [name for path in root.rglob("*.py") for name in _imports(path)]
    assert not any(name.startswith("src.ml.prompt") for name in imported)

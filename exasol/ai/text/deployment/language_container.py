import os
import re
from contextlib import contextmanager

from exasol.python_extension_common.deployment.language_container_builder import (
    LanguageContainerBuilder,
    exclude_cuda,
    find_path_backwards,
)

from exasol.ai.text.impl.utils.constants import DEFAULT_SPACY_PIPELINE

CONTAINER_NAME = "exasol_text_ai_extension_container"


_INSTALL_REQUIREMENTS_PATTERN = re.compile(
    r"^\s*(?i:run)\s+python(?P<version>\d+\.\d+)\s+-m\s+pip\s+install\s+.*"
    r"-r\s+/project/requirements\.txt",
    flags=re.MULTILINE,
)


def _flavor_base_python_version(dockerfile: str) -> str:
    """Determines the Python version installed in the flavor base image."""
    match = _INSTALL_REQUIREMENTS_PATTERN.search(dockerfile)
    if match is None:
        raise RuntimeError(
            "Could not determine the flavor base image's Python version from "
            "its dependencies Dockerfile"
        )
    return match.group("version")


def _add_pytorch_to_requirements(container_builder: LanguageContainerBuilder) -> None:
    """Modifies the default dependencies/Dockerfile"""
    dockerfile_file = "flavor_base/dependencies/Dockerfile"
    dockerfile = container_builder.read_file(dockerfile_file)
    install_extra = "--extra-index-url https://download.pytorch.org/whl/cpu"
    dockerfile = _INSTALL_REQUIREMENTS_PATTERN.sub(
        rf"\g<0> {install_extra}", dockerfile
    )
    container_builder.write_file(dockerfile_file, dockerfile)


def _add_spacy_model_download(container_builder: LanguageContainerBuilder) -> None:
    """Modifies the default dependencies/Dockerfile"""
    dockerfile_file = "flavor_base/dependencies/Dockerfile"
    dockerfile = container_builder.read_file(dockerfile_file)
    python_version = _flavor_base_python_version(dockerfile)
    dockerfile = "\n".join(
        [
            dockerfile,
            f"RUN python{python_version} -m spacy download {DEFAULT_SPACY_PIPELINE} "
            "--break-system-packages",
            "",
        ]
    )
    container_builder.write_file(dockerfile_file, dockerfile)


@contextmanager
def language_container_factory():
    with LanguageContainerBuilder(CONTAINER_NAME) as container_builder:
        _add_spacy_model_download(container_builder)
        _add_pytorch_to_requirements(container_builder)
        project_directory = find_path_backwards("pyproject.toml", __file__).parent
        container_builder.prepare_flavor(
            project_directory, requirement_filter=exclude_cuda
        )
        yield container_builder


@contextmanager
def export_slc(export_dir: str | None = None):
    if export_dir and (not os.path.isdir(export_dir)):
        os.makedirs(export_dir)
    with language_container_factory() as container_builder:
        export_result = container_builder.export(export_dir)
        export_info = export_result.export_infos[str(container_builder.flavor_path)][
            "release"
        ]
        yield export_info.cache_file


if __name__ == "__main__":
    with export_slc():
        pass

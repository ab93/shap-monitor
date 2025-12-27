"""Generate the code reference pages and navigation.

This script automatically generates API reference pages from the source code
using mkdocstrings. It creates a markdown file for each Python module and
builds the navigation structure.
"""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

# Root directory of the package
root = Path(__file__).parent.parent
src = root / "shapmonitor"

# Iterate through all Python files in the package
for path in sorted(src.rglob("*.py")):
    # Get module path relative to package root
    module_path = path.relative_to(src).with_suffix("")
    doc_path = path.relative_to(src).with_suffix(".md")
    full_doc_path = Path("api", doc_path)

    # Get the parts of the module path
    parts = tuple(module_path.parts)

    # Skip __init__.py files at the package level (but process subpackage __init__.py)
    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1].startswith("_") and parts[-1] != "__init__":
        # Skip private modules (but keep __init__)
        continue

    # Skip if no parts left (root __init__.py)
    if not parts:
        continue

    # Add to navigation
    nav[parts] = doc_path.as_posix()

    # Create the markdown file with mkdocstrings directive
    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        # Build the Python module identifier
        ident = ".".join(["shapmonitor"] + list(parts))

        # Write the page header and mkdocstrings directive
        fd.write(f"# {' '.join(parts).replace('_', ' ').title()}\n\n")
        fd.write(f"::: {ident}\n")

    # Set edit path for the generated file
    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))

# Write the navigation structure
with mkdocs_gen_files.open("api/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())

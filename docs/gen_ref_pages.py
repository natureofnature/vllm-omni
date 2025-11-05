"""Generate reference pages for API documentation."""
from pathlib import Path
import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

# Define the modules to document (excluding __init__.py and private modules)
for path in sorted(Path("vllm_omni").rglob("*.py")):
    # Skip __init__.py files and private modules
    if path.name == "__init__.py" or path.name.startswith("_"):
        continue
    
    # Get module path relative to vllm_omni
    module_path = path.relative_to("vllm_omni").with_suffix("")
    doc_path = module_path.with_suffix(".md")
    full_doc_path = Path("api", doc_path)

    # Build navigation structure
    parts = tuple(module_path.parts)
    nav[parts] = doc_path.as_posix()
    
    # Generate module name
    module_name = "vllm_omni." + ".".join(parts)
    
    # Write the markdown file with mkdocstrings directive
    with mkdocs_gen_files.open(full_doc_path, "w") as f:
        f.write(f"# {module_name}\n\n")
        f.write(f"::: {module_name}\n")
        f.write("    options:\n")
        f.write("      show_source: true\n")
        f.write("      show_root_heading: true\n")
        f.write("      show_root_toc_entry: true\n")
        f.write("      show_root_full_path: false\n")
        f.write("      show_object_full_path: false\n")
        f.write("      heading_level: 2\n")
        f.write("      show_if_no_docstring: true\n")
        f.write("      filters: [\"!^_\"]\n")
        f.write("      members_order: alphabetical\n")

# Write navigation summary for literate-nav plugin
# This will be automatically included in the navigation
with mkdocs_gen_files.open("api/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())

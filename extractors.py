"""
Extractors: file discovery, project detection, and structural skeleton extraction.

Skeletons strip implementation details and keep only the structural
elements that reveal patterns — class declarations, annotations, method
signatures, imports, field types, etc.  This compresses files ~5-10x
so many more fit in a single LLM context window.

Skeletons also include explicit ordering and naming metadata to help
the LLM identify conventions around member ordering, naming patterns,
and code organization.
"""

import re
from collections import Counter
from fnmatch import fnmatch
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Category definitions
# ---------------------------------------------------------------------------

CATEGORIES = {
    "java-models": {
        "label": "Java Model",
        "description": "Java model/entity classes",
        "globs": ["**/model/*.java", "**/models/*.java"],
        # Only include files that look like real model classes
        "content_patterns": [r"\bclass\s+\w+"],
    },
    "java-daos": {
        "label": "Java DAO",
        "description": "Java Data Access Objects",
        "globs": ["**/dao/*DAO.java", "**/daos/*DAO.java"],
        # Only include files that contain class definitions
        "content_patterns": [r"\bclass\s+\w+"],
    },
    "java-actionbeans": {
        "label": "Java ActionBean",
        "description": "Java ActionBean controllers",
        "globs": ["**/action/*Action.java", "**/actions/*Action.java"],
        # Only include files implementing ActionBean
        "content_patterns": [r"\bActionBean\b"],
    },
    "frontend": {
        "label": "Front-End (HTML/CSS/JS/Vue)",
        "description": "Front-end templates, styles, scripts, and Vue components",
        "globs": ["**/*.css", "**/*.js", "**/*.mjs", "**/*.vm"],
        # Exclude common noise — critical for large codebases where globs
        # like **/*.js can match thousands of vendor/generated files.
        "exclude_patterns": [
            # Package managers and build output
            "node_modules", "bower_components", ".npm",
            "dist/", "/build/", "/out/", "/target/",
            # Minified / bundled files
            ".min.", ".bundle.", ".chunk.", ".packed.",
            # Common vendor libraries
            "vendor", "third-party", "third_party", "thirdparty",
            "jquery", "bootstrap", "popper", "lodash", "moment",
            "angular", "react.production", "vue.global",
            "d3.v", "chart.js", "highcharts",
            "tinymce", "ckeditor", "codemirror", "ace-builds",
            "select2", "datatables", "fullcalendar",
            # Source maps and generated
            ".map", "__generated__", ".generated.",
            # Test / fixture data
            "__tests__", "__mocks__", "test/fixtures", "spec/fixtures",
            # Python / Java build artifacts
            "__pycache__", ".class",
            # IDE and config
            ".idea/", ".vscode/", ".settings/",
        ],
        # Skip files larger than this (bytes) — likely generated/vendored
        "max_file_size": 200_000,
    },
}


# ---------------------------------------------------------------------------
# Project detection
# ---------------------------------------------------------------------------

def discover_projects(root_dir: str) -> List[str]:
    """
    Detect projects: immediate subdirectories of root_dir.
    Returns sorted list of project names.
    """
    root = Path(root_dir).resolve()
    projects = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and not child.name.startswith("."):
            projects.append(child.name)
    return projects


def get_file_project(filepath: str, root_dir: str) -> str:
    """
    Determine which project a file belongs to.
    The project is the first path component relative to root_dir.
    Returns the project name, or "_root" if the file is directly in root_dir.
    """
    root = Path(root_dir).resolve()
    try:
        rel = Path(filepath).resolve().relative_to(root)
        parts = rel.parts
        if len(parts) > 1:
            return parts[0]
        return "_root"
    except ValueError:
        return "_unknown"


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def discover_files(root_dir: str) -> Dict[str, List[str]]:
    """Walk root_dir and return {category: [filepaths]} for all categories."""
    root = Path(root_dir).resolve()
    result = {}

    for cat_key, cat_def in CATEGORIES.items():
        matched = set()
        exclude = cat_def.get("exclude_patterns", [])
        max_size = cat_def.get("max_file_size", 0)  # 0 = no limit

        for pattern in cat_def["globs"]:
            for path in root.rglob(_glob_tail(pattern)):
                rel = str(path.relative_to(root))

                # Check the directory component of the glob
                if not _matches_dir_pattern(rel, pattern):
                    continue

                # Exclusion filter
                path_str = str(path)
                if any(ex in path_str for ex in exclude):
                    continue

                # File size filter (skip vendor bundles, generated files, etc.)
                if max_size and path.is_file():
                    try:
                        if path.stat().st_size > max_size:
                            continue
                    except OSError:
                        continue

                matched.add(path_str)

        # Content regex filter: if content_patterns is defined, at least one
        # pattern must match somewhere in the file contents for inclusion.
        content_patterns = cat_def.get("content_patterns")
        if content_patterns:
            compiled = [re.compile(p) for p in content_patterns]
            filtered = set()
            for fpath in matched:
                try:
                    text = Path(fpath).read_text(errors="ignore")
                except OSError:
                    continue
                if any(rx.search(text) for rx in compiled):
                    filtered.add(fpath)
            matched = filtered

        result[cat_key] = sorted(matched)

    return result


def _glob_tail(pattern: str) -> str:
    """Extract the filename glob from a full pattern like '**/dao/*DAO.java'."""
    return pattern.split("/")[-1]


def _matches_dir_pattern(rel_path: str, pattern: str) -> bool:
    """Check if the relative path matches the directory components of the pattern."""
    parts = pattern.replace("\\", "/").split("/")
    if parts[0] == "**":
        # Any directory structure — just check the non-** components
        dir_parts = parts[1:-1]  # e.g., ["dao"] from "**/dao/*DAO.java"
        if not dir_parts:
            return True
        path_parts = rel_path.replace("\\", "/").split("/")
        # Check that the directory component appears somewhere in the path
        for pp in path_parts[:-1]:  # exclude filename
            if fnmatch(pp, dir_parts[0]):
                return True
        return False
    return True


# ---------------------------------------------------------------------------
# Skeleton extraction
# ---------------------------------------------------------------------------

def extract_skeleton(filepath: str, category: str) -> str:
    """Extract a structural skeleton from a file based on its category."""
    try:
        source = Path(filepath).read_text(errors="replace")
    except Exception as e:
        return f"[Error reading file: {e}]"

    if category.startswith("java"):
        return _extract_java_skeleton(source, filepath)
    elif category == "frontend":
        ext = Path(filepath).suffix.lower()
        if ext == ".css":
            return _extract_css_skeleton(source, filepath)
        elif ext in (".js", ".mjs"):
            return _extract_js_skeleton(source, filepath)
        elif ext == ".vm":
            return _extract_vm_skeleton(source, filepath)
        else:
            return _extract_generic_skeleton(source, filepath)
    else:
        return _extract_generic_skeleton(source, filepath)


# ---------------------------------------------------------------------------
# Java skeleton
# ---------------------------------------------------------------------------

def _extract_java_skeleton(source: str, filepath: str) -> str:
    """
    Extract package, imports, class structure, fields, and method signatures.
    Also produces ordering and naming metadata to help identify conventions.
    Additionally captures formatting patterns: line breaks between members,
    one-line method bodies, getter/setter proximity, and brace placement.
    """
    lines = source.split("\n")
    sections = {
        "file": Path(filepath).name,
        "loc": len(lines),
        "package": "",
        "imports": [],
        "class_annotations": [],
        "class_declaration": "",
        "implements_extends": "",
        "fields": [],       # list of (raw_text, classification_dict)
        "methods": [],      # list of (raw_text, classification_dict)
        "inner_classes": [],
        "constants": [],    # static final fields (subset of fields)
    }

    # Formatting tracking
    formatting = {
        "brace_same_line": 0,       # opening { on same line as declaration
        "brace_next_line": 0,       # opening { on its own line
        "one_line_methods": 0,      # methods whose body is a single line (incl. braces)
        "multi_line_methods": 0,    # methods with multi-line bodies
        "one_line_getters": 0,      # single-line getter methods
        "one_line_setters": 0,      # single-line setter methods
        "member_line_numbers": [],  # (line_number, member_type, kind) for proximity analysis
        "blank_lines_between": [],  # count of blank lines between consecutive members
        "getter_setter_pairs": 0,   # getter immediately followed by setter (or vice versa)
        "class_brace_same_line": False,  # class opening brace style
    }

    # --- Single-pass extraction ---
    i = 0
    brace_depth = 0
    in_class_body = False
    current_annotations = []
    method_body_depth = 0
    in_method_body = False
    method_start_line = 0
    last_member_line = -1  # line number of last field or method declaration

    while i < len(lines):
        line = lines[i].strip()

        # Package
        if line.startswith("package "):
            sections["package"] = line.rstrip(";")

        # Imports
        elif line.startswith("import "):
            sections["imports"].append(line.rstrip(";"))

        # Annotations
        elif line.startswith("@") and not in_method_body:
            current_annotations.append(line)

        # Class / interface / enum declaration
        elif not in_class_body and _is_class_declaration(line):
            sections["class_annotations"] = current_annotations[:]
            sections["class_declaration"] = line.split("{")[0].strip()
            current_annotations = []
            in_class_body = True
            brace_depth = line.count("{") - line.count("}")
            # Track class brace style
            formatting["class_brace_same_line"] = "{" in line

        # Inside class body
        elif in_class_body and not in_method_body:
            brace_depth += line.count("{") - line.count("}")

            # Field declaration
            if _is_field_declaration(line):
                # Track blank lines between consecutive members
                if last_member_line >= 0:
                    blank_count = _count_blank_lines_between(lines, last_member_line, i)
                    formatting["blank_lines_between"].append(blank_count)
                annotation_start = i - len(current_annotations)
                last_member_line = i

                field_entry = ""
                if current_annotations:
                    field_entry = " ".join(current_annotations) + " "
                field_entry += _clean_field(line)
                # Classify the field
                fclass = _classify_field(line, current_annotations)
                sections["fields"].append((field_entry, fclass))
                if fclass["is_constant"]:
                    sections["constants"].append(field_entry)
                formatting["member_line_numbers"].append((i, "field", fclass["kind"]))
                current_annotations = []

            # Method declaration
            elif _is_method_declaration(line):
                # Track blank lines between consecutive members
                if last_member_line >= 0:
                    annotation_start = i - len(current_annotations)
                    blank_count = _count_blank_lines_between(lines, last_member_line, annotation_start if current_annotations else i)
                    formatting["blank_lines_between"].append(blank_count)

                sig = ""
                if current_annotations:
                    sig = "\n".join(current_annotations) + "\n"
                sig += _clean_method_sig(line)
                # Classify the method
                mclass = _classify_method(line, current_annotations)
                sections["methods"].append((sig, mclass))
                formatting["member_line_numbers"].append((i, "method", mclass["kind"]))
                current_annotations = []
                method_start_line = i

                # Track brace placement style for methods
                if "{" in line:
                    formatting["brace_same_line"] += 1
                    net = line.count("{") - line.count("}")
                    # Check for one-liner: opening and closing brace on same line
                    if net == 0 and "}" in line:
                        # Entire method on one line: e.g. public String getName() { return name; }
                        mclass["is_one_liner"] = True
                        formatting["one_line_methods"] += 1
                        if mclass["kind"] == "GETTER":
                            formatting["one_line_getters"] += 1
                        elif mclass["kind"] == "SETTER":
                            formatting["one_line_setters"] += 1
                        last_member_line = i
                    elif net > 0:
                        in_method_body = True
                        method_body_depth = net
                        mclass["is_one_liner"] = False
                else:
                    # Opening brace might be on the next line
                    # Look ahead for it
                    if i + 1 < len(lines) and lines[i + 1].strip() == "{":
                        formatting["brace_next_line"] += 1
                    mclass["is_one_liner"] = False

            # Inner class
            elif _is_class_declaration(line):
                inner = ""
                if current_annotations:
                    inner = " ".join(current_annotations) + " "
                inner += line.split("{")[0].strip()
                sections["inner_classes"].append(inner)
                current_annotations = []

            else:
                # Non-matching line — reset annotations if it's not blank
                if line and not line.startswith("//") and not line.startswith("*"):
                    current_annotations = []

        # Skip method bodies — but track body length
        elif in_method_body:
            brace_depth += line.count("{") - line.count("}")
            method_body_depth += line.count("{") - line.count("}")
            if method_body_depth <= 0:
                in_method_body = False
                method_body_depth = 0
                body_lines = i - method_start_line
                last_member_line = i
                # Classify as one-liner if the body is just 1 statement line
                # (method_start has {, one statement, then } on line i)
                if body_lines <= 2:
                    formatting["one_line_methods"] += 1
                    last_method = sections["methods"][-1][1] if sections["methods"] else None
                    if last_method:
                        last_method["is_one_liner"] = True
                        if last_method["kind"] == "GETTER":
                            formatting["one_line_getters"] += 1
                        elif last_method["kind"] == "SETTER":
                            formatting["one_line_setters"] += 1
                else:
                    formatting["multi_line_methods"] += 1

        i += 1

    # Detect getter/setter proximity pairs
    formatting["getter_setter_pairs"] = _count_getter_setter_pairs(
        formatting["member_line_numbers"]
    )

    # --- Format output with ordering & naming metadata ---
    out = []
    out.append(f"// File: {sections['file']}  ({sections['loc']} lines)")
    if sections["package"]:
        out.append(sections["package"])
    out.append("")

    # Group imports by top-level package
    if sections["imports"]:
        import_groups = _group_imports(sections["imports"])
        for group in import_groups:
            out.append(group)
        out.append("")

    if sections["class_annotations"]:
        out.extend(sections["class_annotations"])
    if sections["class_declaration"]:
        out.append(sections["class_declaration"] + " {")
    out.append("")

    # --- Member ordering summary ---
    if sections["fields"] or sections["methods"]:
        order_summary = _build_ordering_summary(sections["fields"], sections["methods"])
        out.append(f"  // [MEMBER ORDER: {order_summary}]")
        out.append("")

    # --- Formatting & style summary ---
    fmt_summary = _build_formatting_summary(formatting, sections)
    if fmt_summary:
        out.append(f"  // [FORMATTING: {fmt_summary}]")
        out.append("")

    # --- Fields with classification ---
    if sections["fields"]:
        out.append("  // --- Fields ---")
        prev_kind = None
        for text, fclass in sections["fields"]:
            kind = fclass["kind"]
            if kind != prev_kind and prev_kind is not None:
                out.append(f"  // [{kind}]")
            out.append(f"  {text}")
            prev_kind = kind
        out.append("")

    # --- Methods with classification ---
    if sections["methods"]:
        out.append("  // --- Methods ---")
        prev_kind = None
        for text, mclass in sections["methods"]:
            kind = mclass["kind"]
            if kind != prev_kind and prev_kind is not None:
                out.append(f"  // [{kind}]")
            one_liner_tag = " [ONE-LINE]" if mclass.get("is_one_liner") else ""
            for mline in text.split("\n"):
                out.append(f"  {mline}")
            if one_liner_tag:
                out.append(f"  // ^{one_liner_tag}")
            out.append("")
        # Method naming summary
        method_names = [mc["name"] for _, mc in sections["methods"] if mc.get("name")]
        if method_names:
            naming = _analyze_method_naming(method_names)
            if naming:
                out.append(f"  // [METHOD NAMING: {naming}]")
                out.append("")

    if sections["inner_classes"]:
        out.append("  // --- Inner Classes ---")
        for ic in sections["inner_classes"]:
            out.append(f"  {ic}")
        out.append("")

    # --- Naming pattern summary ---
    naming_summary = _build_naming_summary(sections)
    if naming_summary:
        out.append(f"  // [NAMING: {naming_summary}]")
        out.append("")

    out.append("}")
    return "\n".join(out)


def _classify_field(line: str, annotations: List[str]) -> dict:
    """Classify a field by visibility, static/final, and type."""
    is_static = "static " in line
    is_final = "final " in line
    is_constant = is_static and is_final
    is_private = "private " in line
    is_public = "public " in line
    is_protected = "protected " in line

    # Extract field name
    name = ""
    # Try to get the name: last word before = or ;
    m = re.search(r'\b(\w+)\s*[=;]', line)
    if m:
        name = m.group(1)

    if is_constant:
        kind = "CONSTANT"
    elif is_static:
        kind = "STATIC"
    elif is_public:
        kind = "PUBLIC_FIELD"
    elif is_protected:
        kind = "PROTECTED_FIELD"
    else:
        kind = "PRIVATE_FIELD"

    # Detect naming style
    naming_style = _detect_naming_style(name)

    return {
        "kind": kind,
        "name": name,
        "is_constant": is_constant,
        "is_static": is_static,
        "naming_style": naming_style,
        "has_annotations": len(annotations) > 0,
    }


def _classify_method(line: str, annotations: List[str]) -> dict:
    """Classify a method by type (constructor, getter, setter, lifecycle, etc.)."""
    is_static = "static " in line
    is_override = any("@Override" in a for a in annotations)
    is_public = "public " in line
    is_private = "private " in line
    is_protected = "protected " in line

    # Extract method name
    name = ""
    m = re.search(r'\b(\w+)\s*\(', line)
    if m:
        name = m.group(1)

    # Classify by name pattern
    if name and name[0].isupper() and "class" not in line.lower():
        kind = "CONSTRUCTOR"
    elif name.startswith("get") or name.startswith("is") or name.startswith("has"):
        kind = "GETTER"
    elif name.startswith("set"):
        kind = "SETTER"
    elif name.startswith("find") or name.startswith("search") or name.startswith("lookup"):
        kind = "FINDER"
    elif name.startswith("create") or name.startswith("build") or name.startswith("make"):
        kind = "FACTORY"
    elif name.startswith("save") or name.startswith("persist") or name.startswith("store"):
        kind = "PERSISTENCE"
    elif name.startswith("delete") or name.startswith("remove"):
        kind = "DELETION"
    elif name.startswith("update") or name.startswith("modify"):
        kind = "UPDATE"
    elif name.startswith("validate") or name.startswith("check"):
        kind = "VALIDATION"
    elif name.startswith("to") and len(name) > 2 and name[2].isupper():
        kind = "CONVERSION"
    elif name in ("equals", "hashCode", "toString", "compareTo", "clone"):
        kind = "OBJECT_METHOD"
    elif is_override:
        kind = "OVERRIDE"
    elif is_static:
        kind = "STATIC_METHOD"
    elif is_private:
        kind = "PRIVATE_HELPER"
    else:
        kind = "PUBLIC_METHOD"

    return {
        "kind": kind,
        "name": name,
        "is_static": is_static,
        "is_override": is_override,
        "naming_style": _detect_naming_style(name),
        "visibility": "public" if is_public else "protected" if is_protected else "private",
    }


def _detect_naming_style(name: str) -> str:
    """Detect camelCase, PascalCase, SCREAMING_SNAKE_CASE, snake_case, etc."""
    if not name:
        return "unknown"
    if name.isupper() and "_" in name:
        return "SCREAMING_SNAKE_CASE"
    if name.isupper():
        return "UPPER"
    if "_" in name:
        if name[0].isupper():
            return "Pascal_Snake"
        return "snake_case"
    if name[0].isupper():
        return "PascalCase"
    if any(c.isupper() for c in name[1:]):
        return "camelCase"
    return "lowercase"


def _dedup_consecutive(items: List[str]) -> List[str]:
    """Remove consecutive duplicates: [A, A, B, A] -> [A, B, A]."""
    result = []
    for item in items:
        if not result or result[-1] != item:
            result.append(item)
    return result


def _build_ordering_summary(
    fields: List[Tuple[str, dict]],
    methods: List[Tuple[str, dict]],
) -> str:
    """Produce a compact summary of member ordering in the class."""
    order_parts = []

    if fields:
        kinds = _dedup_consecutive([fc["kind"] for _, fc in fields])
        order_parts.append("Fields: " + " -> ".join(kinds))

    if methods:
        kinds = _dedup_consecutive([mc["kind"] for _, mc in methods])
        order_parts.append("Methods: " + " -> ".join(kinds))

    return " | ".join(order_parts)


def _analyze_method_naming(names: List[str]) -> str:
    """Analyze method naming patterns and produce a summary."""
    prefixes = Counter()
    for name in names:
        # Extract verb prefix
        m = re.match(r'^(get|set|is|has|find|create|delete|remove|update|save|validate|check|build|make|to|on|handle|do|init|load|process|convert)\w*', name)
        if m:
            prefixes[m.group(1)] += 1

    if not prefixes:
        return ""

    parts = [f"{prefix}*({count})" for prefix, count in prefixes.most_common()]
    return ", ".join(parts)


def _build_naming_summary(sections: dict) -> str:
    """Summarize naming conventions observed in this file."""
    parts = []

    # Class name
    class_decl = sections["class_declaration"]
    if class_decl:
        m = re.search(r'\b(class|interface|enum)\s+(\w+)', class_decl)
        if m:
            cls_name = m.group(2)
            # Detect suffix pattern
            for suffix in ["DAO", "Action", "Bean", "Service", "Controller",
                          "Repository", "Factory", "Builder", "Handler",
                          "Listener", "Adapter", "Impl", "Helper", "Utils",
                          "Manager", "Provider", "Processor", "Config"]:
                if cls_name.endswith(suffix):
                    parts.append(f"class suffix={suffix}")
                    break

    # Field naming styles
    if sections["fields"]:
        field_styles = Counter(fc["naming_style"] for _, fc in sections["fields"])
        dominant = field_styles.most_common(1)[0]
        parts.append(f"fields={dominant[0]}({dominant[1]})")

    # Constant naming styles
    if sections["constants"]:
        parts.append(f"constants={len(sections['constants'])}")

    return ", ".join(parts)


def _count_blank_lines_between(lines: list, from_line: int, to_line: int) -> int:
    """Count consecutive blank lines between two member declarations."""
    blank = 0
    for idx in range(from_line + 1, min(to_line, len(lines))):
        stripped = lines[idx].strip()
        if stripped == "":
            blank += 1
        elif stripped.startswith("//") or stripped.startswith("*") or stripped.startswith("/*"):
            # Comments don't break blank-line counting
            continue
        else:
            break
    return blank


def _count_getter_setter_pairs(member_line_numbers: list) -> int:
    """Count how many times a GETTER is immediately followed by a SETTER (or vice versa)."""
    pairs = 0
    method_entries = [(ln, kind) for ln, mtype, kind in member_line_numbers if mtype == "method"]
    for idx in range(len(method_entries) - 1):
        cur_kind = method_entries[idx][1]
        next_kind = method_entries[idx + 1][1]
        if (cur_kind == "GETTER" and next_kind == "SETTER") or \
           (cur_kind == "SETTER" and next_kind == "GETTER"):
            pairs += 1
    return pairs


def _build_formatting_summary(formatting: dict, sections: dict) -> str:
    """Produce a compact summary of code formatting patterns observed."""
    parts = []

    # Brace placement style
    same = formatting["brace_same_line"]
    nxt = formatting["brace_next_line"]
    if same + nxt > 0:
        if nxt == 0:
            parts.append("braces=same-line")
        elif same == 0:
            parts.append("braces=next-line")
        else:
            parts.append(f"braces=same-line({same})/next-line({nxt})")

    # Class brace style
    if formatting["class_brace_same_line"]:
        parts.append("class-brace=same-line")
    elif sections["class_declaration"]:
        parts.append("class-brace=next-line")

    # One-line methods
    one = formatting["one_line_methods"]
    multi = formatting["multi_line_methods"]
    if one + multi > 0:
        if one > 0:
            details = []
            if formatting["one_line_getters"]:
                details.append(f"getters={formatting['one_line_getters']}")
            if formatting["one_line_setters"]:
                details.append(f"setters={formatting['one_line_setters']}")
            other = one - formatting["one_line_getters"] - formatting["one_line_setters"]
            if other > 0:
                details.append(f"other={other}")
            parts.append(f"one-line-methods={one}({', '.join(details)})")
        if multi > 0:
            parts.append(f"multi-line-methods={multi}")

    # Getter/setter pair proximity
    if formatting["getter_setter_pairs"] > 0:
        parts.append(f"getter-setter-pairs={formatting['getter_setter_pairs']}")

    # Blank-line spacing patterns between members
    blanks = formatting["blank_lines_between"]
    if blanks:
        from collections import Counter as _Counter
        blank_counts = _Counter(blanks)
        dominant = blank_counts.most_common(1)[0]
        if len(blank_counts) == 1:
            parts.append(f"member-spacing={dominant[0]}-blank-lines")
        else:
            spacing_desc = ", ".join(
                f"{count}bl({freq})" for count, freq in blank_counts.most_common()
            )
            parts.append(f"member-spacing=[{spacing_desc}]")

    return ", ".join(parts)


def _is_class_declaration(line: str) -> bool:
    return bool(
        re.match(
            r"^(public|private|protected|abstract|static|final|\s)*(class|interface|enum)\s",
            line,
        )
    )


def _is_field_declaration(line: str) -> bool:
    """Heuristic: type + name + semicolon, not a method."""
    if "(" in line or line.startswith("//") or line.startswith("*"):
        return False
    return bool(
        re.match(
            r"^(public|private|protected|static|final|transient|volatile|\s)*"
            r"[A-Z][\w<>,\s\[\]?]*\s+\w+\s*(=.*)?;",
            line,
        )
    )


def _is_method_declaration(line: str) -> bool:
    return bool(
        re.match(
            r"^(public|private|protected|static|final|abstract|synchronized|native|\s)*"
            r"(<[\w<>,\s?]+>\s+)?"  # optional generics
            r"[\w<>,\[\]\s?]+\s+"  # return type
            r"\w+\s*\(",  # method name + paren
            line,
        )
    )


def _clean_field(line: str) -> str:
    """Strip initializers from field declarations."""
    # Keep everything up to '=' or ';'
    match = re.match(r"^(.*?)\s*=", line)
    if match:
        return match.group(1).strip() + ";"
    return line.strip()


def _clean_method_sig(line: str) -> str:
    """Extract just the method signature, stripping the body."""
    # Find the closing paren, possibly followed by throws
    match = re.match(r"^(.*?\))\s*(throws\s+[\w,\s]+)?\s*\{?", line)
    if match:
        sig = match.group(1).strip()
        if match.group(2):
            sig += " " + match.group(2).strip()
        return sig + ";"
    return line.split("{")[0].strip() + ";"


def _group_imports(imports: List[str]) -> List[str]:
    """Group imports by top-level package, show count if many."""
    groups = {}  # type: Dict[str, List[str]]
    for imp in imports:
        clean = imp.replace("import ", "").replace("static ", "").strip()
        top = clean.split(".")[0] if "." in clean else clean
        groups.setdefault(top, []).append(imp)

    result = []
    for top, imps in sorted(groups.items()):
        if len(imps) <= 3:
            result.extend(imps)
        else:
            result.append(f"import {top}.* ({len(imps)} imports)")
    return result


# ---------------------------------------------------------------------------
# CSS skeleton
# ---------------------------------------------------------------------------

def _extract_css_skeleton(source: str, filepath: str) -> str:
    lines = source.split("\n")
    out = [f"/* File: {Path(filepath).name}  ({len(lines)} lines) */"]

    # Extract selectors, custom properties, media queries
    selectors = []
    custom_props = []
    media_queries = []

    # Formatting tracking for CSS
    brace_same_line = 0   # selector { on same line
    brace_next_line = 0   # { on its own line after selector
    one_line_rules = 0    # .foo { color: red; } on one line
    blank_between_rules = []

    last_rule_end = -1

    for idx, line in enumerate(lines):
        stripped = line.strip()

        # Custom properties
        if stripped.startswith("--"):
            custom_props.append(stripped.split(":")[0].strip())

        # Media queries
        if stripped.startswith("@media"):
            media_queries.append(stripped.rstrip("{").strip())

        # Selectors (lines containing '{')
        if "{" in stripped and not stripped.startswith("@") and not stripped.startswith("/*") and not stripped.startswith("--"):
            sel = stripped.split("{")[0].strip()
            if sel and not sel.startswith("*"):
                selectors.append(sel)

            # Track brace style and one-line rules
            if sel:
                brace_same_line += 1
                if "}" in stripped:
                    one_line_rules += 1

                # Track blank lines between rule blocks
                if last_rule_end >= 0:
                    bl = 0
                    for bi in range(last_rule_end + 1, idx):
                        if lines[bi].strip() == "":
                            bl += 1
                        else:
                            break
                    blank_between_rules.append(bl)

        if stripped == "}" or stripped.endswith("}"):
            last_rule_end = idx

        # Check for next-line brace: selector on one line, { alone on next
        if stripped and not stripped.startswith("/*") and not stripped.startswith("//"):
            if "{" not in stripped and "}" not in stripped and ";" not in stripped:
                if idx + 1 < len(lines) and lines[idx + 1].strip() == "{":
                    brace_next_line += 1

    # Formatting summary
    fmt_parts = []
    if brace_same_line + brace_next_line > 0:
        if brace_next_line == 0:
            fmt_parts.append("braces=same-line")
        elif brace_same_line == 0:
            fmt_parts.append("braces=next-line")
        else:
            fmt_parts.append(f"braces=same-line({brace_same_line})/next-line({brace_next_line})")
    if one_line_rules > 0:
        fmt_parts.append(f"one-line-rules={one_line_rules}")
    if blank_between_rules:
        from collections import Counter as _Ctr
        bc = _Ctr(blank_between_rules)
        dominant = bc.most_common(1)[0]
        if len(bc) == 1:
            fmt_parts.append(f"rule-spacing={dominant[0]}-blank-lines")
        else:
            spacing = ", ".join(f"{c}bl({f})" for c, f in bc.most_common())
            fmt_parts.append(f"rule-spacing=[{spacing}]")
    if fmt_parts:
        out.append(f"\n/* [FORMATTING: {', '.join(fmt_parts)}] */")

    if custom_props:
        out.append("\n/* Custom Properties */")
        for cp in sorted(set(custom_props))[:30]:
            out.append(f"  {cp}")

    if media_queries:
        out.append("\n/* Media Queries */")
        for mq in sorted(set(media_queries)):
            out.append(f"  {mq}")

    # Show selector patterns (deduplicated, grouped by type)
    if selectors:
        class_sels = sorted(set(s for s in selectors if s.startswith(".")))
        id_sels = sorted(set(s for s in selectors if s.startswith("#")))
        elem_sels = sorted(set(s for s in selectors if not s.startswith(".") and not s.startswith("#")))

        out.append(f"\n/* Selectors: {len(selectors)} total, {len(set(selectors))} unique */")
        if class_sels:
            out.append(f"/* Class selectors ({len(class_sels)}): */")
            for s in class_sels[:40]:
                out.append(f"  {s}")
            if len(class_sels) > 40:
                out.append(f"  ... and {len(class_sels) - 40} more")

        if id_sels:
            out.append(f"/* ID selectors ({len(id_sels)}): */")
            for s in id_sels[:20]:
                out.append(f"  {s}")

        if elem_sels:
            out.append(f"/* Element selectors ({len(elem_sels)}): */")
            for s in elem_sels[:20]:
                out.append(f"  {s}")

    return "\n".join(out)


# ---------------------------------------------------------------------------
# JavaScript / MJS skeleton
# ---------------------------------------------------------------------------

def _extract_js_skeleton(source: str, filepath: str) -> str:
    lines = source.split("\n")
    out = [f"// File: {Path(filepath).name}  ({len(lines)} lines)"]

    imports = []
    exports = []
    functions = []
    classes = []
    vue_component = False

    # Formatting tracking for JS
    brace_same_line = 0
    brace_next_line = 0
    one_line_functions = 0

    for idx, line in enumerate(lines):
        stripped = line.strip()

        if stripped.startswith("import ") or stripped.startswith("from "):
            imports.append(stripped)
        elif stripped.startswith("export "):
            if "function" in stripped or "class" in stripped or "const" in stripped:
                exports.append(stripped.split("{")[0].split("=>")[0].strip())
            elif stripped.startswith("export default"):
                exports.append(stripped.split("{")[0].strip())

        # Vue detection
        if "Vue.component" in stripped or "defineComponent" in stripped or "createApp" in stripped:
            vue_component = True

        # Standalone function declarations
        func_match = re.match(r"^(async\s+)?function\s+\w+", stripped)
        if func_match:
            exports.append(stripped.split("{")[0].strip())
            # Track brace style
            if "{" in stripped:
                brace_same_line += 1
                if stripped.endswith("}") or stripped.endswith("};"):
                    one_line_functions += 1
            elif idx + 1 < len(lines) and lines[idx + 1].strip() == "{":
                brace_next_line += 1

        # Class declarations
        if re.match(r"^(export\s+)?(default\s+)?class\s+", stripped):
            classes.append(stripped.split("{")[0].strip())
            if "{" in stripped:
                brace_same_line += 1
            elif idx + 1 < len(lines) and lines[idx + 1].strip() == "{":
                brace_next_line += 1

        # Arrow functions assigned to const/let
        match = re.match(r"^(export\s+)?(const|let|var)\s+(\w+)\s*=\s*(async\s+)?\(", stripped)
        if match:
            functions.append(f"{match.group(2)} {match.group(3)} = {'async ' if match.group(4) else ''}(...)")

        # Object methods (for Vue options API, etc.)
        match = re.match(r"^(async\s+)?(\w+)\s*\(.*\)\s*\{", stripped)
        if match and not stripped.startswith("if") and not stripped.startswith("for") and not stripped.startswith("while"):
            pass  # these are tricky to distinguish from control flow, skip

    if vue_component:
        out.append("// [Vue Component detected]")

    # Formatting summary
    fmt_parts = []
    if brace_same_line + brace_next_line > 0:
        if brace_next_line == 0:
            fmt_parts.append("braces=same-line")
        elif brace_same_line == 0:
            fmt_parts.append("braces=next-line")
        else:
            fmt_parts.append(f"braces=same-line({brace_same_line})/next-line({brace_next_line})")
    if one_line_functions > 0:
        fmt_parts.append(f"one-line-functions={one_line_functions}")
    if fmt_parts:
        out.append(f"// [FORMATTING: {', '.join(fmt_parts)}]")

    if imports:
        out.append("\n// --- Imports ---")
        for imp in imports[:30]:
            out.append(imp)
        if len(imports) > 30:
            out.append(f"// ... and {len(imports) - 30} more imports")

    if classes:
        out.append("\n// --- Classes ---")
        for c in classes:
            out.append(c)

    if exports:
        out.append("\n// --- Exports / Functions ---")
        for e in sorted(set(exports)):
            out.append(e)

    if functions:
        out.append("\n// --- Named Arrow Functions ---")
        for f in sorted(set(functions)):
            out.append(f)

    return "\n".join(out)


# ---------------------------------------------------------------------------
# Velocity template (.vm) skeleton
# ---------------------------------------------------------------------------

def _extract_vm_skeleton(source: str, filepath: str) -> str:
    lines = source.split("\n")
    out = [f"## File: {Path(filepath).name}  ({len(lines)} lines)"]

    macros = []
    variables = set()
    includes = []
    html_structure = []
    css_classes = set()

    for line in lines:
        stripped = line.strip()

        # Velocity macros
        if stripped.startswith("#macro"):
            macros.append(stripped)

        # Velocity includes/parses
        if "#include" in stripped or "#parse" in stripped:
            includes.append(stripped)

        # Variable references: $foo, $!foo, ${foo.bar}, $!{foo.bar}
        for match in re.finditer(r'\$!?\{?([\w.]+)\}?', stripped):
            var = "$" + match.group(1)
            variables.add(var)

        # HTML structural elements
        for tag in re.finditer(r'<(html|head|body|div|form|table|section|header|footer|nav|main|script|link|style)\b', stripped, re.IGNORECASE):
            html_structure.append(tag.group())

        # CSS classes used
        for cls in re.finditer(r'class="([^"]*)"', stripped):
            for c in cls.group(1).split():
                css_classes.add(c)

    if includes:
        out.append("\n### Includes / Parses")
        for inc in sorted(set(includes)):
            out.append(f"  {inc}")

    if macros:
        out.append("\n### Velocity Macros")
        for m in macros:
            out.append(f"  {m}")

    if variables:
        out.append(f"\n### Velocity Variables ({len(variables)} unique)")
        for v in sorted(variables)[:50]:
            out.append(f"  {v}")
        if len(variables) > 50:
            out.append(f"  ... and {len(variables) - 50} more")

    if html_structure:
        tag_counts = Counter(html_structure)
        out.append("\n### HTML Structure")
        for tag, count in tag_counts.most_common():
            out.append(f"  {tag}: {count}")

    if css_classes:
        out.append(f"\n### CSS Classes Used ({len(css_classes)} unique)")
        # Show naming patterns
        prefixes = set()
        for c in css_classes:
            parts = re.split(r'[-_]', c)
            if len(parts) > 1:
                prefixes.add(parts[0])
        if prefixes:
            out.append(f"  Common prefixes: {', '.join(sorted(prefixes)[:20])}")
        for c in sorted(css_classes)[:40]:
            out.append(f"  .{c}")
        if len(css_classes) > 40:
            out.append(f"  ... and {len(css_classes) - 40} more")

    # Detect Velocity directive brace/block style
    vm_block_same = 0
    vm_block_next = 0
    for idx, vline in enumerate(lines):
        vs = vline.strip()
        # #if, #foreach, #macro etc. — check if block content starts same line
        if re.match(r'^#(if|foreach|macro|else|elseif)\b', vs):
            # Content after the directive on the same line (beyond the condition)
            after = re.sub(r'^#\w+\s*\(.*?\)', '', vs).strip()
            if after:
                vm_block_same += 1
            else:
                vm_block_next += 1
    if vm_block_same + vm_block_next > 0:
        fmt_parts = []
        if vm_block_next == 0:
            fmt_parts.append("directive-body=same-line")
        elif vm_block_same == 0:
            fmt_parts.append("directive-body=next-line")
        else:
            fmt_parts.append(f"directive-body=same-line({vm_block_same})/next-line({vm_block_next})")
        out.append(f"\n### Formatting")
        out.append(f"  [FORMATTING: {', '.join(fmt_parts)}]")

    return "\n".join(out)


# ---------------------------------------------------------------------------
# Generic fallback skeleton
# ---------------------------------------------------------------------------

def _extract_generic_skeleton(source: str, filepath: str) -> str:
    lines = source.split("\n")
    out = [f"// File: {Path(filepath).name}  ({len(lines)} lines)"]

    # Just show the first 50 and last 20 non-empty lines
    non_empty = [line.rstrip() for line in lines if line.strip()]
    if len(non_empty) <= 70:
        out.extend(non_empty)
    else:
        out.extend(non_empty[:50])
        out.append(f"// ... ({len(non_empty) - 70} lines omitted) ...")
        out.extend(non_empty[-20:])

    return "\n".join(out)

"""
Prompt templates for each phase of the skill generation pipeline.

These are separated from the main script so you can easily tune them
for your codebase and LLM without touching orchestration logic.

Key design choices:
  - Analysis prompts include the project name so per-project patterns
    can be tracked during synthesis.
  - Naming and ordering conventions are first-class extraction targets,
    not afterthoughts.
  - Synthesis produces two tiers: universal patterns (across all projects)
    and project-specific deviations.
"""


def build_analysis_prompt(
    category: str,
    file_path: str,
    skeleton: str,
    raw_source: str,
    project_name: str = "",
) -> str:
    """Prompt for Phase 2: analyze a single file and extract patterns."""
    project_line = ""
    if project_name:
        project_line = f"\n**Project:** `{project_name}`"

    return f"""You are analyzing source files from a large codebase to document patterns and conventions.

## Task
Analyze this **{category}** file and extract all patterns, conventions, and structural choices.

## File
**Path:** `{file_path}`{project_line}

### Structural Skeleton
(Class structure, method signatures, annotations, ordering metadata — implementation stripped)
```
{skeleton}
```

### Full Source (may be truncated for large files)
```
{raw_source}
```

## What to Extract

Report on ALL of the following that apply. Be exhaustive — this data will be merged with
analyses of many other files, so every detail matters.

### 1. Naming Conventions (CRITICAL — be very specific)
   - **Class name**: Exact casing style (PascalCase?), suffix/prefix pattern (e.g. ends with DAO,
     Action, Service), relationship to file name
   - **Method names**: Casing style, verb prefixes used (get/set/find/create/delete/update/is/has),
     any naming formula (e.g. `findByField`, `getXxxForYyy`)
   - **Field/variable names**: Casing style (camelCase? snake_case?), any prefixes (m_ for members?
     s_ for static?), boolean naming (isXxx? hasXxx? xxxEnabled?)
   - **Constants**: Casing style (SCREAMING_SNAKE_CASE?), where declared (top of class? in interface?)
   - **Parameters**: Casing style, any naming patterns
   - **Package naming**: Structure, separator conventions, depth

### 2. Member Ordering (CRITICAL — describe the exact ordering you see)
   - **Field ordering**: What order are fields declared in? (constants first? static before instance?
     public before private? grouped by annotation? grouped by type?)
   - **Method ordering**: What order are methods declared in? (constructors first? then getters/setters?
     then business logic? then private helpers? alphabetical? CRUD order?)
   - **Import ordering**: How are imports grouped? (java.* first? third-party then internal?
     alphabetical within groups?)
   - **Describe the exact order as it appears in this file** — e.g. "constants, private fields,
     constructor, public getters, public setters, business methods, private helpers, toString/equals"

### 3. Class Hierarchy & Inheritance
   - What does this class extend? What interfaces does it implement?
   - Are there @Override methods? What are they overriding?
   - Any abstract methods or template method pattern?
   - Is there a common base class pattern?

### 4. Annotations & Decorators
   - Which annotations are used and on what (class, field, method, parameter)?
   - What annotation parameters / values are common?
   - Custom vs framework annotations
   - Ordering of multiple annotations on the same element

### 5. Common Patterns
   - Design patterns used (Builder, Factory, Repository, etc.)
   - Error handling approach (exceptions, error codes, Optional)
   - Null handling (Optional, @Nullable, null checks)
   - Logging patterns
   - Validation approach
   - Constructor patterns (no-arg? all-args? builder?)

### 6. Boilerplate & Repeated Code
   - Standard methods every file seems to have (toString, equals, hashCode, etc.)
   - Standard imports
   - Copy-paste patterns
   - Standard class-level Javadoc or comments

### 7. Framework-Specific Conventions
   - Framework annotations and their usage
   - Lifecycle methods
   - Configuration patterns
   - Dependency injection approach

### 8. Code Formatting & Layout Style (CRITICAL — be very specific)
   - **Brace placement**: Are opening braces on the SAME LINE as the declaration (K&R/1TBS style,
     e.g. `public void foo() {{`) or on the NEXT LINE (Allman style, e.g. brace on its own line)?
     Check class declarations, method declarations, if/for/while blocks, and inner classes separately.
   - **One-line methods**: Are simple getter/setter methods written on a SINGLE LINE
     (e.g. `public String getName() {{ return name; }}`) or always expanded to multiple lines?
     Also note any other short/transient methods that use single-line style.
   - **Line breaks between members**: How many BLANK LINES separate consecutive members?
     0 blank lines (fields back-to-back)? 1 blank line between each method? 2 blank lines
     between method groups? Are there different spacing rules for fields vs methods?
   - **Getter/setter proximity**: Are getters and setters for the same property placed
     IMMEDIATELY ADJACENT (e.g. getName() followed directly by setName() with only 0-1
     blank lines), or are all getters grouped together separately from all setters?
   - **Statement braces**: Are braces used for SINGLE-STATEMENT if/for/while blocks,
     or are braces omitted for one-liners? (e.g. `if (x) return y;` vs `if (x) {{ return y; }}`)
   - **Field grouping**: Are related fields grouped with no blank lines between them,
     with blank lines separating different groups?

### 9. Notable Specifics
   - Anything unusual or project-specific
   - Anti-patterns or inconsistencies worth documenting
   - Implicit conventions that might not be obvious

## Output Format
Use clear markdown headers. Be specific — quote exact annotation names, method signatures,
field names, and naming patterns. Include short code snippets (1-3 lines) to illustrate patterns.
Do NOT summarize what the file does functionally — focus only on structural patterns,
naming conventions, and ordering."""


def build_synthesis_prompt(
    category: str,
    analyses_text: str,
    skeleton_overview: str = "",
    project_stats: str = "",
) -> str:
    """
    Prompt for Phase 3: synthesize per-file analyses into a SKILL.md.

    project_stats: Summary of which patterns appear in which projects
    (provided by the pipeline for cross-project awareness).
    """
    skeleton_section = ""
    if skeleton_overview:
        skeleton_section = f"""

## Additional Structural Overview
(Patterns observed across the broader set of files, beyond the deeply-analyzed sample)

{skeleton_overview}
"""

    project_section = ""
    if project_stats:
        project_section = f"""

## Project Distribution
(Which projects each file came from — use this to identify project-specific vs universal patterns)

{project_stats}
"""

    return f"""You are writing a SKILL.md file — a reference document that an AI coding assistant will
read before creating or modifying **{category}** files in this codebase.

The codebase contains multiple projects (immediate subdirectories of the root). Most patterns
are shared across projects, but some projects have unique conventions. Your SKILL.md must
capture BOTH universal patterns and project-specific deviations.

## Context
Below are detailed analyses of multiple {category} files from the codebase, drawn from
various projects. Your job is to synthesize these into a single, authoritative SKILL.md.
{skeleton_section}{project_section}

## Per-File Analyses

{analyses_text}

## Output Requirements

Write the SKILL.md with this structure:

```markdown
---
name: {_slugify(category)}
description: [One-line description of what this skill covers and when to use it]
---

# {category} Patterns & Conventions

[Brief overview paragraph: what these files are, what framework/libraries they use,
and the general architectural approach]

## File Structure Template

[Show a canonical skeleton of what a typical file looks like, using a code block
with placeholder names. This is the single most important section — an AI should
be able to create a new file by filling in this template.

The template MUST reflect the correct member ordering convention.]

## Member Ordering Convention

[Describe the exact expected order of members within a file:
1. What order should fields appear in? (constants, static fields, instance fields, etc.)
2. What order should methods appear in? (constructors, getters/setters, business logic, etc.)
3. How should imports be ordered?
Show this as a numbered list showing the canonical ordering from top to bottom.]

## Code Formatting & Layout Style

[Document the exact formatting conventions observed. Be specific about each:

### Brace Placement
- Where does the opening brace go for class declarations? (same line or next line?)
- Where does the opening brace go for method declarations? (same line or next line?)
- Where does the opening brace go for control flow (if/for/while)?
- Show a short example of the brace style.

### One-Line Methods
- Are simple getters/setters written as single-line methods?
  (e.g. `public String getName() {{ return name; }}`)
- Are other short/transient methods written on one line?
- What is the threshold — when does a method get expanded to multiple lines?

### Line Breaks & Spacing
- How many blank lines between consecutive fields?
- How many blank lines between consecutive methods?
- How many blank lines between field section and method section?
- Are there different spacing rules for different member groups?

### Getter/Setter Proximity
- Are getter and setter for the same property placed immediately adjacent?
  (e.g. getName() immediately followed by setName())
- Or are all getters grouped together, then all setters?

### Statement Braces
- Are braces required for single-statement if/for/while blocks?
- Or are braces omitted for one-liners?]

## Naming Conventions

[Very specific naming rules. For each of these, state the exact convention:
- Class names: casing, suffix/prefix pattern, examples
- Method names: casing, verb prefixes used, naming formulas
- Field names: casing, prefix patterns, boolean naming
- Constants: casing, naming pattern
- Parameters: casing, naming pattern
- Packages: structure, naming pattern

Include concrete examples for each, taken from the analyzed files.]

## Class Hierarchy & Inheritance

[Common base classes, interfaces, override patterns. What does a typical class
extend/implement? What @Override methods are standard?]

## Required Annotations & Imports

[Standard annotations and imports every file should have, with explanation of each.
Note the expected ordering of annotations when multiple are stacked.]

## Common Patterns

[Design patterns, method patterns, CRUD patterns — with short code examples]

## Error Handling

[How errors are handled in this category of file]

## Do's and Don'ts

[Explicit list of conventions to follow and anti-patterns to avoid.
Base these on what you ACTUALLY observed, not generic best practices.]

## Project-Specific Variations

[IMPORTANT: Document any patterns that differ between projects. For each variation:
- Which project(s) use this variation?
- How does it differ from the universal pattern?
- When should an AI coding assistant use the variant vs the default?

If all projects follow the same conventions, state that explicitly:
"All projects follow the same conventions for this file category."]

## Canonical Example

[One complete, realistic (but simplified) example file that demonstrates
all the universal patterns above. This should be 30-60 lines, not a full production file.]
```

## Important Guidelines

- Only document patterns you actually observed in the analyses. Do NOT invent conventions.
- If you saw variation/inconsistency, document the dominant pattern and note the variation.
- Pay special attention to NAMING, ORDERING, and FORMATTING (brace placement, line breaks,
  one-line methods, getter/setter proximity) — these are the most common style violations.
- When a pattern is unique to a specific project, put it in "Project-Specific Variations",
  not in the universal sections.
- Be specific: use actual annotation names, method signatures, field names as examples.
- Prefer showing code over describing code.
- The audience is an AI coding assistant. Be precise and unambiguous.
- Keep the total document under 500 lines. Prioritize the template, ordering, and naming."""


def build_validation_prompt(
    category: str,
    skill_md: str,
    test_files: str,
    test_file_projects: str = "",
) -> str:
    """Prompt for Phase 4: validate a SKILL.md against unseen files."""
    project_note = ""
    if test_file_projects:
        project_note = f"""
Note: These test files come from these projects: {test_file_projects}
Check whether project-specific variations documented in the SKILL.md are accurate.
"""

    return f"""You are validating a SKILL.md document against files it has never seen.

## The SKILL.md (for {category} files)

```markdown
{skill_md}
```

## Unseen Test Files
{project_note}
The following files were NOT used to create the SKILL.md above. Check how well the
documented patterns match these files.

{test_files}

## Your Task

For each test file:

1. **Conformance Score** (1-10): How well does it match the SKILL.md patterns?
2. **Naming Compliance**: Do class names, method names, field names, and constants follow
   the documented conventions? Note any deviations.
3. **Ordering Compliance**: Does the member ordering match the documented convention?
   Note any deviations.
4. **Formatting Compliance**: Does the brace placement, line break spacing, one-line method
   style, and getter/setter proximity match the documented formatting conventions?
5. **Pattern Matches**: Which documented patterns are correctly present?
6. **Gaps**: What patterns does this file use that are NOT in the SKILL.md?
7. **Conflicts**: Does anything directly contradict the SKILL.md?
8. **Project Accuracy**: If this file is from a specific project, do the project-specific
   variations apply correctly?

Then provide an overall assessment:

### Overall Assessment
- **Coverage Score** (1-10): What percentage of real patterns does the SKILL.md capture?
- **Accuracy Score** (1-10): Are the documented patterns correct?
- **Naming Convention Accuracy** (1-10): How accurate are the naming rules?
- **Ordering Convention Accuracy** (1-10): How accurate is the member ordering description?
- **Formatting Convention Accuracy** (1-10): How accurate are the brace placement, line break,
  and code layout rules?
- **Missing Patterns**: Significant patterns found in test files but absent from SKILL.md
- **Suggested Additions**: Specific text to add to the SKILL.md to fill gaps
- **Suggested Corrections**: Specific text to change in the SKILL.md

Be specific. Quote exact names, annotations, or code from both the SKILL.md and test files."""


def _slugify(text: str) -> str:
    """Convert a label to a slug for SKILL.md frontmatter."""
    return text.lower().replace(" ", "-").replace("/", "-").replace("(", "").replace(")", "")

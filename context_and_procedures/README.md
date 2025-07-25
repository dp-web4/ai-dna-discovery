# Context and Procedures System

## Overview

This directory contains living checklists and procedures that evolve as we learn from our experiences. Like aviation checklists, these procedures ensure safety, consistency, and quality in our complex AI research work.

## Structure

```
context_and_procedures/
├── README.md                          # This file
├── checklists/                        # Specific task checklists
│   ├── version_compatibility.md      # ALWAYS CHECK FIRST
│   ├── git_operations.md             # Git workflow procedures
│   ├── pdf_generation.md             # PDF creation from markdown
│   ├── hardware_changes.md           # Hardware configuration safety
│   ├── research_documentation.md     # Research paper standards
│   └── code_development.md           # Development workflows
├── prohibitions/                      # Things we must never do
│   ├── git_prohibitions.md           # Git don'ts
│   ├── hardware_prohibitions.md      # Hardware don'ts
│   └── general_prohibitions.md       # General safety rules
├── advisories/                        # Best practices and recommendations
│   ├── project_structure.md          # How to organize projects
│   ├── naming_conventions.md         # File and variable naming
│   └── testing_guidelines.md         # Testing before deployment
└── templates/                         # Reusable templates
    ├── experiment_template.py         # Standard experiment structure
    ├── report_template.md            # Research report format
    └── README_template.md            # Project README format
```

## Philosophy

1. **Prescriptive**: Step-by-step procedures for complex tasks
2. **Advisory**: Best practices and recommendations
3. **Prohibitive**: Clear boundaries on what not to do
4. **Living Document**: Updated based on lessons learned
5. **Accessible**: Clear, concise, actionable

## Usage

Before starting any task:
1. Check if a relevant checklist exists
2. Follow the checklist completely
3. Update the checklist if you discover improvements
4. Document any incidents for future learning

## Quick Links

- **[Version Compatibility Checklist](checklists/version_compatibility.md)** ⚠️ CHECK FIRST
- [Git Operations Checklist](checklists/git_operations.md)
- [PDF Generation Checklist](checklists/pdf_generation.md)
- [Hardware Changes Checklist](checklists/hardware_changes.md)
- [Git Prohibitions](prohibitions/git_prohibitions.md)
- [Hardware Prohibitions](prohibitions/hardware_prohibitions.md)

## Contributing

When updating these procedures:
1. Be specific and actionable
2. Include examples where helpful
3. Explain the "why" behind rules
4. Date significant changes
5. Test procedures before committing
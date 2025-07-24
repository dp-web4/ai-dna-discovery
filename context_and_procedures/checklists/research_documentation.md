# Research Documentation Checklist

*Last Updated: 2025-01-24*

## Planning Phase

### Before Starting Research
1. [ ] Define clear research questions
2. [ ] Set success criteria
3. [ ] Plan experiment phases
4. [ ] Identify required resources
5. [ ] Create timeline estimates

### Documentation Structure
1. [ ] Create dedicated results directory
2. [ ] Set up phase-based organization
3. [ ] Plan naming convention (timestamps)
4. [ ] Prepare visualization directory
5. [ ] Set up git tracking

## During Research

### Experiment Execution
1. [ ] Name files with timestamps: `experiment_YYYYMMDD_HHMMSS.json`
2. [ ] Save raw results immediately
3. [ ] Create visualizations for each phase
4. [ ] Write phase report while fresh
5. [ ] Commit results after each phase

### Data Collection Standards
```python
{
    "experiment": "name",
    "timestamp": "ISO 8601 format",
    "parameters": {...},
    "results": {...},
    "metrics": {...},
    "notes": "observations"
}
```

### Progress Tracking
1. [ ] Update cumulative progress report
2. [ ] Link new results in main README
3. [ ] Create phase summary
4. [ ] Note unexpected findings
5. [ ] Document any failures

## Writing Reports

### Phase Reports Should Include
1. [ ] Objective and hypothesis
2. [ ] Methodology
3. [ ] Results with visualizations
4. [ ] Analysis and interpretation
5. [ ] Connection to overall research

### Synthesis Reports
1. [ ] Connect all phases coherently
2. [ ] Identify emergent patterns
3. [ ] State unified findings
4. [ ] Include all visualizations
5. [ ] Provide executive summary

### Research Papers (50+ pages)
1. [ ] Abstract (1 page)
2. [ ] Introduction with context
3. [ ] Literature review
4. [ ] Detailed methodology
5. [ ] Results for each phase
6. [ ] Analysis and discussion
7. [ ] Conclusions
8. [ ] Future work
9. [ ] References
10. [ ] Appendices with raw data

## Visualization Standards

### Required Visualizations
1. [ ] Time series for temporal data
2. [ ] Heatmaps for matrices
3. [ ] Network graphs for relationships
4. [ ] Bar charts for comparisons
5. [ ] Save as PNG with descriptive names

### Visualization Guidelines
```python
plt.figure(figsize=(12, 8))
plt.title('Descriptive Title', fontsize=16)
plt.xlabel('Clear Label', fontsize=12)
plt.ylabel('Clear Label', fontsize=12)
plt.tight_layout()
plt.savefig('phase_X_metric_description.png', dpi=300, bbox_inches='tight')
```

## File Organization

### Directory Structure
```
research_project/
├── phase_1_results/
│   ├── experiment_20250124_093000.json
│   ├── phase_1_report.md
│   └── visualizations/
├── phase_2_results/
├── synthesis/
│   ├── FINAL_SYNTHESIS_REPORT.md
│   └── comprehensive_research_paper.md
├── AUTONOMOUS_RESEARCH_PLAN.md
└── CUMULATIVE_PROGRESS_REPORT.md
```

### Naming Conventions
- Results: `experiment_YYYYMMDD_HHMMSS.json`
- Reports: `phase_N_report.md`
- Visualizations: `description_metric_phaseN.png`
- NO COLONS in filenames (Windows compatibility)

## Quality Checks

### Before Committing Results
1. [ ] All experiments have timestamps
2. [ ] Visualizations are generated
3. [ ] Phase report is complete
4. [ ] Data is backed up
5. [ ] No sensitive information exposed

### Before Publishing Papers
1. [ ] All claims are supported by data
2. [ ] Methodology is reproducible
3. [ ] Visualizations are high quality
4. [ ] References are accurate
5. [ ] Formatting is consistent

## Common Pitfalls

### Data Management
- ❌ Overwriting previous results
- ❌ Forgetting to save raw data
- ❌ Using unclear filenames
- ❌ Missing timestamp information

### Documentation
- ❌ Writing reports weeks later
- ❌ Omitting failed experiments
- ❌ Not explaining anomalies
- ❌ Forgetting reproducibility details

### Visualization
- ❌ Unclear axes labels
- ❌ Missing units
- ❌ Inconsistent color schemes
- ❌ Too small text/elements

## Best Practices

### Reproducibility
1. ✅ Document all parameters
2. ✅ Include random seeds
3. ✅ Note software versions
4. ✅ Describe hardware used
5. ✅ Provide runnable code

### Clarity
1. ✅ Write for uninformed audience
2. ✅ Define all terms
3. ✅ Use consistent notation
4. ✅ Include examples
5. ✅ Provide context

### Integrity
1. ✅ Report all results (including failures)
2. ✅ Acknowledge limitations
3. ✅ Distinguish correlation from causation
4. ✅ Be honest about uncertainty
5. ✅ Credit all contributions

## Post-Research

### Archival
1. [ ] Create complete archive
2. [ ] Include all dependencies
3. [ ] Document environment setup
4. [ ] Create README for archive
5. [ ] Test reproducibility

### Sharing
1. [ ] Generate PDF versions
2. [ ] Create presentation slides
3. [ ] Prepare executive summary
4. [ ] Make data available
5. [ ] Consider licensing

---

**Remember**: Good documentation during research saves months of reconstruction later. Document as if you'll forget everything (because you will).
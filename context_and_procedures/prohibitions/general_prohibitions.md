# General Development Prohibitions

*Last Updated: 2025-01-24*

## File and Naming Conventions

### Forbidden Characters in Filenames
- ❌ **NEVER** use colons (:) - breaks Windows
- ❌ **NEVER** use angle brackets (< >)
- ❌ **NEVER** use quotes (" ')
- ❌ **NEVER** use pipe (|)
- ❌ **NEVER** use question mark (?)
- ❌ **NEVER** use asterisk (*) except in .gitignore

**Why**: These characters are reserved on various operating systems and cause sync/pull failures.

### Naming Conflicts
- ❌ **NEVER** create files differing only in case (File.txt vs file.txt)
- ❌ **NEVER** use spaces at start/end of names
- ❌ **NEVER** use only dots and spaces
- ❌ **NEVER** exceed 255 character names
- ❌ **NEVER** create paths over 260 characters (Windows limit)

**Why**: Filesystem differences between platforms cause conflicts.

## Development Practices

### Version Assumptions
- ❌ **NEVER** assume version compatibility
- ❌ **NEVER** skip version checks before installation
- ❌ **NEVER** mix major versions without research
- ❌ **NEVER** ignore deprecation warnings
- ❌ **NEVER** assume newer is always compatible

**Why**: Version mismatches are a leading cause of system failures.

### Testing and Deployment
- ❌ **NEVER** deploy untested code
- ❌ **NEVER** test in production first
- ❌ **NEVER** skip error handling
- ❌ **NEVER** ignore warnings
- ❌ **NEVER** assume "it works on my machine" is enough

**Why**: Untested code leads to emergencies and data loss.

### Documentation
- ❌ **NEVER** commit undocumented breaking changes
- ❌ **NEVER** delete documentation without replacement
- ❌ **NEVER** use internal references in public docs
- ❌ **NEVER** document passwords or keys
- ❌ **NEVER** assume readers have context

**Why**: Poor documentation wastes everyone's time and creates confusion.

## Resource Management

### Memory and Storage
- ❌ **NEVER** ignore memory leaks
- ❌ **NEVER** load entire datasets into memory
- ❌ **NEVER** create infinite loops without breaks
- ❌ **NEVER** fill up disk space
- ❌ **NEVER** ignore cleanup in error paths

**Why**: Resource exhaustion crashes systems and loses work.

### Process Management
- ❌ **NEVER** leave zombie processes
- ❌ **NEVER** fork bomb (even accidentally)
- ❌ **NEVER** ignore process limits
- ❌ **NEVER** kill -9 without trying graceful first
- ❌ **NEVER** assume infinite resources

**Why**: System stability depends on proper resource management.

## Security and Privacy

### Data Handling
- ❌ **NEVER** log personally identifiable information
- ❌ **NEVER** store credentials in code
- ❌ **NEVER** transmit passwords in plain text
- ❌ **NEVER** ignore security warnings
- ❌ **NEVER** use production data in development

**Why**: Data breaches have legal and ethical consequences.

### Access Control
- ❌ **NEVER** share access tokens
- ❌ **NEVER** disable security features for convenience
- ❌ **NEVER** use admin/root unnecessarily
- ❌ **NEVER** ignore permission errors
- ❌ **NEVER** make security assumptions

**Why**: Security breaches can destroy trust and projects.

## Communication and Collaboration

### Code Sharing
- ❌ **NEVER** share code without license clarity
- ❌ **NEVER** use code without checking licenses
- ❌ **NEVER** remove attribution
- ❌ **NEVER** claim others' work as your own
- ❌ **NEVER** violate NDAs or confidentiality

**Why**: Legal issues can end projects and careers.

### Team Interaction
- ❌ **NEVER** push breaking changes without warning
- ❌ **NEVER** force push shared branches
- ❌ **NEVER** ignore code review feedback
- ❌ **NEVER** work on same files without coordination
- ❌ **NEVER** assume others know your changes

**Why**: Team friction reduces productivity and morale.

## System Administration

### Package Management
- ❌ **NEVER** use sudo pip install
- ❌ **NEVER** mix package managers
- ❌ **NEVER** ignore version conflicts
- ❌ **NEVER** install from untrusted sources
- ❌ **NEVER** modify system Python

**Why**: System corruption requires OS reinstallation.

### Configuration Changes
- ❌ **NEVER** edit configs without backups
- ❌ **NEVER** apply untested configurations
- ❌ **NEVER** ignore syntax errors
- ❌ **NEVER** use hardcoded paths
- ❌ **NEVER** commit local-only configs

**Why**: Configuration errors can make systems unusable.

## AI/ML Specific

### Model Management
- ❌ **NEVER** commit model weights to git
- ❌ **NEVER** train on test data
- ❌ **NEVER** ignore bias in datasets
- ❌ **NEVER** deploy without evaluation
- ❌ **NEVER** assume model outputs are safe

**Why**: ML mistakes can be subtle but impactful.

### GPU Usage
- ❌ **NEVER** monopolize shared GPUs
- ❌ **NEVER** ignore out-of-memory errors
- ❌ **NEVER** leave processes running unnecessarily
- ❌ **NEVER** assume GPU availability
- ❌ **NEVER** skip GPU memory cleanup

**Why**: GPUs are expensive shared resources.

## Edge Cases That Bit Us

### The Virtual Environment Incident
```
Action: Created venv then updated .gitignore
Result: Committed thousands of venv files
Lesson: ALWAYS update .gitignore FIRST
```

### The Colon Filename Issue
```
Action: Named file "phi3:mini_results.json"
Result: Windows users couldn't pull
Lesson: Cross-platform compatibility matters
```

### The ACPI Setting
```
Action: Changed ARM device to use ACPI
Result: Device wouldn't boot
Lesson: Research hardware changes FIRST
```

## Recovery Principles

When you violate a prohibition:
1. **Stop immediately** - Don't compound the error
2. **Assess damage** - Understand what happened
3. **Document issue** - For future prevention
4. **Fix properly** - No quick hacks
5. **Update procedures** - Learn from mistakes

## The Meta-Rule

❌ **NEVER** assume you know better than established procedures

These prohibitions exist because someone (often us) learned the hard way. Ignoring them doesn't show expertise - it shows inexperience.

---

**Remember**: Every prohibition here cost someone hours or days of work. Learn from our mistakes rather than repeating them.
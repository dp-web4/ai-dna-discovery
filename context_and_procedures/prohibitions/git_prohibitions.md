# Git Prohibitions

*Last Updated: 2025-01-24*

## NEVER Do These Things

### 1. Virtual Environments
- ❌ **NEVER** commit virtual environment directories
- ❌ **NEVER** create a venv before checking `.gitignore`
- ❌ **NEVER** assume a directory is ignored without verifying

**Why**: Virtual environments can contain thousands of files and hundreds of MB. They're platform-specific and should be recreated locally.

### 2. Large Files
- ❌ **NEVER** commit files over 100MB without Git LFS
- ❌ **NEVER** commit model weights (.safetensors, .pth, .bin)
- ❌ **NEVER** commit datasets or large media files

**Why**: Git is designed for code, not large binaries. Large files bloat the repository forever.

### 3. Sensitive Information
- ❌ **NEVER** commit API keys, tokens, or passwords
- ❌ **NEVER** commit `.env` files with real credentials
- ❌ **NEVER** commit personal information or PII

**Why**: Once committed, sensitive data is in git history forever and can be exposed.

### 4. Git Add Wildcards
- ❌ **NEVER** use `git add .` without checking `git status` first
- ❌ **NEVER** use `git add *` in a directory with mixed content
- ❌ **NEVER** stage files you haven't reviewed

**Why**: Wildcards can accidentally include unintended files.

### 5. Force Pushing
- ❌ **NEVER** force push to main/master branch
- ❌ **NEVER** force push without coordinating with team
- ❌ **NEVER** force push to rewrite public history

**Why**: Force pushing can destroy others' work and break CI/CD.

### 6. Windows-Incompatible Filenames
- ❌ **NEVER** use colons (:) in filenames
- ❌ **NEVER** use reserved Windows characters (< > : " | ? *)
- ❌ **NEVER** create files with only case differences

**Why**: These cause failures when Windows users pull the repository.

### 7. Committing Without Testing
- ❌ **NEVER** commit code that doesn't run
- ❌ **NEVER** commit with syntax errors
- ❌ **NEVER** commit breaking changes without documentation

**Why**: Broken commits disrupt everyone's workflow.

### 8. Rewriting Public History
- ❌ **NEVER** rebase commits that have been pushed
- ❌ **NEVER** amend public commits
- ❌ **NEVER** change commit messages after pushing

**Why**: This causes conflicts for everyone who has pulled those commits.

### 9. Ignoring Git Status Warnings
- ❌ **NEVER** proceed when git shows errors
- ❌ **NEVER** ignore merge conflicts
- ❌ **NEVER** commit with unresolved issues

**Why**: Git warnings exist to prevent data loss and corruption.

### 10. Binary Files in Code Directories
- ❌ **NEVER** mix binary files with source code
- ❌ **NEVER** commit compiled outputs
- ❌ **NEVER** include build artifacts

**Why**: Binary files change completely even with small modifications, bloating history.

## What To Do Instead

### Virtual Environments
✅ Add to `.gitignore` BEFORE creating
✅ Use standard names (venv, env)
✅ Document setup in README

### Large Files
✅ Use Git LFS for files over 100MB
✅ Store models/data externally
✅ Use URLs or scripts to download

### Sensitive Information
✅ Use environment variables
✅ Create `.env.example` with dummy values
✅ Use secrets management tools

### Adding Files
✅ Review `git status` first
✅ Add files individually or by specific directory
✅ Use `git add -p` for partial adds

### Collaboration
✅ Communicate before force operations
✅ Use feature branches
✅ Test before pushing

## Recovery If You Break These Rules

1. **Don't push** - fix locally first
2. **Ask for help** - two heads better than one
3. **Document** - update these rules with lessons learned
4. **Learn** - mistakes are how we improve

---

**Remember**: These rules exist because someone (maybe us!) learned the hard way. Following them saves time and prevents headaches.
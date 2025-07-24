# Git Operations Checklist

*Last Updated: 2025-01-24*

## Before Any Git Operation

### 1. Check Git Status
```bash
git status
```
- [ ] Review all modified files
- [ ] Check for untracked files
- [ ] Verify you're on the correct branch

### 2. Virtual Environment Check
**CRITICAL**: Virtual environments must NEVER be committed
- [ ] Check for any `venv/`, `env/`, `*_venv/` directories
- [ ] Verify `.gitignore` includes all venv patterns
- [ ] If venv exists in untracked files, verify it's ignored

## Creating/Updating .gitignore

### Before Creating Virtual Environment
1. [ ] Ensure `.gitignore` exists
2. [ ] Add venv patterns BEFORE creating venv:
   ```
   # Virtual Environments
   env/
   venv/
   ENV/
   .venv
   *_venv/
   */venv/
   */env/
   virtualenv/
   .virtualenv/
   ```
3. [ ] Commit `.gitignore` changes
4. [ ] THEN create virtual environment

## Adding Files to Git

### Pre-Add Checklist
1. [ ] Run `git status` to see all changes
2. [ ] Review each untracked file/directory
3. [ ] Check file sizes (nothing over 100MB)
4. [ ] Verify no sensitive information:
   - [ ] No API keys or tokens
   - [ ] No passwords or credentials
   - [ ] No personal information
   - [ ] No large model files (.safetensors, .pth, .bin)

### Safe Add Procedure
1. [ ] Use specific file paths, not wildcards:
   ```bash
   # Good
   git add specific_file.py
   git add specific_directory/
   
   # Avoid
   git add .
   git add *
   ```
2. [ ] After adding, run `git status` again
3. [ ] Verify only intended files are staged

## Removing Accidentally Tracked Files

### If Virtual Environment Was Tracked
1. [ ] Remove from tracking (keeps local files):
   ```bash
   git rm -r --cached path/to/venv/
   ```
2. [ ] Update `.gitignore` to prevent re-adding
3. [ ] Commit the removal:
   ```bash
   git commit -m "Remove virtual environment from tracking"
   ```
4. [ ] Push changes

### General File Removal
1. [ ] Identify the file/directory to remove
2. [ ] Use `--cached` to keep local copy:
   ```bash
   git rm --cached file_to_remove
   ```
3. [ ] Update `.gitignore` if needed
4. [ ] Commit and push

## Committing Changes

### Pre-Commit Checklist
1. [ ] Run `git diff --staged` to review changes
2. [ ] Ensure commit is focused (one logical change)
3. [ ] Check no debug code is included
4. [ ] Verify no large files are staged

### Commit Message Guidelines
1. [ ] Use present tense ("Add" not "Added")
2. [ ] First line under 50 characters
3. [ ] Reference issue numbers if applicable
4. [ ] Be specific about what changed

## Common Pitfalls to Avoid

### Virtual Environment Issues
- **Never** create venv before updating `.gitignore`
- **Never** use `git add .` without checking untracked files
- **Always** check if directory name matches `.gitignore` patterns

### Large Files
- **Check** file sizes before committing
- **Use** Git LFS for files over 100MB
- **Never** commit model weights or datasets

### Sensitive Information
- **Never** commit files containing credentials
- **Always** use environment variables for secrets
- **Review** files before staging

## Recovery Procedures

### If You Committed a Virtual Environment
1. [ ] Don't panic - it's fixable
2. [ ] Follow removal procedure above
3. [ ] Force push if necessary (coordinate with team)
4. [ ] Update `.gitignore` to prevent recurrence

### If You Committed Sensitive Data
1. [ ] Remove from all commits using `git filter-branch` or BFG
2. [ ] Force push all branches
3. [ ] Rotate any exposed credentials immediately
4. [ ] Document incident for team awareness

## Post-Operation Verification

1. [ ] Run `git status` - should be clean or expected state
2. [ ] Check remote with `git log origin/branch -1`
3. [ ] Verify `.gitignore` is comprehensive
4. [ ] Document any lessons learned

---

**Remember**: When in doubt, ask before pushing. It's easier to fix local mistakes than remote ones.
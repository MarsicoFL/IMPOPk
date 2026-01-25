# Contributing to HPRCv2-IBD

This document provides guidelines for commits, pushes, and contributions to the project.

## Repository Focus

This repository tracks the **package development** (src/) - the Rust CLI tools for IBD detection. Experiment data and results are generated locally and excluded from version control due to size.

## What Gets Tracked

### Tracked (committed to git)
- `src/` - All Rust source code
- `docs/` - Documentation and tutorials
- `data/samples/` - Population sample lists
- `data/README.md` - Data documentation
- `experiments/**/scripts/` - Analysis scripts
- `experiments/**/results/json/` - Small JSON summaries
- `experiments/**/README.md` - Experiment documentation
- `reports/main/*.tex` - LaTeX sources
- `reports/main/*.md` - Markdown documentation
- Configuration files (Cargo.toml, .gitignore, etc.)

### NOT Tracked (gitignored)
- `**/target/` - Build artifacts
- `experiments/**/data/*.tsv` - Large data files
- `data/assemblies/` - External data symlinks
- `data/alignments/` - External data symlinks
- `reports/**/figures/` - Generated figures (regenerable)
- `archive/` - Old versions
- `.claude/` - Local AI configuration

## Commit Guidelines

### Commit Message Format

```
<type>(<scope>): <short description>

<optional body>

<optional footer>
```

### Types
- `feat` - New feature
- `fix` - Bug fix
- `docs` - Documentation changes
- `refactor` - Code refactoring
- `perf` - Performance improvement
- `test` - Adding tests
- `chore` - Maintenance tasks

### Scopes
- `ibd` - ibd-cli tool
- `ibs` - ibs-cli tool
- `jacquard` - jacquard-cli tool
- `docs` - Documentation
- `experiments` - Experiment scripts
- `reports` - Reports and figures

### Examples

```bash
# Feature
git commit -m "feat(ibd): add parallel Viterbi decoding"

# Bug fix
git commit -m "fix(ibs): correct window boundary calculation"

# Documentation
git commit -m "docs: update quickstart tutorial"

# Refactoring
git commit -m "refactor(ibd): simplify HMM state transitions"
```

## Branching Strategy

### Main Branches
- `main` - Stable release code
- `develop` - Development integration (if used)

### Feature Branches
```bash
# Create feature branch
git checkout -b feat/parallel-viterbi

# Work on feature
git add src/ibd-cli/src/hmm.rs
git commit -m "feat(ibd): implement parallel Viterbi"

# Push to remote
git push -u origin feat/parallel-viterbi

# Create PR to main
```

### Naming Convention
- `feat/<description>` - New features
- `fix/<description>` - Bug fixes
- `docs/<description>` - Documentation
- `refactor/<description>` - Refactoring

## Push Workflow

### Before Pushing

1. **Check status**
   ```bash
   git status
   ```

2. **Review changes**
   ```bash
   git diff --staged
   ```

3. **Run tests** (if available)
   ```bash
   cd src/ibd-cli && cargo test
   cd ../ibs-cli && cargo test
   ```

4. **Build check**
   ```bash
   cd src/ibd-cli && cargo build --release
   cd ../ibs-cli && cargo build --release
   ```

### Pushing

```bash
# Push current branch
git push origin <branch-name>

# Push main (after merge)
git push origin main
```

### Never Force Push to Main
```bash
# NEVER do this on main
git push --force origin main  # DANGEROUS
```

## Code Style

### Rust
- Follow rustfmt conventions
- Run `cargo fmt` before committing
- Run `cargo clippy` for lints

### Python
- Follow PEP 8
- Use meaningful variable names
- Add docstrings to functions

### Documentation
- Use Markdown
- Include code examples
- Keep README files updated

## Pull Request Checklist

Before creating a PR:

- [ ] Code compiles without errors
- [ ] Tests pass (if applicable)
- [ ] Documentation updated
- [ ] Commit messages follow format
- [ ] No large data files included
- [ ] .gitignore respected

## Useful Commands

```bash
# Check what will be committed
git status

# Stage specific files
git add src/ibd-cli/src/hmm.rs

# Stage all changes in directory
git add src/ibd-cli/

# Commit with message
git commit -m "feat(ibd): description"

# Push to remote
git push origin <branch>

# Pull latest changes
git pull origin main

# Create and switch to branch
git checkout -b feat/new-feature

# Switch to existing branch
git checkout main

# Merge branch into main
git checkout main
git merge feat/new-feature

# Delete local branch
git branch -d feat/new-feature

# View commit history
git log --oneline -10

# View what's ignored
git status --ignored
```

## Contact

For questions about contributing, open an issue on the repository.

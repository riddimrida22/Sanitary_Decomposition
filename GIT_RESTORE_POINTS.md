# Git Restore Points (Beginner Guide)

This project now uses Git to track changes.

## 1) Check what changed
```bash
git status
git diff
```

## 2) Save a restore point (commit)
```bash
git add -A
git commit -m "Describe what you changed"
```

## 3) See history
```bash
git log --oneline --decorate -n 20
```

## 4) Restore one file from an older commit
```bash
git checkout <commit_id> -- path/to/file.py
```

## 5) Return the whole project to a commit (safe method)
```bash
git switch -c restore-test-branch <commit_id>
```
This creates a new branch at that old point so you can inspect it safely.

## 6) Create a named restore point tag
```bash
git tag restore-YYYYMMDD-HHMM
```

## 7) List restore point tags
```bash
git tag -l "restore-*"
```

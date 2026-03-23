# Code Quality & Audit Readiness Plan

**Date:** March 18, 2026
**Project:** AMS Admin Tool
**Purpose:** Prioritized plan to prepare for post-launch third-party code audit

---

## Current State (as of today)

| Metric                                    | Value                                                                              | Status         |
| ----------------------------------------- | ---------------------------------------------------------------------------------- | -------------- |
| Total source files                        | 424 (230 tsx, 189 ts)                                                              | —              |
| TypeScript errors (`tsc --noEmit`)        | 60 errors across 15 files                                                          | Needs fixing   |
| ESLint errors                             | 30 errors (17 `no-explicit-any`, 7 unused vars, 3 export issues, 1 rules-of-hooks) | Needs fixing   |
| Tests                                     | 597 passed, 4 failed (36 test files, 601 total tests)                              | Needs fixing   |
| Dependency vulnerabilities (`pnpm audit`) | 13 (1 low, 2 moderate, 10 high)                                                    | Needs fixing   |
| Test coverage                             | Not measured                                                                       | Not configured |
| CI pipeline                               | Runs tests + build only. Lint and type-check are **commented out**                 | Incomplete     |
| SonarQube / static analysis               | Not configured                                                                     | Not set up     |
| Pre-commit hooks                          | Not configured                                                                     | Not set up     |

---

## P1 — High Value, Do First

These are the items auditors will check immediately. Failing any of these is a red flag.

### 1.1 Fix Failing Tests

- **Current:** 4 tests failing across 4 files
- **Files:**
  - `src/pages/JourneyListPage/__tests__/JourneyListPage.hook.test.ts`
  - `src/organisms/ScriptEditorPanel/__tests__/ScriptEditorPanel.test.tsx`
  - `src/organisms/ScriptEditorPanel/Layout/DynamicTemplateSection/__tests__/DynamicButtonsEditor.test.tsx`
  - `src/atoms/ConditionBuilder/ConditionBuilder.test.tsx`
- **Effort:** Low (likely broken by recent UI changes)
- **Audit impact:** High — 100% test pass rate is expected

### 1.2 Fix All ESLint Errors

- **Current:** 30 errors
- **Breakdown:**
  - 17 `no-explicit-any` — replace with proper types
  - 7 `no-unused-vars` — remove dead code
  - 3 `react-refresh/only-export-components` — minor restructuring
  - 1 `react-hooks/rules-of-hooks` — logic fix needed
- **Effort:** Medium (1-2 hours)
- **Audit impact:** High — zero lint errors shows discipline

### 1.3 Fix TypeScript Errors

- **Current:** 60 errors across 15 files
- **Top offenders:**
  - `src/data/marketingJourneyFallback.ts` — 14 errors (type mismatches)
  - `src/organisms/BuilderFlowCanvas/__tests__/` — 11 errors (test mocks)
  - `src/organisms/FlowCanvas/__tests__/` — 5 errors (test mocks)
  - `src/constants/templates/defaults/dynamicDefaults.ts` — 4 errors
  - `src/store/slices/user/userSlice.ts` — 3 errors
- **Effort:** Medium (2-3 hours)
- **Audit impact:** High — strict TS with zero errors is a key quality signal

### 1.4 Enable Lint + Type-Check in CI

- **Current:** Both are commented out in `.github/workflows/ci.yml` (lines 50-54)
- **Action:** Uncomment `pnpm lint` and `pnpm exec tsc --noEmit`
- **Effort:** 5 minutes (after 1.1-1.3 are done)
- **Audit impact:** High — proves quality gates are enforced, not optional

### 1.5 Fix Dependency Vulnerabilities

- **Current:** 13 vulnerabilities (10 high severity)
- **Action:** Run `pnpm audit`, upgrade or patch affected packages
- **Effort:** Low-Medium (depends on breaking changes)
- **Audit impact:** High — security vulnerabilities are audit blockers

---

## P2 — Medium Value, Do Before Audit

These demonstrate maturity and make the audit report look strong.

### 2.1 Add Test Coverage Reporting

- **Action:** Configure `vitest` with coverage (`@vitest/coverage-v8`)
- **Steps:**
  1. Install `@vitest/coverage-v8`
  2. Add coverage config in `vitest.config.ts`
  3. Set baseline threshold (e.g., 50% lines, 40% branches)
  4. Add `pnpm test:coverage` script
  5. Add coverage step in CI
- **Effort:** 1 hour
- **Audit impact:** Medium-High — auditors want to see a coverage number

### 2.2 Set Up SonarQube (Docker)

- **Action:** Run SonarQube Community Edition via Docker for automated code quality analysis
- **Why Docker (not SonarCloud):** Free, self-hosted, no data leaves our infra, same dashboards and quality gates
- **What it gives you:**
  - Code smells, bugs, security hotspots
  - Duplication percentage
  - Maintainability rating (A-E grade)
  - A **shareable dashboard URL** for auditors (accessible on internal network or via tunnel)
- **Steps:**
  1. Run SonarQube container: `docker run -d --name sonarqube -p 9000:9000 sonarqube:community`
  2. Create project + generate token in SonarQube UI (http://localhost:9000)
  3. Add `sonar-project.properties` to repo root:
     ```
     sonar.projectKey=ams-admin-tool
     sonar.sources=src
     sonar.exclusions=**/*.test.*,**/*.stories.*,**/node_modules/**
     sonar.typescript.lcov.reportPaths=coverage/lcov.info
     ```
  4. Run scanner via Docker:
     ```
     docker run --rm -v "$(pwd):/usr/src" sonarsource/sonar-scanner-cli \
       -Dsonar.host.url=http://host.docker.internal:9000 \
       -Dsonar.token=YOUR_TOKEN
     ```
  5. Optionally add scanner step to CI (or run on-demand before audits)
  6. Fix any critical/blocker issues it finds
- **Effort:** 1-2 hours (setup + initial scan)
- **Audit impact:** Medium-High — gold standard for audit presentation

### 2.3 Add CI Badge to README

- **Action:** Add build status + coverage badges to `README.md`
- **Shows:** Build passing, coverage %, SonarQube quality gate
- **Effort:** 15 minutes
- **Audit impact:** Medium — visual proof of quality at a glance

---

## P3 — Nice to Have, Strengthens Position

These show engineering maturity but are not audit blockers.

### 3.1 Pre-Commit Hooks (Husky + lint-staged)

- **Action:** Auto-run lint + format on every commit
- **Steps:**
  1. Install `husky` + `lint-staged`
  2. Configure to run ESLint + Prettier on staged files
- **Effort:** 30 minutes
- **Audit impact:** Low-Medium — proves quality is enforced at dev level

### 3.2 Bundle Size Tracking

- **Action:** Track and report bundle size in CI
- **Steps:**
  1. Add `@rsbuild/plugin-bundle-analyzer` or similar
  2. Set size budget thresholds
  3. Fail CI if bundle exceeds limit
- **Effort:** 1 hour
- **Audit impact:** Low — performance awareness signal

### 3.3 Dependency License Audit

- **Action:** Ensure all dependencies have acceptable licenses (MIT, Apache 2.0, etc.)
- **Tool:** `license-checker` or `pnpm licenses list`
- **Effort:** 30 minutes
- **Audit impact:** Low-Medium — legal compliance check

### 3.4 Accessibility Audit Baseline

- **Action:** Run `axe-core` or similar on key pages, document findings
- **Effort:** 1-2 hours
- **Audit impact:** Low — unless accessibility is in scope

---

## Execution Order

| Order | Item                                 | Effort     | Audit Impact |
| ----- | ------------------------------------ | ---------- | ------------ |
| 1     | Fix failing tests (1.1)              | Low        | High         |
| 2     | Fix ESLint errors (1.2)              | Medium     | High         |
| 3     | Fix TypeScript errors (1.3)          | Medium     | High         |
| 4     | Enable lint + typecheck in CI (1.4)  | 5 min      | High         |
| 5     | Fix dependency vulnerabilities (1.5) | Low-Medium | High         |
| 6     | Add test coverage (2.1)              | 1 hr       | Medium-High  |
| 7     | Set up SonarQube via Docker (2.2)    | 1-2 hrs    | Medium-High  |
| 8     | CI badges in README (2.3)            | 15 min     | Medium       |
| 9     | Pre-commit hooks (3.1)               | 30 min     | Low-Medium   |
| 10    | Bundle size tracking (3.2)           | 1 hr       | Low          |
| 11    | License audit (3.3)                  | 30 min     | Low-Medium   |
| 12    | Accessibility baseline (3.4)         | 1-2 hrs    | Low          |

---

## Audit Presentation Plan

When the audit happens, present:

1. **SonarQube dashboard** — live link showing quality gate, ratings, coverage
2. **CI pipeline** — show green checks on recent PRs (lint, typecheck, tests, build)
3. **Coverage report** — HTML or SonarQube view
4. **Dependency audit** — zero high-severity vulnerabilities
5. **Architecture docs** — already written (architecture.md, firebase, redux, theming, code quality)

---

## Target State (audit-ready)

| Metric                            | Current            | Target                                      |
| --------------------------------- | ------------------ | ------------------------------------------- |
| TypeScript errors                 | 60                 | 0                                           |
| ESLint errors                     | 30                 | 0                                           |
| Test pass rate                    | 99.3% (597/601)    | 100%                                        |
| Test coverage                     | Unknown            | 50%+ (measured)                             |
| Dependency vulnerabilities (high) | 10                 | 0                                           |
| CI quality gates                  | Tests + build only | Lint + typecheck + tests + coverage + build |
| SonarQube (Docker)                | Not set up         | Quality gate passing                        |

---

_Ready to start? Pick an item and let's go._

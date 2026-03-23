# Code Quality & Style Guide for Developers

> **Parent:** [Architecture](./architecture.md)  
> **Last Updated:** March 2026

---

## Overview

This guide defines the coding standards, patterns, and best practices for contributing to the AMS Admin Tool. Every developer should read this before writing code. The goal is consistency, type safety, and maintainability across a growing codebase.

---

## TypeScript Rules

### Strict Typing

- **No `any`** without explicit justification in a comment
- Define `interface` for all component props
- Define return types for hooks that return complex objects
- Use discriminated unions over loose optional fields

```typescript
// GOOD - typed props
interface ButtonConfig {
  label: string;
  variant: 'contained' | 'outlined' | 'text';
  onClick: () => void;
}

// BAD - loose typing
interface ButtonConfig {
  label: any;
  variant: string;
  onClick: Function;
}
```

### Import Conventions

- **Type-only imports** use `import type` to avoid runtime overhead:

```typescript
import type { ActionTypeDefinition } from '../../interfaces/actionType.types';
import type { ChangeHistoryEntry } from '../../interfaces/history.types';
```

- **MUI imports** use the specific subpath (not barrel):

```typescript
// CORRECT - specific imports
import Button from '@mui/material/Button';
import TextField from '@mui/material/TextField';

// WRONG - barrel imports (larger bundle)
import { Button, TextField } from '@mui/material';
```

- **Alphabetical order** within import groups:
  1. React
  2. MUI icons
  3. MUI components
  4. Internal atoms/molecules/organisms
  5. Types
  6. Utils/constants

---

## Component Patterns

### File Structure

Every component lives in its own folder:

```
ComponentName/
  ComponentName.tsx          # JSX + props interface
  ComponentName.style.tsx    # Styled components (theme only)
  ComponentName.hook.tsx     # Business logic (if needed)
  ComponentName.stories.tsx  # Storybook (atoms/molecules only)
  __tests__/
    ComponentName.test.tsx   # Unit tests
```

### Component Template

```typescript
import { memo } from 'react';

interface ComponentNameProps {
  title: string;
  onAction: () => void;
}

export const ComponentName = memo(function ComponentName({
  title,
  onAction,
}: ComponentNameProps) {
  const { data, isLoading, error, handleClick } = useComponentName({ onAction });

  if (isLoading) return <Skeleton />;
  if (error) return <ErrorMessage message="Something went wrong" />;
  if (!data) return <EmptyState />;

  return (
    <StyledContainer>
      <Typography>{title}</Typography>
      <Button onClick={handleClick}>Action</Button>
    </StyledContainer>
  );
});
```

**Rules:**
- Wrap in `memo()` with a **named function** (not arrow function) for React DevTools
- Always handle **loading**, **error**, and **empty** states
- Props interface defined in the same file, above the component
- Destructure props in the function signature

### Hook Pattern

```typescript
interface UseComponentNameProps {
  onAction: () => void;
}

interface UseComponentNameReturn {
  data: SomeType | null;
  isLoading: boolean;
  error: string | null;
  handleClick: () => void;
}

export function useComponentName({
  onAction,
}: UseComponentNameProps): UseComponentNameReturn {
  // State
  const [data, setData] = useState<SomeType | null>(null);

  // Callbacks
  const handleClick = useCallback(() => {
    onAction();
  }, [onAction]);

  return { data, isLoading: false, error: null, handleClick };
}
```

**Rules:**
- Define both input props and return type interfaces
- Use `useCallback` for all event handlers
- Use `useMemo` for expensive computations
- Keep hooks focused -- split into multiple hooks if complex

---

## Atomic Design Classification

| Level | Location | Has Logic? | Has Storybook? | Examples |
|---|---|---|---|---|
| **Atoms** | `src/atoms/` | No business logic | Required | ConditionBuilder, DiffViewer, ConfirmationDialog |
| **Molecules** | `src/molecules/` | Minimal logic | Required | HistoryBatchCard, EditorToolbar, NodeContent |
| **Organisms** | `src/organisms/` | Complex hooks | Optional | ScriptEditorPanel, BuilderFlowCanvas, RichTextEditor |
| **Pages** | `src/pages/` | Route-level | No | JourneyEditorPage, ActionBuilderPage |

**When to split:**
- If a component exceeds **400 lines**, extract sub-components
- If a hook exceeds **400 lines**, split into focused sub-hooks
- If a `.style.tsx` exceeds **300 lines**, group related styles

---

## Styling Rules

### The Non-Negotiable Rule

**All styling uses theme values. No exceptions.**

```typescript
// CORRECT
color: theme.palette.text.primary,
padding: theme.spacing(2),
borderRadius: theme.shape.borderRadius,

// FORBIDDEN
color: '#ffffff',
padding: '16px',
borderRadius: '8px',
```

### styled() vs sx Prop

| Use Case | Approach |
|---|---|
| Reusable component with complex styles | `styled()` in `.style.tsx` |
| One-off margin/padding adjustment | `sx` prop |
| Conditional styles based on props | `styled()` with `shouldForwardProp` |
| Quick responsive layout | `sx` prop |

### Styled Component Naming

- Prefix with the purpose: `StyledCard`, `HeaderContainer`, `FilterRow`
- Be descriptive: `EmptyListPlaceholder` not `Placeholder`
- Group related styles in the same `.style.tsx`

---

## State Management Rules

### What Goes Where

| State Type | Technology | Example |
|---|---|---|
| Server data (journeys, actions) | TanStack Query | `useJourneys()`, `useActionTypes()` |
| Global UI (editor, builder, auth) | Redux Toolkit | `editorSlice`, `builderSlice` |
| Persisted preference (theme) | Redux + redux-persist | `themeSlice` |
| Local component UI | `useState` | Dialog open, expanded state |
| Form state | `useState` / hook | Form field values |
| URL-derived state | `useParams` / `useSearchParams` | Journey ID, node ID |

### Anti-Patterns

- **Never** store server data in Redux -- use TanStack Query
- **Never** dispatch in render -- use `useEffect` or callbacks
- **Never** read stale closure state in async operations -- use `store.getState()`
- **Never** mutate state outside Immer (Redux reducers)

---

## Error Handling

### Component Level

Every data-dependent component must handle three states:

```typescript
if (isLoading) return <CircularProgress />;
if (error) return <Alert severity="error">{error.message}</Alert>;
if (!data || data.length === 0) return <EmptyState />;
```

### Async Operations

All async operations use try/catch with user feedback:

```typescript
try {
  await mutation.mutateAsync(data);
  setFeedback({ severity: 'success', message: 'Saved successfully' });
} catch (err) {
  const msg = err instanceof Error ? err.message : 'Unknown error';
  setFeedback({ severity: 'error', message: `Failed: ${msg}` });
}
```

### Non-Blocking Side Effects

History recording, analytics, and other non-critical operations never block the primary action:

```typescript
// Primary operation (must succeed)
await updateJourney(data);

// Side effect (can fail silently)
try {
  await recordHistory(changes);
} catch {
  logger.error('History recording failed (non-blocking)');
}
```

---

## Testing Standards

### Tools

- **Vitest** for test runner
- **Testing Library** for component testing
- **jsdom** for DOM environment

### What to Test

| Priority | What | How |
|---|---|---|
| High | Hooks with business logic | `renderHook` + assertions |
| High | Utility functions | Direct function calls |
| Medium | Component rendering + interaction | `render` + `userEvent` |
| Low | Styled components | Visual regression (Storybook) |

### Test Pattern (AAA)

```typescript
describe('useComponentName', () => {
  it('returns correct initial state', () => {
    // Arrange
    const props = { onAction: vi.fn() };

    // Act
    const { result } = renderHook(() => useComponentName(props));

    // Assert
    expect(result.current.data).toBeNull();
    expect(result.current.isLoading).toBe(false);
  });
});
```

### Mocking

- Mock Firebase services, not TanStack Query hooks
- Mock the logger to avoid console noise:

```typescript
vi.mock('../../../utils/logger', () => ({
  createLogger: () => ({
    forFunction: () => ({
      debug: vi.fn(), info: vi.fn(), warn: vi.fn(), error: vi.fn(),
      apiStart: vi.fn(), apiSuccess: vi.fn(), apiError: vi.fn(),
    }),
  }),
}));
```

### Running Tests

```bash
pnpm test              # Watch mode
pnpm test:run          # Single run
pnpm test:coverage     # Coverage report
pnpm test --run src/utils/changeDetector  # Specific directory
```

---

## Accessibility

### Required for All Interactive Elements

- **ARIA labels** on buttons, inputs, and interactive elements
- **Semantic HTML** -- use `<button>`, `<nav>`, `<main>`, not `<div onClick>`
- **Keyboard support** -- all clickable elements must be focusable and activatable via keyboard
- **Role attributes** on custom interactive elements

```typescript
<IconButton
  onClick={handleDelete}
  aria-label="Delete this item"     // Required
  size="small"
>
  <DeleteIcon />
</IconButton>

<Box
  role="toolbar"                     // Semantic role
  aria-label="Editor toolbar"
>
  {children}
</Box>
```

---

## Naming Conventions

| Item | Convention | Example |
|---|---|---|
| Components | PascalCase | `HistoryBatchCard`, `DynamicPricingEditor` |
| Hooks | camelCase with `use` prefix | `useActionBuilderPage`, `useNodeHistory` |
| Types/Interfaces | PascalCase | `ActionTypeDefinition`, `ChangeHistoryEntry` |
| Constants | UPPER_SNAKE_CASE | `DEFAULT_PAGE_SIZE`, `EDGE_COLORS` |
| Files (component) | PascalCase matching export | `HistoryBatchCard.tsx` |
| Files (utility) | camelCase | `autoLayout.ts`, `pageReferenceUtils.ts` |
| Files (style) | PascalCase + `.style.tsx` | `HistoryBatchCard.style.tsx` |
| Files (hook) | PascalCase + `.hook.tsx` | `ActionBuilderPage.hook.tsx` |
| Files (test) | PascalCase + `.test.ts(x)` | `BuilderFlowCanvas.hook.test.ts` |
| Redux actions | camelCase verbs | `setSelectedNode`, `updateTemplateProps` |
| Redux selectors | camelCase with `select` prefix | `selectUserIdentity`, `selectIsAuthenticated` |

---

## Code Review Checklist

Before submitting code, verify:

```
[ ] Correct folder (atoms/molecules/organisms/pages)
[ ] Required files created (.tsx, .style.tsx, .hook.tsx if needed)
[ ] All styling uses theme values (no hardcoded colors/spacing)
[ ] TypeScript types defined (no any)
[ ] Loading + error + empty states handled
[ ] Accessibility: ARIA labels, semantic HTML, keyboard support
[ ] Files under 400 lines (split if larger)
[ ] Hooks use useCallback/useMemo where appropriate
[ ] No console.log (use logger utility instead)
[ ] Tests written for new hooks/utilities
[ ] Storybook stories for new atoms/molecules
[ ] Imports in correct order and using specific MUI subpaths
```

---

## Logging

Use the structured logger, not `console.log`:

```typescript
import { createLogger } from '../../utils/logger';

const logger = createLogger('ComponentName');

// In a function
const log = logger.forFunction('handleSave');
log.info('Starting save', { journeyId });
log.apiStart('updateJourney', { pageCount: 5 });
log.apiSuccess('updateJourney', { journeyId });
log.apiError('updateJourney', error, { journeyId });
log.warn('Unexpected state', { selectedNodeId });
log.debug('Computed value', { result });
```

**When to log:**
- API calls (start, success, error) -- always
- State transitions -- when debugging complex flows
- Errors -- always with context
- Never log sensitive data (auth tokens, passwords)

---

## Git Conventions

### Commit Messages

Use conventional commits:

```
feat: add pricing section editor to DynamicPricing
fix: prevent double-slash in system navigation paths
refactor: extract shared PathOptionItem component
docs: add history service technical documentation
test: add missing mocks for JourneyBuilderPage hook
```

### Branch Naming

```
feature/pricing-editor
fix/system-path-navigation
refactor/dynamic-template-extraction
docs/architecture
```

---

## Performance Guidelines

- **Memoize** components with `memo()` -- especially list items rendered in loops
- **useCallback** for all event handlers passed as props
- **useMemo** for expensive computations and derived data
- **Lazy load** pages via `React.lazy()`
- **Specific MUI imports** to avoid importing the entire library
- **Avoid re-renders** -- use fine-grained Redux selectors, not whole-slice selections

```typescript
// GOOD - fine-grained selector
const selectedNodeId = useAppSelector((state) => state.editor.selectedNodeId);

// BAD - selects entire slice, re-renders on any change
const editor = useAppSelector((state) => state.editor);
```

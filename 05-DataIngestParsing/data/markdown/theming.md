# Multi-Theming and Styling -- In-Depth Guide

> **Parent:** [Architecture](./architecture.md)  
> **Last Updated:** March 2026

---

## Overview

The AMS Admin Tool supports **6 pre-built themes** (3 light, 3 dark) using MUI's theming system. All styling is done via MUI's `styled()` API and `sx` prop -- **no hardcoded colors, spacing, or font sizes anywhere in the codebase**. The active theme is persisted across sessions via Redux + `redux-persist`.

---

## Theme System Architecture

### How Themes Flow

```
User selects theme in Settings
  -> dispatch(setThemeId('ocean-blue'))
  -> themeSlice updates -> persisted to localStorage
  -> ThemeProvider reads themeId from Redux
  -> createAppThemeFromId('ocean-blue') -> MUI Theme object
  -> <MuiThemeProvider theme={muiTheme}>
  -> All components access via theme.palette.*, theme.spacing(), etc.
```

### Theme Registry

```
src/theme/
  index.ts              # createAppTheme(), createAppThemeFromId()
  palette.ts            # Light/dark base palettes + node colors
  themes.ts             # Re-exports from themes/
  themes/
    types.ts          # ThemeId, ThemeConfig, ThemePalette, ColorScale
    index.ts          # Theme registry + getThemeConfig()
    light/
      defaultLight.ts, sunsetWarm.ts, lavenderMist.ts
    dark/
      defaultDark.ts, oceanBlue.ts, forestGreen.ts
```

---

## Available Themes

| Theme ID | Name | Mode | Description |
|---|---|---|---|
| `default-light` | Default Light | Light | Clean white with blue accents |
| `sunset-warm` | Sunset Warm | Light | Warm oranges and earth tones |
| `lavender-mist` | Lavender Mist | Light | Soft purple accents |
| `default-dark` | Default Dark | Dark | Standard dark with blue accents |
| `ocean-blue` | Ocean Blue | Dark | Deep navy with teal highlights |
| `forest-green` | Forest Green | Dark | Dark with green accents |

---

## Theme Configuration Structure

Every theme follows the `ThemeConfig` interface:

```typescript
interface ThemeConfig {
  id: ThemeId;
  name: string;              // Display name in theme picker
  description: string;
  mode: 'light' | 'dark';
  preview: ThemePreview;     // Colors for theme picker cards
  palette: ThemePalette;     // Full color palette
}

interface ThemePalette {
  primary: ColorScale;       // { main, light, dark, contrastText }
  secondary: ColorScale;
  error: ColorScale;
  warning: ColorScale;
  info: ColorScale;
  success: ColorScale;
  grey: Record<number, string>;
  background: { default: string; paper: string };
  text: { primary: string; secondary: string; disabled: string };
  divider: string;
  sidebar?: SidebarColors;   // Optional sidebar-specific colors
}
```

### Adding a New Theme

- Create a new file in `src/theme/themes/light/` or `src/theme/themes/dark/`
- Export a `ThemeConfig` object with all palette colors
- Export it from the folder's `index.ts`
- Add it to the `themes` record in `src/theme/themes/index.ts`
- Add its ID to the `ThemeId` union type in `types.ts`

---

## MUI Theme Creation

The `createAppThemeFromId()` function in `src/theme/index.ts` takes a `ThemeId` and produces a full MUI `Theme` object with:

**Typography:**
- Font family: Hanken Grotesk (primary), Roboto (fallback)
- Heading scale: h1 (2.5rem) down to h6 (1rem), all weight 600
- Body: body1 (1rem), body2 (0.875rem), caption (0.75rem)
- Buttons: `textTransform: 'none'` (no uppercase)

**Shape:**
- Border radius: 8px globally
- Cards: 12px radius
- Chips: 6px radius

**Spacing:**
- Base unit: 8px (`theme.spacing(1) = 8px`)

**Component Overrides:**
- `MuiButton` -- no box-shadow on contained, 8px radius
- `MuiCard` -- 12px radius, subtle shadow
- `MuiPaper` -- no background image gradient
- `MuiTextField` -- 8px radius on outlined variant
- `MuiOutlinedInput` -- native date picker color scheme matches dark/light mode
- `MuiChip` -- 6px radius
- `MuiMenu` -- dark mode gets border, no gradient
- `MuiTooltip` -- inverted colors for visibility

---

## Styling Patterns

### The Golden Rule

**Never hardcode colors, spacing, sizes, or font values.** Always use theme tokens:

```typescript
// CORRECT
const StyledBox = styled(Box)(({ theme }) => ({
  color: theme.palette.text.primary,
  backgroundColor: theme.palette.background.paper,
  padding: theme.spacing(2),
  borderRadius: theme.shape.borderRadius,
  fontSize: theme.typography.body2.fontSize,
  border: `1px solid ${theme.palette.divider}`,
}));

// WRONG - hardcoded values
const BadBox = styled(Box)({
  color: '#ffffff',           // Use theme.palette.text.primary
  backgroundColor: '#1e1e1e', // Use theme.palette.background.paper
  padding: '16px',            // Use theme.spacing(2)
  borderRadius: '8px',        // Use theme.shape.borderRadius
  fontSize: '14px',           // Use theme.typography.body2.fontSize
});
```

### styled() API

Primary styling method -- creates reusable styled components:

```typescript
import { styled } from '@mui/material/styles';
import Box from '@mui/material/Box';

export const PageContainer = styled(Box)(({ theme }) => ({
  padding: theme.spacing(3),
  maxWidth: 900,
  margin: '0 auto',
}));
```

### Custom Props in styled()

Use `shouldForwardProp` to prevent custom props from being passed to the DOM:

```typescript
interface ChangeTypeChipProps {
  changeType: 'create' | 'update' | 'delete';
}

export const ChangeTypeChip = styled(Box, {
  shouldForwardProp: (prop) => prop !== 'changeType',
})<ChangeTypeChipProps>(({ theme, changeType }) => {
  const colors = {
    create: theme.palette.success.main,
    update: theme.palette.info.main,
    delete: theme.palette.error.main,
  };
  return {
    color: colors[changeType],
    backgroundColor: colors[changeType] + '15',
    padding: theme.spacing(0.25, 0.75),
    borderRadius: theme.shape.borderRadius,
  };
});
```

### sx Prop

For one-off styling that doesn't need a dedicated styled component:

```typescript
<Typography
  variant="caption"
  color="text.secondary"
  sx={{ mb: 1, display: 'block' }}
>
  Results count
</Typography>
```

**When to use which:**
- **styled()** -- reusable components, complex styles, conditional styling via props
- **sx prop** -- one-off adjustments, quick overrides, spacing/margin tweaks

### File Organization

Every component with non-trivial styling has a `.style.tsx` file:

```
ComponentName/
  ComponentName.tsx          # JSX (imports from .style.tsx)
  ComponentName.style.tsx    # All styled components
  ComponentName.hook.tsx     # Logic (no styling)
```

**Rules for .style.tsx files:**
- Max 300 lines
- Only exports styled components
- All values from theme -- no magic numbers
- No business logic

---

## Node Colors (Canvas-Specific)

The canvas uses specialized node colors defined in `src/theme/palette.ts`:

```typescript
export const darkNodeColors = {
  script:    { background: '#0d2137', border: '#1565c0', text: '#64b5f6' },
  form:      { background: '#2d1540', border: '#7b1fa2', text: '#ce93d8' },
  list:      { background: '#0d3321', border: '#2e7d32', text: '#66bb6a' },
  condition: { background: '#3d2800', border: '#ef6c00', text: '#ffb74d' },
  end:       { background: '#3d0a0a', border: '#c62828', text: '#ef5350' },
  journey:   { background: '#003333', border: '#00838f', text: '#4dd0e1' },
  datePicker:{ background: '#3d0d24', border: '#ad1457', text: '#f06292' },
  default:   { background: '#1e1e1e', border: '#424242', text: '#9e9e9e' },
};

export const lightNodeColors = { /* matching light variants */ };
```

Edge colors are defined in `src/utils/constants.ts`:

| Edge Type | Color | Hex |
|---|---|---|
| Default (nextScreen) | Gray-blue | `#64748b` |
| Condition | Amber | `#d97706` |
| List | Green | `#059669` |
| Button | Purple | `#7c3aed` |
| Journey | Cyan | `#0891b2` |
| Selected | Blue | `#3b82f6` |

---

## Dark Mode Implementation

### Toggle Mechanism

The `ThemeProvider` reads from Redux and creates the appropriate MUI theme:

```typescript
function ThemeProvider({ children }) {
  const themeId = useAppSelector((state) => state.theme.themeId);
  const muiTheme = useMemo(() => createAppThemeFromId(themeId), [themeId]);

  return (
    <MuiThemeProvider theme={muiTheme}>
      <CssBaseline />
      {children}
    </MuiThemeProvider>
  );
}
```

### What Adapts Automatically

When switching themes, the following adapt via MUI's `palette.mode`:
- All `theme.palette.*` colors
- `CssBaseline` sets body background + text color
- Native date input icons (filtered via CSS for dark mode)
- Tooltip colors (inverted)
- Menu backgrounds (dark gets border)
- Box shadows (heavier in dark mode)

### What Needs Manual Handling

Some areas use direct color values that MUI can't auto-adapt:
- **Canvas edge colors** -- hardcoded hex values in `EDGE_COLORS` (designed to work in both modes)
- **Node colors** -- separate `darkNodeColors` / `lightNodeColors` palettes
- **CodeMirror editor** -- uses `oneDark` theme in dark mode, default in light
- **Recharts** -- chart colors need explicit dark/light variants

---

## Common Styling Patterns

### Responsive Spacing

```typescript
// Use theme.spacing() with multipliers
padding: theme.spacing(2),           // 16px
gap: theme.spacing(1.5),             // 12px
marginBottom: theme.spacing(0.5),    // 4px
```

### Conditional Dark/Light

```typescript
const StyledCard = styled(Paper)(({ theme }) => ({
  backgroundColor: theme.palette.background.paper,
  boxShadow: theme.palette.mode === 'dark'
    ? '0px 2px 8px rgba(0, 0, 0, 0.3)'
    : '0px 2px 8px rgba(0, 0, 0, 0.08)',
}));
```

### Color with Opacity

```typescript
// Append hex alpha to theme colors
backgroundColor: theme.palette.primary.main + '10',  // 6% opacity
backgroundColor: theme.palette.error.main + '15',     // 8% opacity
```

### Hover States

```typescript
const StyledRow = styled(Box)(({ theme }) => ({
  '&:hover': {
    backgroundColor: theme.palette.action.hover,
  },
}));
```

---

## Known Limitations

- **No per-component theme overrides** -- all components share the same theme; no component-level theme nesting
- **Canvas colors are hardcoded** -- edge and node colors don't adapt to custom themes (only light/dark)
- **No CSS variables** -- theming is runtime JS, not CSS custom properties (limits SSR potential)
- **No user-created themes** -- only the 6 pre-built themes are available

# Redux Implementation — In-Depth Guide

> **Parent:** [Architecture](./architecture.md)  
> **Last Updated:** March 2026

---

## Overview

The AMS Admin Tool uses **Redux Toolkit** for global client-side state management. Redux handles state that is either too complex for local component state or needs to be shared across unrelated component trees — specifically the journey editor, journey builder, authentication, and theme preferences.

Server data (journeys, action types, history) is managed by **TanStack Query**, not Redux. This separation keeps Redux focused on UI orchestration while TanStack Query handles caching, refetching, and server synchronization.

---

## Store Configuration

### Root Store (`src/store/index.ts`)

```typescript
const rootReducer = combineReducers({
  editor: editorReducer,                                    // Journey editing (no persist)
  auth: authReducer,                                        // Authentication (no persist)
  linkEditor: linkEditorReducer,                            // Link editing (no persist)
  builder: persistReducer(builderPersistConfig, builderReducer),  // Builder (persisted)
  theme: persistReducer(themePersistConfig, themeReducer),        // Theme (persisted)
});
```

### Persistence

Two slices are persisted via `redux-persist` with `localStorage`:

| Slice | Persisted Fields | Reason |
|---|---|---|
| `builder` | `journeyData`, `journeyName`, `nodePositions`, `hasUnsavedChanges`, `journeyCategory` | In-progress journey survives page refresh |
| `theme` | `mode`, `themeId` | User's dark/light preference |

`PersistGate` wraps the app to delay rendering until rehydration completes:

```tsx
<Provider store={store}>
  <PersistGate loading={<PageLoader />} persistor={persistor}>
    {children}
  </PersistGate>
</Provider>
```

### Middleware

Redux Toolkit's default middleware is used with a serializable check exception for `redux-persist` action types:

```typescript
middleware: (getDefaultMiddleware) =>
  getDefaultMiddleware({
    serializableCheck: {
      ignoredActions: [FLUSH, REHYDRATE, PAUSE, PERSIST, PURGE, REGISTER],
    },
  }),
```

DevTools are enabled in non-production builds.

---

## Typed Hooks (`src/store/hooks.ts`)

All components use typed hooks instead of raw `useSelector`/`useDispatch`:

```typescript
import { useDispatch, useSelector } from 'react-redux';
import type { RootState, AppDispatch } from './index';

export const useAppDispatch = useDispatch.withTypes<AppDispatch>();
export const useAppSelector = useSelector.withTypes<RootState>();
```

**Rule:** Never import `useSelector` or `useDispatch` directly — always use `useAppSelector` and `useAppDispatch`.

---

## Slice Architecture

### Editor Slice (`editorSlice`)

**Purpose:** Manages state for editing **existing** journeys loaded from Firestore.

**Not persisted** — data is always fresh from Firestore on page load.

```typescript
interface EditorState {
  selectedNodeId: string | null;
  journeyData: JourneyData | null;
  hasUnsavedChanges: boolean;
  isLoading: boolean;
  error: string | null;
}
```

**Actions (30+):**

| Category | Actions | Description |
|---|---|---|
| **Lifecycle** | `setJourneyData`, `resetEditor`, `setLoading`, `setError` | Load/reset journey |
| **Selection** | `setSelectedNode` | Select node for editing |
| **Script** | `updateScript` | Edit script + optional condition expression |
| **Template** | `updateTemplateProps` | Replace entire templateProps JSON |
| **Conditions** | `addCondition`, `deleteCondition`, `updateConditions` | Manage script conditions |
| **Navigation** | `updateNextScreen`, `updateListItemDestination`, `updateDynamicButtonPath`, `updateOptionRedirect`, `updateOptionCondition` | Update page connections |
| **Router** | `updateRouterOptions` | Replace all router options at once |
| **Page CRUD** | `addPage`, `deletePage`, `duplicatePage`, `reorderPages` | Page management |
| **Canvas** | `connectPages` | Create edge from canvas drag |
| **Dynamic List** | `updateDynamicList`, `addDynamicListItem`, `updateDynamicListItem`, `deleteDynamicListItem` | Marketing template lists |
| **Chips** | `addDynamicChip`, `updateDynamicChip`, `deleteDynamicChip` | Marketing chip sections |
| **Save** | `markAsSaved` | Reset unsaved flag |

**Key pattern — `hasUnsavedChanges`:**
Every mutation action sets `state.hasUnsavedChanges = true`. This flag drives:
- The "Unsaved" indicator in the editor toolbar
- The save reminder timer
- The browser unload warning

The `markAsSaved` action resets it after a successful Firestore write.

**Key pattern — cascade on delete:**
`deletePage` clears all references to the deleted page's path across every other page (nextScreen, listItem destinations, button paths, option redirects, journeyProps). This prevents dangling references.

---

### Builder Slice (`builderSlice`)

**Purpose:** Manages state for **creating new** journeys from scratch.

**Persisted** — in-progress journeys survive page refreshes via `redux-persist`.

```typescript
interface BuilderState {
  journeyData: JourneyData | null;
  journeyName: string;
  selectedNodeId: string | null;
  selectedEdgeId: string | null;
  hasUnsavedChanges: boolean;
  isLoading: boolean;
  error: string | null;
  nodePositions: Record<string, { x: number; y: number }>;
  journeyCategory: JourneyCategory;
}
```

**Actions:**

| Category | Actions | Description |
|---|---|---|
| **Lifecycle** | `createJourney`, `loadJourney`, `resetBuilder` | Initialize/load/clear |
| **Node CRUD** | `addNode`, `addNodeWithDefaults`, `deleteNode`, `updateNodeData` | Create/edit/remove pages |
| **Canvas** | `updateNodePosition`, `connectNodes`, `disconnectNodes`, `autoArrangeNodes` | Visual operations |
| **Selection** | `setSelectedNode`, `setSelectedEdge` | Selection state |
| **Rename** | `renamePageKey`, `updatePagePath` | Rename with cascade |
| **Metadata** | `setJourneyName`, `setJourneyCategory` | Journey-level settings |
| **Sync** | `syncJourneyDataFromEditor`, `markAsSaved` | Cross-slice sync |

**Key pattern — dual selection slices:**
When a node is selected on the canvas, both `builderSlice.setSelectedNode` and `editorSlice.setSelectedNode` are dispatched. This allows the `ScriptEditorPanel` (which reads from `editorSlice`) to work seamlessly in both builder and editor contexts.

```typescript
const onNodeClick = (nodeId: string) => {
  dispatch(setBuilderSelectedNode(nodeId));
  dispatch(setEditorSelectedNode(nodeId));
};
```

**Key pattern — `addNodeWithDefaults`:**
Uses `createDefaultPage()` from `constants/templates` to create pages with comprehensive template-specific defaults (e.g., DynamicScript gets conditions, buttons, onLoad/onMount actions). This avoids empty pages that would fail runtime validation.

**Key pattern — cascade on connect/disconnect:**
`connectNodes` inspects the source page type to determine where to write the connection:
- Router pages → `options[].redirect`
- Template pages → `templateProps.nextScreen`
- Journey ref pages → `journeyProps.nextScreen`

`disconnectNodes` reverses the process, clearing all matching paths.

---

### Auth Slice (`authSlice`)

**Purpose:** Stores the authenticated user's identity for use across the app.

```typescript
interface AuthState {
  user: AuthUser | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
}
```

**Actions:** `setUser`, `clearUser`, `setLoading`, `setError`, `setUserRole`

**Selectors:**
```typescript
selectCurrentUser(state)      // Full AuthUser object
selectIsAuthenticated(state)  // boolean
selectAuthLoading(state)      // boolean
selectUserIdentity(state)     // { name, email } for history attribution
```

**Pattern — AuthProvider → Redux sync:**
The `AuthProvider` component subscribes to Firebase Auth state changes and dispatches to Redux:

```typescript
authService.onAuthStateChanged((authUser) => {
  if (authUser) dispatch(setUser(authUser));
  else dispatch(clearUser());
});
```

This dual-source pattern (Context + Redux) allows:
- **Context** for auth actions (signIn, signOut)
- **Redux** for reading auth state in any component without provider coupling

---

### Theme Slice (`themeSlice`)

**Purpose:** Manages dark/light mode and multi-theme selection.

**Persisted** — user's preference survives sessions.

```typescript
interface ThemeState {
  mode: 'light' | 'dark';
  themeId: ThemeId;
}
```

**Actions:** `setThemeMode`, `toggleThemeMode`, `setThemeId`

---

### Link Editor Slice (`linkEditorSlice`)

**Purpose:** Manages state for the navigation link editing workflow.

```typescript
interface LinkEditorState {
  selectedLink: { source: string; target: string } | null;
  isEditing: boolean;
}
```

Lightweight slice used only during active link editing operations.

---

## Data Flow Patterns

### Editor Save Flow

```
User edits in ScriptEditorPanel
  ↓
Local state changes (useRouterOptionsState / useTemplatePropsState)
  ↓
User clicks Save
  ↓
handleSave() in ScriptEditorPanel.hook.tsx
  ↓
Dispatches to editorSlice:
  - updateRouterOptions() or updateTemplateProps()
  ↓
onSaveToFirestore callback fires
  ↓
JourneyEditorPage.hook.tsx handleSave()
  ↓
Reads FRESH state from store (not closure):
  const latestData = store.getState().editor.journeyData
  ↓
detectJourneyChanges(originalData, latestData) → history entries
  ↓
updateJourneyMutation.mutateAsync() → Firestore
  ↓
addHistoryBatch.mutateAsync() → History (non-blocking)
  ↓
dispatch(markAsSaved())
```

**Critical pattern — fresh state from store:**
The `handleSave` function reads state directly from `store.getState()` instead of using the `journeyData` from the closure. This prevents stale data issues when multiple edits happen between renders.

### Builder Publish Flow

```
User clicks Publish
  ↓
Reads latest data from store.getState().editor.journeyData
  ↓
Validates journey (validateJourneyData)
  ↓
createJourneyMutation.mutateAsync() → Firestore
  ↓
buildJourneyCreateHistory() → History (non-blocking)
  ↓
dispatch(resetBuilder())
  ↓
navigate('/editor/:newJourneyId')
```

---

## Anti-Patterns to Avoid

| Anti-Pattern | Why It's Bad | What to Do Instead |
|---|---|---|
| **Storing server data in Redux** | Duplicates TanStack Query cache, goes stale | Use `useQuery()` for server data |
| **Dispatching in render** | Causes infinite re-render loops | Dispatch in callbacks or effects |
| **Reading stale closure state** | Save saves old data | Use `store.getState()` for fresh reads |
| **Putting derived data in state** | Redundant, gets out of sync | Use `useMemo` or `createSelector` |
| **Mutating state outside Immer** | Breaks Redux immutability | Only mutate inside reducer functions |
| **Using `any` in action payloads** | Loses type safety | Define typed payload interfaces |

---

## File Structure

```
src/store/
├── index.ts                    # Store config, persistor, types
├── hooks.ts                    # Typed useAppSelector, useAppDispatch
└── slices/
    ├── editor/
    │   ├── editorSlice.ts      # 30+ actions for journey editing
    │   └── editorTypes.ts      # Action payload interfaces
    ├── builder/
    │   ├── builderSlice.ts     # Canvas + page CRUD actions
    │   └── builderTypes.ts     # Action payload interfaces
    ├── auth/
    │   └── authSlice.ts        # User state + selectors
    ├── linkEditor/
    │   ├── linkEditorSlice.ts  # Link editing state
    │   └── linkEditorTypes.ts
    └── theme/
        └── themeSlice.ts       # Dark/light + multi-theme
```

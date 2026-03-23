# AMS Admin Tool — Architecture Documentation

> **Version:** 1.0  
> **Last Updated:** March 2026  
> **Owner:** Admin Tool Engineering  
> **Status:** Production

---

## Overview

The AMS Admin Tool is a single-page application (SPA) for managing CMS2 journey configurations used by call center agents at All My Sons Moving. It provides a visual journey editor, action builder, change history, variable management, and monitoring — all backed by Firebase Firestore as the data layer.

The application follows **atomic design principles** for component organization, uses **Redux Toolkit** for global state, **TanStack Query** for server data, and **MUI (Material UI v7)** for the design system. It is built with **React 19**, **TypeScript 5.9**, and bundled with **Rsbuild**.

---

## Tech Stack

| Layer | Technology | Version | Purpose |
|---|---|---|---|
| **Framework** | React | 19 | UI rendering |
| **Language** | TypeScript | 5.9 | Type safety |
| **Bundler** | Rsbuild | 1.7 | Build & dev server |
| **UI Library** | MUI (Material UI) | 7 | Component library + theming |
| **Styling** | Emotion (styled) | 11 | CSS-in-JS via MUI's `styled()` |
| **State (global)** | Redux Toolkit | 2.11 | Editor, builder, auth, theme state |
| **State (server)** | TanStack Query | 5.90 | Firestore data fetching + caching |
| **State (persistence)** | redux-persist | 6.0 | Builder drafts + theme across refreshes |
| **Routing** | React Router | 7.12 | SPA routing |
| **Database** | Firebase Firestore | 12.8 | Journey, action type, history, variable storage |
| **Auth** | Firebase Auth (Google) | 12.8 | Google SSO |
| **Canvas** | React Flow (@xyflow/react) | 12.10 | Visual node-and-edge flow graph |
| **Layout** | dagre (@dagrejs/dagre) | 2.0 | Automatic graph layout |
| **Code Editor** | CodeMirror 6 | 6.0 | GraphQL + JSON editing |
| **GraphQL** | graphql.js | 16 | Query parsing + syntax validation |
| **Charts** | Recharts | 3.8 | Dashboard analytics |
| **Drag & Drop** | dnd-kit | 6.3 | Sortable lists (conditions, page order) |
| **Testing** | Vitest + Testing Library | 4.0 | Unit + component tests |
| **Stories** | Storybook | 10.1 | Component documentation |
| **Linting** | ESLint + Prettier | 9.39 | Code quality |

---

## Application Structure

```
src/
├── App.tsx                    # Root: providers, routing, lazy loading
├── index.tsx                  # Entry point
│
├── pages/                     # Route-level components (lazy loaded)
│   ├── LoginPage/             # Google SSO login
│   ├── DashboardPage/         # Analytics dashboard
│   ├── JourneyListPage/       # Journey CRUD list
│   ├── JourneyEditorPage/     # Edit existing journeys (Firestore)
│   ├── JourneyBuilderPage/    # Create new journeys (local → publish)
│   ├── ActionBuilderPage/     # Action type CRUD
│   ├── ChangeHistoryPage/     # Audit trail viewer
│   ├── VariablesPage/         # Context variable management
│   ├── MonitoringPage/        # System monitoring
│   └── ComingSoonPage/        # Placeholder for upcoming features
│
├── organisms/                 # Complex components with business logic
│   ├── ScriptEditorPanel/     # Side panel for editing journey pages
│   ├── BuilderFlowCanvas/     # Visual canvas for journey builder
│   ├── FlowCanvas/            # Visual canvas for journey editor
│   ├── RichTextEditor/        # WYSIWYG script editor
│   ├── NodeHistoryModal/      # Per-node change history
│   ├── RollbackPreviewDialog/ # Rollback confirmation
│   ├── VersionCompareModal/   # Side-by-side version diff
│   ├── AppLayout/             # Main layout (sidebar + content)
│   └── AuthGuard/             # Route protection
│
├── molecules/                 # Composed UI components
│   ├── EditorToolbar/         # Save/close/history toolbar
│   ├── NodeContent/           # Journey node card
│   ├── HistoryBatchCard/      # Grouped history entries
│   ├── HistoryCard/           # Individual history entry
│   ├── HistoryFilters/        # History search + filters
│   ├── ExportMenu/            # CSV/JSON export
│   └── ...
│
├── atoms/                     # Smallest reusable components
│   ├── ConditionBuilder/      # Visual condition expression editor
│   ├── GraphQLCodeEditor/     # CodeMirror-based code editor
│   ├── DiffViewer/            # Before/after diff display
│   ├── ExpressionTextField/   # TextField with {{variable}} validation
│   ├── ConfirmationDialog/    # Reusable confirm/cancel dialog
│   └── ...
│
├── store/                     # Redux state management
│   ├── index.ts               # Store config + persist
│   ├── hooks.ts               # Typed useAppSelector/useAppDispatch
│   └── slices/
│       ├── editor/            # Journey editor state (editing existing)
│       ├── builder/           # Journey builder state (creating new)
│       ├── auth/              # Authentication state
│       ├── linkEditor/        # Navigation link editing
│       └── theme/             # Light/dark mode
│
├── api/hooks/                 # TanStack Query hooks
│   ├── useJourneys.ts         # Journey CRUD
│   ├── useActionTypes.ts      # Action type CRUD
│   ├── useHistory.ts          # Change history queries
│   ├── useVariables.ts        # Variable management
│   └── useScriptEnhance.ts    # AI script enhancement
│
├── firebase/                  # Firestore services
│   ├── index.ts               # Firebase init + compatibility wrapper
│   ├── journeyService.ts      # Journey CRUD operations
│   ├── actionTypeService.ts   # Action type CRUD
│   ├── historyService.ts      # Change history tracking
│   ├── variableService.ts     # Variable management
│   └── monitoring/            # System health monitoring
│
├── providers/                 # React Context providers
│   ├── AuthProvider.tsx        # Firebase Auth + Redux sync
│   ├── ThemeProvider.tsx       # MUI theme (light/dark)
│   ├── VariablesProvider.tsx   # Global variable context
│   └── ActionTypesProvider.tsx # Global action type context
│
├── services/                  # External service integrations
│   ├── auth/                  # Firebase Auth service
│   └── estimator/             # Estimator API integration
│
├── interfaces/                # TypeScript type definitions
│   ├── journey.types.ts       # Journey, page, template types
│   ├── actionType.types.ts    # Action type definitions
│   ├── history.types.ts       # Change history types
│   ├── variable.types.ts      # Variable types
│   ├── auth.types.ts          # Auth user types
│   └── ...
│
├── utils/                     # Shared utilities
│   ├── autoLayout.ts          # Dagre graph layout engine
│   ├── constants.ts           # Edge colors, node types, layout config
│   ├── pageReferenceUtils.ts  # Find/clear page cross-references
│   ├── linkExtractor.ts       # Navigation target extraction
│   ├── changeDetector/        # Deep diff + history entry builder
│   ├── logger.ts              # Structured logging utility
│   └── ...
│
├── theme/                     # MUI theme configuration
│   ├── index.ts               # Theme creation (light + dark)
│   ├── palette.ts             # Color palettes + node colors
│   └── themes/                # Component-level overrides
│
├── constants/                 # Static configuration
│   └── templates/             # Template metadata, defaults, labels
│
└── data/                      # Static data files
    ├── schema.graphql          # Backend GraphQL schema
    └── ...
```

---

## Provider Architecture

The application wraps the entire component tree in a layered provider stack:

```
<Provider store={store}>                    ← Redux store
  <PersistGate persistor={persistor}>       ← Rehydrate persisted state
    <QueryClientProvider>                   ← TanStack Query cache
      <ThemeProvider>                       ← MUI dark/light theme
        <BrowserRouter>                     ← React Router
          <AuthProvider>                    ← Firebase Auth + Redux sync
            <Routes>
              <Route path="/login" />       ← Public
              <Route path="/*">             ← Protected
                <AuthGuard>                 ← Redirect if not authenticated
                  <VariablesProvider>       ← Global variables context
                    <ActionTypesProvider>   ← Global action types context
                      <AppLayout>           ← Sidebar + main content
                        <Suspense>          ← Lazy page loading
                          <Routes />
                        </Suspense>
                      </AppLayout>
                    </ActionTypesProvider>
                  </VariablesProvider>
                </AuthGuard>
              </Route>
            </Routes>
          </AuthProvider>
        </BrowserRouter>
      </ThemeProvider>
    </QueryClientProvider>
  </PersistGate>
</Provider>
```

---

## State Management Strategy

### When to Use What

| State Type | Technology | Examples |
|---|---|---|
| **Server data** | TanStack Query | Journeys, action types, history entries, variables |
| **Global UI** | Redux Toolkit | Editor state, builder state, auth, theme |
| **Persisted local** | redux-persist | Builder drafts, theme preference |
| **Local UI** | useState / useReducer | Form inputs, dialog open/close, expanded state |
| **URL state** | React Router (useParams, useSearchParams) | Journey ID, selected node ID, action type ID |
| **Shared context** | React Context | Variables list, action types list (read-heavy, write-rare) |

### Redux Slices

| Slice | Persisted | Purpose |
|---|---|---|
| `editorSlice` | No | Journey data for editing, selected node, unsaved changes |
| `builderSlice` | Yes (partial) | Journey data for building, node positions, journey name |
| `authSlice` | No | Current user, loading state, auth errors |
| `linkEditorSlice` | No | Navigation link editing state |
| `themeSlice` | Yes | Dark/light mode preference |

### TanStack Query Domains

| Domain | Hook | Cache Config |
|---|---|---|
| Journeys | `useJourneys()`, `useJourney(id)` | 1-min stale, 5-min GC |
| Action Types | `useActionTypes()`, `useActionType(type)` | 1-min stale, 5-min GC |
| History | `useChangeHistory(filters)`, `useNodeHistory(id, key)` | 30-sec stale |
| Variables | `useVariables()` | 1-min stale |

---

## Data Layer (Firebase Firestore)

### Collections by Environment

All data collections are environment-scoped via the `VITE_ENV` variable:

| Data | Collection Pattern | Environments |
|---|---|---|
| Journeys | `journey-{env}` | wip, prewip, uat, prod, preprodflex |
| Action Types | `actiontype-{env}` | wip, prewip, uat, prod, preprodflex |
| Change History | `change-history-{env}` | wip, prewip, uat, prod, preprodflex |
| Variables | `variables-{env}` | wip, prewip, uat, prod, preprodflex |
| Monitoring | `monitoring-{env}` | dev, staging, prod |

### Firestore Compatibility Layer

The `firebase/index.ts` provides a wrapper around the Firebase v9 modular SDK that mimics the v8 chained API pattern (`firestore.collection('x').doc('y').get()`). This keeps service files clean and compatible with the CMS2 backend conventions.

### Service Layer

Each Firestore domain has a dedicated service class:

| Service | File | Responsibilities |
|---|---|---|
| `JourneyService` | `journeyService.ts` | CRUD journeys, page management, Firestore read/write |
| `ActionTypeService` | `actionTypeService.ts` | CRUD action types, bulk create, import, type code generation |
| `HistoryService` | `historyService.ts` | Write change entries (batch), query with filters, rollback |
| `VariableService` | `variableService.ts` | CRUD variables, categories, validation |

---

## Authentication

### Flow

```
User visits app
  → AuthProvider subscribes to Firebase Auth state
  → If not authenticated → AuthGuard redirects to /login
  → User clicks "Sign in with Google"
  → Firebase Auth popup → Google SSO
  → Auth state changes → AuthProvider syncs to Redux (authSlice)
  → AuthGuard allows access to protected routes
```

### User Identity

The authenticated user's identity flows through the app for:
- **Change history attribution** — every save records `user: { name, email }`
- **Rollback attribution** — rollback entries are attributed to the user who triggered them
- **Display** — UserBadge shows avatar + name in history cards

---

## Component Architecture (Atomic Design)

### Hierarchy

```
Pages          → Route-level, lazy loaded, compose organisms
  └── Organisms    → Complex components with hooks and business logic
        └── Molecules   → Composed UI from atoms, some local state
              └── Atoms       → Smallest reusable components, no business logic
```

### File Structure Convention

```
ComponentName/
├── ComponentName.tsx          # JSX + props interface
├── ComponentName.style.tsx    # Styled components (theme-only values)
├── ComponentName.hook.tsx     # Business logic (if complex)
├── ComponentName.stories.tsx  # Storybook (atoms/molecules)
└── __tests__/
    └── ComponentName.test.tsx # Unit tests
```

### Styling Rules

- **All styling uses theme values** — no hardcoded colors, spacing, or sizes
- `theme.palette.*` for colors
- `theme.spacing()` for gaps/padding
- `theme.shape.borderRadius` for corners
- `theme.typography.*` for font sizes
- Styled components via MUI's `styled()` API from `@mui/material/styles`

---

## Routing

### Route Map

| Route | Page | Auth | Description |
|---|---|---|---|
| `/login` | LoginPage | Public | Google SSO login |
| `/` | JourneyListPage | Protected | Journey list with CRUD |
| `/dashboard` | DashboardPage | Protected | Analytics dashboard |
| `/editor/:journeyId/:nodeId?` | JourneyEditorPage | Protected | Edit existing journey |
| `/builder` | JourneyBuilderPage | Protected | Create new journey |
| `/history` | ChangeHistoryPage | Protected | Change audit trail |
| `/variables` | VariablesPage | Protected | Variable management |
| `/action-builder` | ActionBuilderPage | Protected | Action type CRUD |
| `/action-builder/edit/:actionTypeId` | ActionBuilderPage | Protected | Edit specific action type |
| `/monitoring` | MonitoringPage | Protected | System health |
| `/integrations/*` | ComingSoonPage | Protected | Placeholder |
| `/settings` | ComingSoonPage | Protected | Placeholder |

### Code Splitting

All pages are lazy-loaded via `React.lazy()` + `Suspense` with a spinner fallback. This keeps the initial bundle small — only the login page loads immediately.

---

## Key Architectural Patterns

### Hook Composition

Complex pages use a main `.hook.tsx` that composes smaller focused hooks:

```typescript
// ScriptEditorPanel.hook.tsx composes:
const routerOptionsState = useRouterOptionsState({ selectedPage });
const templatePropsState = useTemplatePropsState({ selectedPage });
const linkState = useLinkState({ journeyData });
const dialogState = useDialogState();
const modalState = useModalState();
```

### Non-Blocking Side Effects

All non-critical operations (history recording, analytics) are wrapped in try/catch and never block the primary action:

```typescript
// Save journey first (critical)
await updateJourneyMutation.mutateAsync(data);

// Record history (non-blocking)
try {
  await addHistoryBatch.mutateAsync(changes);
} catch (historyError) {
  logger.error('History failed (non-blocking)', historyError);
}
```

### Deep Diff Change Detection

On every save, the system compares the original data snapshot against the current Redux state to generate granular change entries:

```
originalDataRef.current (snapshot at load)
  vs
store.getState().editor.journeyData (current)
  → detectJourneyChanges() → field-level change entries → Firestore history
```

### Template-Driven Editors

Each CMS2 template type has a dedicated structured editor (extracted into `editors/` subfolder). The main `DynamicTemplateSection` acts as a router:

```typescript
switch (templateName) {
  case 'DynamicPricing': return <DynamicPricingEditor {...props} />;
  case 'DynamicScript': return <DynamicScriptEditor {...props} />;
  // ... 11 template types
}
```

### Environment Isolation

Every data domain uses `VITE_ENV` to determine its Firestore collection. This ensures complete isolation between environments — edits in `wip` never affect `prod`.

---

## Build & Dev

| Command | Purpose |
|---|---|
| `pnpm dev` | Start dev server on port 3001 |
| `pnpm build` | Production build via Rsbuild |
| `pnpm test` | Run Vitest in watch mode |
| `pnpm test:run` | Single test run |
| `pnpm test:coverage` | Coverage report |
| `pnpm storybook` | Storybook on port 6006 |
| `pnpm lint` | ESLint check |
| `pnpm format` | Prettier format |

### Environment Variables

| Variable | Purpose |
|---|---|
| `VITE_ENV` | Environment name (wip, prewip, uat, prod, preprodflex) |
| `VITE_FIREBASE_API_KEY` | Firebase API key |
| `VITE_FIREBASE_AUTH_DOMAIN` | Firebase auth domain |
| `VITE_FIREBASE_PROJECT_ID` | Firebase project ID |
| `VITE_FIREBASE_DATABASE_URL` | Firestore URL |
| `VITE_FIREBASE_STORAGE_BUCKET` | Firebase storage |
| `VITE_FIREBASE_MESSAGING_SENDER_ID` | FCM sender ID |
| `VITE_FIREBASE_APP_ID` | Firebase app ID |

---

## Observability

### Structured Logger

The `createLogger()` utility provides consistent, context-rich logging:

```typescript
const logger = createLogger('JourneyEditorPage');
const log = logger.forFunction('handleSave');

log.apiStart('updateJourney', { journeyId });
log.apiSuccess('updateJourney', { journeyId });
log.apiError('updateJourney', error, { journeyId });
```

Produces structured entries with timestamp, component, function, level, and metadata. Enabled in dev by default, in production via `?debug` URL param or `localStorage.debug=true`.

### Monitoring Service

The `MonitoringPage` provides system health dashboards:
- Firestore read/write counts
- Error rates
- User activity metrics
- Journey edit frequency

---

## Security Considerations

- **Authentication required** — all routes except `/login` are protected by `AuthGuard`
- **Google SSO only** — no password-based auth, leverages corporate Google accounts
- **Environment isolation** — each env has separate Firestore collections
- **No API keys in code** — all Firebase config via environment variables
- **Non-destructive defaults** — deactivation over deletion, confirmation dialogs for destructive actions
- **History attribution** — every change is tied to the authenticated user's name and email

---

## Cross-Cutting Concerns

| Concern | Implementation |
|---|---|
| **Error handling** | Try/catch with user-facing Alert components + logger |
| **Loading states** | Skeleton/spinner components for every async operation |
| **Empty states** | Dedicated empty state components with helpful messaging |
| **Accessibility** | ARIA labels, semantic HTML, keyboard navigation on all interactive elements |
| **Dark mode** | Full dark/light theme via MUI theming, persisted preference |
| **Responsive** | Flex-based layouts, but primarily designed for desktop (admin tool) |
| **Code splitting** | Lazy-loaded pages via React.lazy + Suspense |
| **Type safety** | Strict TypeScript with no `any` (except justified cases) |

---

## Related Documentation

| Document | Path | Covers |
|---|---|---|
| [History Service](./history-service.md) | `docs/history-service.md` | Change tracking, rollback, field labels |
| [Action Builder](./action-builder.md) | `docs/action-builder.md` | Action type management, GraphQL, migration |
| [Journey Canvas](./journey-canvas.md) | `docs/journey-canvas.md` | Visual editor, auto-layout, node connections |

---

## Glossary

| Term | Definition |
|---|---|
| **Journey** | A sequence of CMS2 screens that a call center agent follows during a call |
| **Page** | A single screen within a journey, identified by a pageKey and path |
| **Template** | The screen type (e.g., DynamicScript, DynamicPricing) that determines the UI and editor |
| **templateProps** | JSON configuration for a page — scripts, buttons, list items, actions, conditions |
| **pageOrder** | Array of pageKeys defining the sequence of screens in a journey |
| **Action Type** | A reusable action definition (API call, GraphQL query, remote action) managed in Action Builder |
| **Condition/Router** | A page with `options[]` that routes to different screens based on expressions |
| **Journey Reference** | A page that embeds another journey via `journey` field |
| **Context** | Runtime key-value store available to all screens — populated by actions, read by variables |
| **Variable** | A `{{variableName}}` expression that resolves to a context value at runtime |
| **Atomic Design** | Component hierarchy: atoms → molecules → organisms → pages |
| **editorSlice** | Redux state for editing existing journeys from Firestore |
| **builderSlice** | Redux state for creating new journeys (persisted via redux-persist) |

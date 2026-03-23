# Action Builder — Technical Documentation

> **Version:** 1.0  
> **Last Updated:** March 2026  
> **Owner:** Admin Tool Engineering  
> **Status:** Production

---

## Overview

The Action Builder is the centralized management system for dynamic action definitions used across CMS2 journeys. It serves as the **single source of truth** for all configurable actions — REST API calls, GraphQL operations, and Remote (Twilio) actions — that journey screens can trigger at runtime.

Instead of hardcoding action configurations inside individual journey pages, the Action Builder allows teams to define reusable action types once and reference them by type code across any journey. This decouples action configuration from journey structure, enabling independent updates to API endpoints, payloads, and response mappings without touching journey definitions.

### Key Capabilities

- **Three action kinds** — REST API, GraphQL, and Remote (Twilio postMessage) actions
- **Centralized management** — create, edit, activate/deactivate, and delete action types from a single UI
- **Dynamic integration** — action types automatically appear in the journey editor's action dropdowns
- **GraphQL-aware** — syntax validation, CodeMirror editor, auto-populate response context from query AST
- **Environment-scoped** — each environment has its own Firestore collection
- **Cross-env migration** — export/import all action types as JSON for environment data migration
- **Change history** — all CRUD operations are tracked in the change history service
- **Bulk GraphQL import** — import multiple GraphQL operations from a schema manifest

---

## Motivation & Inspiration

### The Problem

Early CMS2 journeys hardcoded action logic — API URLs, payloads, GraphQL queries, and response mappings — directly inside page `templateProps`. This created several problems:

- **Duplication** — the same API call configured in 10+ journey pages
- **Fragility** — changing an endpoint URL required editing every page that used it
- **No visibility** — no central view of what actions existed or which journeys used them
- **No governance** — anyone could add arbitrary API calls with no review or standardization
- **Tight coupling** — journey structure and API configuration were intertwined

### The Solution

The Action Builder separates **what an action does** (its definition) from **where it's used** (journey pages). Journey editors pick from a dropdown of pre-configured actions instead of manually entering URLs and payloads.

### Design Principles

| Principle | Implementation |
|---|---|
| **Single source of truth** | All custom actions live in Firestore, fetched once and cached |
| **Type-safe** | Discriminated union types (`api \| graphql \| remote`) with kind-specific validation |
| **Non-destructive** | Deactivation instead of deletion as default; delete requires explicit confirmation |
| **Auto-generated codes** | Type codes generated from name + kind prefix (`api_save_lead`, `gql_get_pricing`) |
| **Environment isolation** | Separate Firestore collections per environment via `VITE_ENV` |

---

## Architecture

### How Action Types Flow into Journeys

```
Action Builder Page
  └─ Creates/edits ActionTypeDefinition in Firestore
       └─ Collection: actiontype-{env}

Journey Editor (ScriptEditorPanel)
  └─ useActionTypes() fetches all active action types
       └─ mergeDynamicActionDefinitions() combines:
            ├── 5 built-in actions (updateContext, clearContext, wait, navigate, remote)
            └── All Action Builder types (api, graphql, remote)
                 └─ DynamicActionsEditor renders them in categorized dropdowns
```

### Data Model

```typescript
interface ActionTypeDefinition {
  id: string;                          // Firestore document ID
  type: string;                        // Unique type code (e.g., "gql_get_pricing")
  name: string;                        // Display name (e.g., "Get Pricing Details")
  actionType: ActionTypeKind;          // "api" | "graphql" | "remote"
  isActive: boolean;                   // Soft enable/disable
  description?: string;                // Human-readable description

  // API-specific
  url?: string;                        // Endpoint URL
  method?: ActionHttpMethod;           // GET | POST | PUT | DELETE
  payload?: Record<string, unknown>;   // Request body key-value pairs

  // GraphQL-specific
  query?: string;                      // Full query/mutation string
  variables?: Record<string, unknown>; // Variable definitions (supports nesting)

  // Remote-specific (Twilio)
  remoteType?: string;                 // postMessage type (e.g., "ldTransfer")

  // Shared
  updateApiContext?: Record<string, unknown>;  // Response → context mapping
  createdAt: Date | string;
  updatedAt: Date | string;
}
```

### Firestore Collections

| Environment | Collection Name |
|---|---|
| wip | `actiontype-wip` |
| prewip | `actiontype-prewip` |
| dev | `actiontype-wip` (shares with wip) |
| uat | `actiontype-uat` |
| prod | `actiontype-prod` |
| preprodflex | `actiontype-preprodflex` |

### Type Code Generation

Type codes are auto-generated from the action name and kind:

```
Name: "Get Pricing Details"  +  Kind: graphql  →  gql_get_pricing_details
Name: "Save Lead"            +  Kind: api      →  api_save_lead
Name: "Transfer Call"        +  Kind: remote   →  remote_transfer_call
```

If a code already exists, a numeric suffix is appended (`gql_get_pricing_details_1`).

---

## Action Kinds

### REST API Actions

For standard HTTP API calls to backend services.

| Field | Required | Description |
|---|---|---|
| `url` | Yes | Full endpoint URL |
| `method` | Yes | HTTP method (GET, POST, PUT, DELETE) |
| `payload` | No | Request body as key-value pairs (supports `{{variables}}`) |
| `updateApiContext` | No | Maps response fields into journey context |

**Runtime behavior:** The CMS2 runtime sends an HTTP request with the configured method, URL, and payload. Variables like `{{customerId}}` are resolved from journey context before the request is made. Response values mapped in `updateApiContext` are written back to context.

### GraphQL Actions

For AppSync and other GraphQL API calls.

| Field | Required | Description |
|---|---|---|
| `url` | Yes | GraphQL endpoint URL |
| `query` | Yes | Full query or mutation string (validated with `graphql` parser) |
| `variables` | No | Operation variables (supports nested objects via dot notation) |
| `updateApiContext` | No | Maps response fields into journey context |

**GraphQL-specific features:**
- **Syntax validation** — queries are parsed with the `graphql` library on save; invalid syntax is rejected
- **CodeMirror editor** — full GraphQL editor with syntax highlighting via `GraphQLCodeEditor`
- **Auto-populate response context** — "Auto Populate from Query" button parses the query AST, extracts leaf field paths, and generates `updateApiContext` entries automatically
- **Nested variables** — `NestedKeyValueEditor` supports dot notation for building nested input objects (e.g., `input.customerId`, `input.details.amount`)

**Auto-populate example:**

```graphql
mutation($input: AddBonusInput!) {
  addBonusForDriver(input: $input) {
    status
    msg
  }
}
```

Generates:
```json
{
  "status": "${{response.data.addBonusForDriver.status}}",
  "msg": "${{response.data.addBonusForDriver.msg}}"
}
```

### Remote Actions (Twilio)

For Twilio Flex postMessage-based actions like call transfers, hold, end call.

| Field | Required | Description |
|---|---|---|
| `remoteType` | Yes | The postMessage type (e.g., `ldTransfer`, `endCall`, `holdCall`) |
| `payload` | No | Additional data to send (common fields like `sid`, `DNIS`, `CALLERID` are auto-included from context) |

**Runtime behavior:** The CMS2 runtime sends a `window.postMessage` to the parent Twilio Flex frame with the configured `remoteType` and payload. Twilio Flex listens for these messages and executes the corresponding telephony operation.

---

## Built-in Actions vs Action Builder Actions

The journey editor supports two sources of actions:

### Built-in Actions (Hardcoded)

Always available, cannot be modified from the Action Builder:

| Type | Category | Description |
|---|---|---|
| `updateContext` | Context | Write key-value pairs to journey context |
| `clearContext` | Context | Clear all non-protected context values |
| `wait` | Flow | Delay execution by N milliseconds |
| `navigate` | Flow | Navigate to a specified screen path |
| `remote` | Remote | Generic postMessage to parent window |

### Action Builder Actions (Dynamic)

Created and managed via the Action Builder UI. Appear in the journey editor alongside built-in actions:

- **API actions** → categorized under "API" in the action dropdown
- **GraphQL actions** → categorized under "GraphQL"
- **Remote actions** → categorized under "Remote"

The merge happens at runtime via `mergeDynamicActionDefinitions()`, which combines built-in definitions with all active Action Builder types into a single list for the `DynamicActionsEditor` dropdown.

---

## UI Components

### Action Builder Page (`/action-builder`)

The main management interface with a data table of all action types.

**Features:**
- **Search** — filter by name, type code, description, or URL
- **Filter by kind** — API, GraphQL, Remote
- **Filter by status** — Active, Inactive
- **Sort** — by name (A-Z, Z-A) or last updated
- **Pagination** — configurable page size (default 50)
- **Inline toggle** — activate/deactivate actions directly from the table
- **Edit** — opens full editor dialog via URL routing (`/action-builder/:actionTypeId`)
- **Delete** — confirmation dialog with action name display

### Action Type Editor Dialog

Full-screen dialog for creating or editing an action type.

**Sections by kind:**

| Kind | Sections shown |
|---|---|
| API | Basic Details → API Configuration (URL, Method, Payload, Update API Context) |
| GraphQL | Basic Details → GraphQL Configuration (URL, Query Editor, Variables, Update API Context) |
| Remote | Basic Details → Remote Configuration (Remote Type, Payload) |

**Shared features across all kinds:**
- **Name + Description** — basic identity
- **Active toggle** — enable/disable without deleting
- **Type code preview** — shows the auto-generated type code before saving
- **Validation** — kind-specific validation (URL required for API/GraphQL, query required for GraphQL, remoteType required for Remote)
- **Update API Context** — key-value editor with JSON toggle for mapping response fields to context variables

### Export/Import (Cross-Environment Migration)

**Export:**
- Downloads all action types as a single JSON file
- Filename format: `action-types-{env}-{date}.json`
- Contains full `ActionTypeDefinition` objects with metadata

**Import:**
- Upload a previously exported JSON file
- Overwrites each action type by its `type` key
- Supports api, graphql, and remote action types
- Feedback banner shows count of imported actions

### Bulk GraphQL Import

Specialized bulk import for GraphQL operations from a query manifest.

- Three-phase flow: Select → Importing (progress bar) → Results
- Duplicate detection (skips existing type codes)
- Per-operation status: created, skipped, or error

---

## Integration with Journey Editor

### How Actions Appear in the Editor

When a user adds an onLoad, onMount, or onClick action in any CMS2 dynamic template editor:

```
DynamicActionsEditor
  └─ useActionTypes() → fetches all ActionTypeDefinitions
  └─ mergeDynamicActionDefinitions(actionTypes) → combined list
  └─ Autocomplete grouped by category:
       ├── Context: updateContext, clearContext
       ├── Flow: wait, navigate
       ├── API: api_save_lead, api_get_customer, ...
       ├── GraphQL: gql_get_pricing, gql_add_bonus, ...
       └── Remote: remote, remote_transfer_call, ...
```

### Action Field Rendering

Each action type defines its own fields via `DynamicActionFieldDefinition[]`. The `ActionFieldRenderers` component renders the appropriate input for each field type:

| Input Type | Renders As |
|---|---|
| `text` | TextField with variable suggestion autocomplete |
| `number` | Number input |
| `path` | Path autocomplete with system navigation support |
| `keyValue` | Single key + value text fields |
| `keyValueList` | Key-value map editor with add/delete + JSON toggle |
| `keyValueMap` | Same as keyValueList (alias) |

---

## Firestore Service

### `ActionTypeService` (`src/firebase/actionTypeService.ts`)

| Method | Description |
|---|---|
| `getAllActionTypes()` | Fetches all api/graphql/remote action types, sorted by name |
| `getActionTypeByType(type)` | Fetches a single action type by its type code |
| `createActionType(input)` | Creates with auto-generated unique type code |
| `updateActionType(type, input)` | Updates existing, preserves kind and type code |
| `deleteActionType(type)` | Hard deletes the Firestore document |
| `bulkCreateActionTypes(inputs)` | Creates multiple GraphQL actions with progress callback |
| `importActionType(type, payload)` | Overwrites a document by key (for cross-env migration) |

### TanStack Query Hooks (`src/api/hooks/useActionTypes.ts`)

| Hook | Purpose |
|---|---|
| `useActionTypes()` | Fetch all action types (1-min stale time) |
| `useActionType(type)` | Fetch single action type by code |
| `useCreateActionType()` | Create mutation with cache invalidation |
| `useUpdateActionType()` | Update mutation |
| `useDeleteActionType()` | Delete mutation |
| `useToggleActionTypeActive()` | Quick active/inactive toggle |
| `useBulkCreateActionTypes()` | Bulk create with progress tracking |

---

## File Structure

```
src/
├── interfaces/
│   └── actionType.types.ts                    # Type definitions, filters, sort config
├── firebase/
│   └── actionTypeService.ts                   # Firestore CRUD + import + bulk create
├── api/hooks/
│   └── useActionTypes.ts                      # TanStack Query hooks
├── pages/ActionBuilderPage/
│   ├── ActionBuilderPage.tsx                   # Main page with table, toolbar, dialogs
│   ├── ActionBuilderPage.hook.tsx              # State management, CRUD handlers, export/import
│   ├── ActionBuilderPage.style.tsx             # Styled components
│   ├── ActionBuilderPage.history.ts            # Change history integration
│   └── Layout/
│       ├── ActionTypeEditorDialog.tsx           # Create/edit dialog
│       ├── ActionTypeEditorDialog.hook.tsx      # Editor form state + submission
│       ├── ActionTypeEditorDialog.utils.ts      # Validation, form state, response extraction
│       ├── BasicDetailsSection.tsx              # Name, description, active toggle
│       ├── ApiConfigSection.tsx                 # URL, method, payload, context mapping
│       ├── GraphQLConfigSection.tsx             # Query editor, variables, auto-populate
│       ├── RemoteConfigSection.tsx              # Remote type, payload
│       ├── ActionTypeRecordEditor.tsx           # Key-value editor with JSON toggle
│       ├── ActionBuilderFilters.tsx             # Search, kind, status filters
│       ├── ActionBuilderPagination.tsx          # Page navigation
│       ├── BulkGraphQLImportDialog.tsx          # Bulk import UI
│       ├── BulkGraphQLImportDialog.hook.tsx     # Bulk import state
│       ├── BulkGraphQLImportDialog.style.tsx    # Bulk import styles
│       └── BulkImportResultsSection.tsx         # Import results display
└── organisms/ScriptEditorPanel/
    └── Layout/DynamicTemplateSection/
        ├── actionDefinitions.ts                 # Built-in + merge logic
        ├── DynamicActionsEditor.tsx              # Action list editor in journey pages
        └── ActionFieldRenderers.tsx              # Per-field input components
```

---

## Validation Rules

### API Actions
- Name is required
- URL is required
- HTTP method is required

### GraphQL Actions
- Name is required
- URL is required
- Query is required and must be valid GraphQL syntax (parsed with `graphql` library)

### Remote Actions
- Name is required
- Remote type is required
- Duplicate detection: name + remoteType combination must be unique

### Cross-Kind Rules
- Action kind cannot be changed after creation (enforced by preserving `actionType` on update)
- Type code is immutable after creation
- Deactivated actions remain in Firestore but are excluded from journey editor dropdowns

---

## Change History Integration

All Action Builder operations are tracked in the Change History Service:

| Operation | Tracked Fields |
|---|---|
| **Create** | name, actionType, description, url, method, payload, updateApiContext, query, variables, path, context |
| **Update** | Only fields that changed (field-level diff) |
| **Delete** | All fields recorded as deletions |
| **Toggle active** | isActive field change |

History entries use `journeyId: 'actiontype-library'` and are filtered out of the main journey Change History page to avoid mixing journey and action type changes.

---

## Cross-Environment Migration

### Export Flow
- Click **Export** on the Action Builder page
- Downloads `action-types-{env}-{date}.json`
- Contains all action types with full configuration

### Import Flow
- Click **Import** on the Action Builder page
- Select a previously exported `.json` file
- Each action type is written to Firestore using `doc(type).set()` — **overwrites existing**
- Feedback banner shows number of imported actions

### JSON Format

```json
{
  "exportedAt": "2026-03-16T08:35:31.845Z",
  "environment": "prewip",
  "count": 175,
  "actionTypes": [
    {
      "id": "gql_add_bonus_for_driver",
      "type": "gql_add_bonus_for_driver",
      "name": "Add Bonus For Driver",
      "actionType": "graphql",
      "isActive": true,
      "url": "https://...",
      "query": "mutation($input: AddBonusInput!) { ... }",
      "variables": { "input": { ... } },
      "updateApiContext": { "status": "${{response.data...}}" }
    }
  ]
}
```

> **Note:** The `environment` field is informational only — imports always write to the current environment's Firestore collection regardless of the source environment.

---

## Known Limitations

- **No versioning** — action types are overwritten on update with no built-in undo (rollback is via Change History)
- **No usage tracking** — no way to see which journeys reference a given action type
- **No bulk delete** — actions must be deleted one at a time
- **Sequential bulk create** — bulk GraphQL import processes actions sequentially, not in parallel
- **No schema validation for variables** — variable types are not validated against the GraphQL schema
- **Date strings on import** — `createdAt`/`updatedAt` are stored as ISO strings after cross-env import (not native Firestore Timestamps)

---

## Future Considerations

- **Usage analytics** — show which journeys and pages reference each action type
- **Action type versioning** — maintain version history within the action type itself
- **Parallel bulk operations** — batch Firestore writes for faster bulk import
- **Variable type validation** — validate variables against the GraphQL schema's input types
- **Action testing** — "Test Run" button to execute an action with sample data and see the response
- **Action templates** — pre-built templates for common patterns (save lead, get pricing, transfer call)
- **Deprecation workflow** — mark actions as deprecated with migration guidance before deletion

---

## Glossary

| Term | Definition |
|---|---|
| **Action type** | A reusable action definition stored in Firestore (API call, GraphQL operation, or Remote action) |
| **Type code** | Auto-generated unique identifier (e.g., `gql_get_pricing_details`) used to reference the action in journey pages |
| **Action kind** | The category of action: `api`, `graphql`, or `remote` |
| **Built-in action** | One of the 5 hardcoded actions (updateContext, clearContext, wait, navigate, remote) always available in the editor |
| **updateApiContext** | A key-value map that extracts fields from an API/GraphQL response and writes them into journey context |
| **Remote type** | The `postMessage` type string sent to the Twilio Flex parent frame (e.g., `ldTransfer`, `endCall`) |
| **Variables** | GraphQL operation variables, defined as key-value pairs with support for nested objects via dot notation |
| **Action Builder** | The admin UI page (`/action-builder`) for managing action type definitions |
| **DynamicActionsEditor** | The journey editor component that renders action lists (onLoad, onMount, onClick) using merged action definitions |

# Change History Service — Technical Documentation

> **Version:** 2.0  
> **Last Updated:** March 2026  
> **Owner:** Admin Tool Engineering  
> **Status:** Production

---

## Overview

The Change History Service is a comprehensive audit trail system built into the AMS Admin Tool. It tracks every modification made to journey configurations — who changed what, when, and what the value was before and after the change. It provides full visibility into journey editor operations with rollback capability, enabling the business to **fail safely** and maintain accountability over CMS2 journey configurations.

### Key Capabilities

- **Field-level change tracking** — every individual property change is recorded with before/after values
- **Batch grouping** — changes from a single save operation are grouped together
- **User attribution** — every change is tied to the authenticated user
- **Environment isolation** — each environment (dev, wip, prewip, uat, prod) has its own history collection
- **One-click rollback** — any individual field change or entire batch can be reverted
- **Version comparison** — side-by-side diff viewer for comparing historical versions
- **Category-based filtering** — changes classified into navigation, script, template config, etc.

---

## Motivation & Inspiration

### The Problem

CMS2 journeys are the backbone of the call center agent experience. A single misconfigured screen — a wrong navigation path, a broken condition expression, a missing API context mapping — can disrupt live operations. As the journey editor grew in complexity (11+ dynamic template types, structured action editors, visual condition builders, form field editors), the risk surface expanded significantly.

Without history tracking:

- **No accountability** — impossible to know who changed what
- **No recovery** — a bad edit required manual Firestore intervention to fix
- **No visibility** — business stakeholders had no way to audit what was being modified
- **No safety net** — editors worked without confidence, afraid of breaking live journeys

### The Solution

A production-grade audit trail that:

1. Records changes **automatically** on every save — no manual action needed
2. Shows changes in **human-readable labels** (not raw JSON paths)
3. Allows **instant rollback** to any previous version
4. Provides **business-friendly UI** with search, filters, and batch grouping

### Design Principles

| Principle              | Implementation                                                                                                           |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **Non-blocking**       | History recording failures never prevent saves. All history writes are wrapped in try/catch.                             |
| **Automatic**          | Change detection runs on every save via deep diff — no manual tagging required.                                          |
| **Human-readable**     | 60+ field path patterns produce labels like "Button #1 Text" instead of "templateProps.dynamicButtons[0].btnProps.text"  |
| **Environment-scoped** | Each env has its own Firestore collection (`change-history-{env}`), so history is always relevant to the current context |
| **Scalable**           | Firestore composite indexes, batch writes, value truncation at 10KB                                                      |

---

## Architecture

### Data Flow

```
User edits journey in editor
        ↓
User clicks Save
        ↓
Redux state updated (editorSlice)
        ↓
Firestore journey document updated
        ↓
detectJourneyChanges() runs deep diff:
  old data (originalDataRef) vs new data (Redux store)
        ↓
Change entries created with:
  - fieldPath, fieldLabel, changeType
  - previousValue, newValue (encoded)
  - user, environment, batchId
  - changeCategory (auto-inferred)
        ↓
historyService.addHistoryEntries()
  → Firestore writeBatch (chunks of 500)
        ↓
History available in UI immediately
  (TanStack Query cache invalidation)
```

### Firestore Schema

**Collection:** `change-history-{env}` (e.g., `change-history-wip`, `change-history-prod`)

**Document structure:**

```typescript
{
  // Identity
  batchId: string;          // UUID grouping changes from one save
  journeyId: string;        // Journey being modified
  journeyName: string;      // Display name
  pageKey?: string;         // Specific page within journey

  // Change data
  fieldPath: string;        // e.g., "pages.LeadType.templateProps.dynamicButtons[0].path"
  fieldLabel: string;       // e.g., "Button #1 Path"
  changeType: "create" | "update" | "delete";
  previousValue: string | null;
  newValue: string | null;

  // Attribution
  user: { name: string; email: string };
  environment: string;
  timestamp: Firestore.Timestamp;

  // Metadata
  metadata?: {
    description?: string;
    pagePath?: string;
    changeCategory?: ChangeCategory;
    previousValueEncoding?: "plain" | "json";
    newValueEncoding?: "plain" | "json";
    isTruncated?: boolean;
  }
}
```

### Composite Indexes

Defined in `firestore.indexes.json` for each environment collection:

| Index                                      | Purpose                                          |
| ------------------------------------------ | ------------------------------------------------ |
| `journeyId` + `timestamp DESC`             | Filter history by journey                        |
| `journeyId` + `pageKey` + `timestamp DESC` | Filter history by specific page within a journey |
| `batchId` + `timestamp DESC`               | Fetch all entries in a batch                     |

---

## What We Track

### Journey Page Changes

Every field within a journey page is tracked via deep diff comparison. When a user saves, the system compares the previous journey state with the new state and generates change entries for every modified field.

**Page lifecycle:**

- Page created
- Page deleted
- Page duplicated (tracked as create)

**Page metadata:**

- Template type changed
- Page path changed
- Journey reference changed

**Page order:**

- Pages reordered (drag-and-drop in canvas)

### Template Configuration (CMS2 Dynamic Templates)

All 11 dynamic template types are fully tracked with human-readable labels:

| Template                    | Tracked Fields                                                            |
| --------------------------- | ------------------------------------------------------------------------- |
| **DynamicScript**           | Script, conditions, data field, buttons, onLoad/onMount actions           |
| **DynamicScriptedForm**     | Form fields (type, label, validators, conditions), buttons, next screen   |
| **DynamicScriptedList**     | List items (label, destination, data value, condition, icon), actions     |
| **DynamicScriptedListForm** | Branching condition, true/false form fields, list items                   |
| **DynamicPricing**          | Rebuttal type, show rebuttal, sync legacy visibility, dynamic script mode |
| **DynamicEndCall**          | Buttons, onLoad/onMount actions                                           |
| **DynamicDatePicker**       | Available months, bottom options, buttons                                 |
| **DynamicAddStopForm**      | Address fields, button options                                            |
| **DynamicMixedForm**        | Chip sections, form fields, conditions                                    |
| **DynamicMoveScope**        | Next screen, form fields, buttons                                         |
| **DynamicDeposit**          | onClick actions                                                           |

**Specific field examples:**

- `Button #1 Text` — button label changed
- `Button #1 Display Condition` — conditional visibility updated
- `List Item #2 Destination` — navigation target changed
- `Form Field #3 → Email Validator` — validator rule modified
- `onLoad Action #1 → API Context "status"` — response mapping changed
- `Condition #2 Expression` — routing logic updated

### Router/Condition Pages

- Route redirect paths
- Route condition expressions
- Route additions/deletions

### Navigation Connections

- Next screen connections (templateProps.nextScreen)
- Button navigation paths
- List item destinations
- Canvas drag-connect operations (tracked as nextScreen changes)

### Journey-Level Metadata

- Journey category changes (marketing ↔ cms2)
- Journey creation (publish from builder)
- Journey deletion

### Action Builder Changes

Action type CRUD operations are tracked separately:

- Action type created/updated/deleted
- Tracked fields: name, action kind, description, URL, method, payload, API context, GraphQL query, variables, navigation path

---

## Change Categories

Every change entry is automatically classified into a category for filtering:

| Category           | Description                                                | Example                     |
| ------------------ | ---------------------------------------------------------- | --------------------------- |
| `page-structure`   | Page create, delete, duplicate                             | Created Page "PricingPage"  |
| `navigation`       | Path connections between pages                             | Updated Button #1 Path      |
| `script`           | Script content and condition expressions                   | Updated Condition #2 Script |
| `template-config`  | Template fields, buttons, list items, form fields, actions | Updated Form Field #1 Label |
| `page-order`       | Page reordering                                            | Reordered pages             |
| `journey-metadata` | Journey name, category, settings                           | Updated Journey Category    |
| `action-type`      | Action Builder changes                                     | Updated API URL             |

---

## UI Components

### Change History Page (`/history`)

The dedicated history page with full filtering and pagination.

**Features:**

- **Search** — text search across field labels, page keys, user names, journey names
- **Filters** — Journey, Type (created/updated/deleted), Category, Date Range
- **View modes** — Batched (grouped by save operation) or Individual (flat list)
- **Batch summary** — auto-generated one-liner describing what changed
- **Category badges** — color-coded chips on each entry
- **Infinite scroll** — auto-loads more entries, with explicit "Load More" fallback
- **Export** — download history as CSV or JSON

### Node History Modal

Per-node history accessible from the ScriptEditorPanel via the History button.

**Features:**

- Entries grouped by batch with user/timestamp headers
- Inline diff viewer showing before/after values
- Rollback and compare buttons per entry

### Editor Toolbar — "Last Modified By"

The ScriptEditorPanel toolbar shows who last edited the current node and when:

> _Last edited by John Smith · 5m ago_

### Diff Viewer

Supports three display modes:

- **Inline** — compact `old → new` display
- **Full** — stacked before/after blocks
- **Side-by-side** — two-panel with line-by-line highlighting

Auto-detects JSON values and pretty-prints them for readability.

---

## Rollback System

### Individual Rollback

Any single field change can be rolled back to its previous value:

1. User clicks "Rollback" on a history entry
2. `RollbackPreviewDialog` shows the current value and the value that will be restored
3. On confirm, the system:
   - Reads the current journey from Firestore
   - Navigates to the field path and restores the previous value
   - Saves the updated journey
   - Creates a new history entry documenting the rollback
4. For page-level rollbacks (create → delete, delete → restore), `pageOrder` is automatically synchronized

### Batch Rollback

Available via the "Rollback Batch" button on batch cards. Rolls back all entries in a batch sequentially.

---

## Value Encoding & Storage

### Encoding

Values are stored as strings with an encoding marker:

- **`plain`** — raw string values stored as-is
- **`json`** — non-string values (objects, arrays, booleans, numbers) are JSON-stringified

### Truncation

Values exceeding **10 KB** are truncated with `isTruncated: true` in metadata. This prevents oversized Firestore documents while preserving the most relevant portion of the data.

### Decoding

On read, values are decoded back to their original type using the encoding marker. Backward compatibility is maintained for older entries without encoding metadata via JSON auto-detection.

---

## Performance Characteristics

| Aspect                | Implementation                                                             |
| --------------------- | -------------------------------------------------------------------------- |
| **Write performance** | `writeBatch()` with 500-doc chunks — atomic, parallel writes               |
| **Read performance**  | Firestore composite indexes — `where()` + `orderBy()` instead of full-scan |
| **Non-blocking**      | History failures never prevent journey saves                               |
| **Pagination**        | Cursor-based pagination via TanStack Query `useInfiniteQuery`              |
| **Caching**           | 30-second stale time on history queries, invalidated on write              |
| **Value size**        | 10 KB truncation limit per value                                           |

---

## File Structure

```
src/
├── interfaces/
│   └── history.types.ts              # Type definitions
├── firebase/
│   └── historyService.ts             # Firestore CRUD operations
├── api/hooks/
│   └── useHistory.ts                 # TanStack Query hooks
├── utils/changeDetector/
│   ├── detector.ts                   # Main change detection logic
│   ├── deepDiff.ts                   # Recursive deep diff algorithm
│   ├── entryFactory.ts               # Change entry builder
│   ├── pathFormatter.ts              # 60+ field label patterns + category inference
│   ├── valueCodec.ts                 # Value encoding/decoding + truncation
│   ├── batchSummary.ts               # Batch summary generator
│   └── index.ts                      # Barrel exports
├── pages/
│   └── ChangeHistoryPage/            # Dedicated history page
│       ├── ChangeHistoryPage.tsx
│       ├── ChangeHistoryPage.hook.tsx
│       └── ChangeHistoryPage.style.tsx
├── organisms/
│   ├── NodeHistoryModal/             # Per-node history modal
│   ├── RollbackPreviewDialog/        # Rollback confirmation
│   └── VersionCompareModal/          # Version comparison
├── molecules/
│   ├── HistoryBatchCard/             # Batch display card
│   ├── HistoryCard/                  # Individual entry card
│   └── HistoryFilters/               # Filter controls
└── atoms/
    └── DiffViewer/                   # Before/after diff display
```

---

## Integration Points

### Where History Is Recorded

| Location                | What's Tracked                    | File                                |
| ----------------------- | --------------------------------- | ----------------------------------- |
| Journey Editor save     | All page-level changes + metadata | `JourneyEditorPage.hook.tsx`        |
| Journey Builder publish | Journey creation                  | `JourneyBuilderPage.hook.tsx`       |
| Journey deletion        | Journey deleted                   | `useJourneys.ts` (useDeleteJourney) |
| Action Builder CRUD     | Action type field changes         | `ActionBuilderPage.hook.tsx`        |

### What Is NOT Tracked

| Item                  | Reason                               |
| --------------------- | ------------------------------------ |
| Bulk uploads          | Dev-mode tooling, not user-facing    |
| JSON import/export    | Temporary migration utility          |
| Canvas node positions | Layout-only, no business impact      |
| UI preferences        | Theme, view mode — per-user settings |

---

## Known Limitations

1. **No real-time conflict detection** — if two users edit the same journey simultaneously, the last save wins (no merge strategy)
2. **No full-page snapshots** — only field-level diffs are stored, not complete page state at each version
3. **Action Builder history in same collection** — filtered out in UI but shares the Firestore collection (filtered by `journeyId !== 'actiontype-library'`)
4. **Date fields stored as ISO strings on import** — cross-environment imports store dates as strings instead of native Firestore Timestamps
5. **No retention policy** — history entries accumulate indefinitely (no TTL or cleanup)

---

## Future Considerations

- **Retention policy** — auto-purge entries older than N days to manage Firestore costs
- **Real-time collaboration awareness** — show "User X is editing this page" indicators
- **Full page snapshots** — store complete page state at each save for richer comparison
- **Pre-save diff preview** — show users what they're about to change before confirming save
- **Webhook notifications** — notify Slack/Teams when critical journey pages are modified
- **Separate Action Builder collection** — move action type history to `action-type-history-{env}`

---

## Glossary

| Term                | Definition                                                                          |
| ------------------- | ----------------------------------------------------------------------------------- |
| **Batch**           | A group of change entries created from a single save operation, linked by `batchId` |
| **Field path**      | Dot-notation path to a specific field (e.g., `pages.LeadType.templateProps.script`) |
| **Field label**     | Human-readable name for a field path (e.g., "Script")                               |
| **Change category** | Classification of a change type (navigation, script, template-config, etc.)         |
| **Value encoding**  | How before/after values are stored — `plain` for strings, `json` for objects        |
| **Rollback**        | Restoring a field to its previous value using a history entry                       |
| **Deep diff**       | Recursive comparison of two objects to find all changed fields                      |

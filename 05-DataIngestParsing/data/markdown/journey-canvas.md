# Journey Canvas — Technical Documentation

> **Version:** 1.0  
> **Last Updated:** March 2026  
> **Owner:** Admin Tool Engineering  
> **Status:** Production

---

## Overview

The Journey Canvas is the visual graph editor at the heart of the AMS Admin Tool. It renders CMS2 journey configurations as an interactive node-and-edge flowchart, enabling users to visually build, edit, and understand the flow of call center agent screens. The canvas powers both the **Journey Editor** (editing existing journeys from Firestore) and the **Journey Builder** (creating new journeys from scratch).

Built on **React Flow** (`@xyflow/react`), the canvas provides drag-and-drop node creation, visual edge connections, auto-layout via **dagre**, and real-time synchronization with the Redux store.

### Key Capabilities

- **Visual flow graph** — journey pages rendered as typed, color-coded nodes connected by edges
- **Drag-and-drop creation** — drag template types from a palette onto the canvas to create new screens
- **Edge connections** — draw connections between nodes by dragging from source to target handles
- **Auto-layout** — intelligent hierarchical left-to-right layout powered by dagre
- **Color-coded edges** — edges colored by connection type (navigation, condition, list, button, journey reference)
- **Compact router nodes** — condition/router pages displayed as small pill-shaped nodes
- **Node selection** — clicking a node opens the ScriptEditorPanel for editing
- **Edge deletion** — select and delete connections with one click
- **Node deletion** — cascading reference cleanup when a screen is removed
- **MiniMap** — thumbnail overview with node-type coloring for orientation

---

## Architecture

### Two Canvas Contexts

The canvas operates in two distinct contexts with different data flows:

| Context | Page | Data Source | Save Target | State Slice |
|---|---|---|---|---|
| **Journey Editor** | `/editor/:journeyId` | Firestore via TanStack Query | Firestore (`updateJourney`) | `editorSlice` |
| **Journey Builder** | `/builder` | Redux (persisted via redux-persist) | Firestore (`createJourney` on publish) | `builderSlice` |

Both contexts render the same `BuilderFlowCanvas` component but with different state management:

```
Journey Editor:
  Firestore → useJourney() → Redux editorSlice → FlowCanvas (read-only edges)
                                                 → ScriptEditorPanel (edit props)
                                                 → Save → Firestore + History

Journey Builder:
  Redux builderSlice (persisted) → BuilderFlowCanvas (full CRUD)
                                  → ScriptEditorPanel (edit props)
                                  → Publish → Firestore + History
```

### Data Flow: Journey Data → Nodes & Edges

```
journeyData (Redux)
  └─ buildNodesAndEdges(journeyData, nodePositions)
       ├─ For each pageKey in pageOrder:
       │    ├─ Determine nodeType from template or page structure
       │    ├─ Get position: stored (Redux) → auto-layout (dagre) → fallback (0,0)
       │    └─ Create JourneyNode with data (label, path, nodeType, destinations, script)
       │
       └─ For each node, extract destinations:
            ├─ templateProps.nextScreen → "default" edge
            ├─ templateProps.dynamicButtons[].path → "button" edge
            ├─ templateProps.listItems[].destination → "list" edge
            ├─ options[].redirect → "condition" edge (dashed)
            └─ journeyProps.nextScreen → "journey" edge
```

---

## Node Types

Every journey page maps to a visual node type based on its template or structure:

| Node Type | Visual | Templates |
|---|---|---|
| **script** | Blue card | Script, DynamicScript, PQPrice, ViewDetails, etc. |
| **form** | Purple card | ScriptedForm, DynamicScriptedForm, AddStopForm, DynamicMixedForm, etc. |
| **list** | Green card | ScriptedList, DynamicScriptedList, DynamicPricing, etc. |
| **condition** | Amber pill (compact) | Pages with `options[]` array and no template |
| **end** | Red card | EndCall, DynamicEndCall |
| **journey** | Cyan card | Pages with `journey` field (sub-journey references) |
| **datePicker** | Pink card | DatePicker, DynamicDatePicker |
| **default** | Gray card | Unrecognized templates |

### Standard Node (`BuilderNodeComponent`)

Full-size card (280×130px) with:
- **Left handle** (gray) — target for incoming connections
- **Right handle** (blue) — source for outgoing connections
- **START badge** — shown on the first node in pageOrder (entry point)
- **Delete button** — appears on selection (top-right red circle)
- **NodeContent** — renders page key, path, node type badge, script preview, connection count

### Compact Router Node (`CompactRouterNode`)

Pill-shaped (180×44px) for condition/router pages:
- **Amber handles** — both source and target
- Shows icon + page name + route count (e.g., "MainConditions · 3 routes")
- Saves ~65% vertical space vs standard nodes

---

## Edge System

### Connection Types

Edges are color-coded by how pages are connected:

| Type | Color | Hex | Style | Source Field |
|---|---|---|---|---|
| **default** | Gray-blue | `#64748b` | Solid | `templateProps.nextScreen` |
| **condition** | Amber | `#d97706` | Dashed | `options[].redirect` |
| **list** | Green | `#059669` | Solid | `templateProps.listItems[].destination` |
| **button** | Purple | `#7c3aed` | Solid | `templateProps.dynamicButtons[].path` |
| **journey** | Cyan | `#0891b2` | Solid | `journeyProps.nextScreen` |

### Edge Behavior

- **At rest** — thin (1.5px), 60% opacity, per-type color
- **On hover** — bold (3px), full opacity, glow shadow
- **On select** — bold blue (`#3b82f6`), delete button appears at midpoint
- **Arrow markers** — per-type colored arrowheads via `BuilderArrowMarker`
- **Smooth step routing** — edges use `getSmoothStepPath` with 16px border radius

### Edge Creation

**In Builder mode:**
- Drag from a source handle (right, blue) to a target handle (left, gray)
- `connectNodes` dispatch updates the source page's data:
  - Router pages → fills first empty `options[].redirect` or adds new option
  - Template pages → sets `templateProps.nextScreen`
  - Journey ref pages → sets `journeyProps.nextScreen`

**In Editor mode:**
- Same drag interaction, dispatches `connectPages` to `editorSlice`
- Also checks for existing connections to avoid duplicates

### Edge Deletion

- Click an edge to select it → red delete button appears at midpoint
- Clicking delete dispatches `disconnectNodes`
- Clears the matching path from the source page (nextScreen, option redirect, list item destination, or button path)

---

## Auto-Layout Engine

### Dagre Integration

The auto-layout system uses `@dagrejs/dagre` for hierarchical graph layout:

```typescript
// Configuration
{
  rankdir: 'LR',           // Left-to-right flow
  align: 'UL',             // Upper-left alignment within ranks
  nodesep: 160,            // Vertical gap between nodes in same rank
  ranksep: 400,            // Horizontal gap between ranks (columns)
  edgesep: 80,             // Edge separation to avoid overlap
  ranker: 'network-simplex' // Optimal rank assignment algorithm
}
```

### Weighted Edges

Dagre uses edge weights to determine which connections should be "tight" (short) vs "loose" (can stretch). This produces a natural flow where the main path stays linear and branches spread out:

| Connection Type | Weight | Behavior |
|---|---|---|
| `nextScreen` | 10 | Highest — keeps main linear flow tight |
| `dynamicButtons` | 8 | High — primary navigation stays close |
| Router default (last option) | 8 | High — happy path stays on main line |
| `journeyProps.nextScreen` | 6 | Medium — sub-journey continuation |
| `listItems` destinations | 5 | Medium — user choices can spread |
| Router conditions (non-default) | 2 | Low — branches spread vertically |

### Node Dimensions

| Node Type | Width | Height |
|---|---|---|
| Standard nodes | 280px | 130px |
| Router/condition nodes | 180px | 44px |

### When Auto-Layout Runs

- **Automatically** on first render when no stored positions exist (`nodePositions` is empty)
- **Manually** via the "Auto Layout" toolbar button
- **Never overwrites** manually dragged positions — dagre only applies when no stored position exists for a node

### Position Priority

```
1. Stored position (Redux builderSlice.nodePositions) — user-dragged positions
2. Auto-layout (dagre) — calculated when no stored positions
3. Fallback (0, 0) — should never occur in practice
```

---

## Screen Management

### Adding a New Screen

**Via drag-and-drop (Builder mode):**
- Drag a template type from the component palette onto the canvas
- `onDrop` handler:
  - Parses the dragged template JSON
  - Generates unique page key (e.g., `Script1`, `DynamicScriptedForm2`)
  - Generates path from key (e.g., `script-1`, `dynamic-scripted-form-2`)
  - Dispatches `addNodeWithDefaults` with position, template, key, and path
  - Creates page with comprehensive template defaults from `createDefaultPage()`
  - Auto-selects the new node

**Via "Add Page" dialog (Editor mode):**
- Opens `AddPageDialog` with template picker
- User selects template type, enters page key and path
- Dispatches `addPage` to `editorSlice`
- Can specify `insertAfter` to position in pageOrder

### Deleting a Screen

**Cascading reference cleanup:**

When a screen is deleted, all references to its path are cleared across the entire journey:

```
deleteNode / deletePage dispatched
  └─ For each page in journey:
       ├─ Clear templateProps.nextScreen if it matches deleted path
       ├─ Clear each listItems[].destination that matches
       ├─ Clear each dynamicButtons[].path that matches
       ├─ Clear each options[].redirect that matches
       └─ Clear journeyProps.nextScreen if it matches
  └─ Remove page from pages object
  └─ Remove from pageOrder array
  └─ Remove stored node position
  └─ Clear selection if deleted page was selected
```

**Impact preview (Editor mode):**
Before deletion, `getDeleteImpactSummary()` shows which pages reference the target:
> "Deleting this page will affect: PricingPage (nextScreen), MainConditions (route #1)"

### Duplicating a Screen

- Deep clones the source page
- Appends `-copy` to the path (e.g., `pricing` → `pricing-copy`)
- Inserts after the source in pageOrder
- Preserves all templateProps, options, and settings

### Renaming a Screen

**Builder mode** supports full rename via `renamePageKey`:
- Updates the page key in `pages` object
- Updates `pageOrder` array
- Updates `nodePositions` mapping
- Updates `selectedNodeId` if it was the renamed node
- Cascades path updates to all referencing pages

### Reordering Screens

- `reorderPages` accepts a new `pageOrder` array
- Validates all keys exist, appends any missing keys to the end
- Affects which node gets the "START" badge (always index 0)

---

## Node Interaction

### Selection

- **Click node** → dispatches to both `builderSlice.setSelectedNode` and `editorSlice.setSelectedNode`
- Opens `ScriptEditorPanel` side panel for editing
- Node shows selection ring (blue border + shadow)

### Drag

- Nodes are draggable by default (`draggable: true`)
- Position changes tracked in `builderSlice.nodePositions` via `updateNodePosition`
- Position updates do NOT mark journey as "unsaved" (cosmetic only)

### Edge selection

- **Click edge** → dispatches `setSelectedEdge`
- Clears node selection (mutually exclusive)
- Shows delete button at edge midpoint

---

## State Management

### Builder Slice (`builderSlice`)

Used by Journey Builder for full CRUD operations:

| Action | Description |
|---|---|
| `createJourney` | Initialize empty journey with name |
| `loadJourney` | Load existing journey data + positions |
| `addNode` / `addNodeWithDefaults` | Create new page with defaults |
| `updateNodePosition` | Track drag position |
| `connectNodes` | Create edge between pages |
| `disconnectNodes` | Remove edge between pages |
| `deleteNode` | Remove page + cascade cleanup |
| `updateNodeData` | Update page path/script/nextScreen |
| `renamePageKey` | Rename page key + cascade refs |
| `updatePagePath` | Change page path + cascade refs |
| `autoArrangeNodes` | Apply dagre-computed positions |
| `setSelectedNode` / `setSelectedEdge` | Selection management |
| `resetBuilder` | Clear all state |
| `syncJourneyDataFromEditor` | Sync edited data back |

**Persistence:** Builder state is persisted via `redux-persist` so in-progress journeys survive page refreshes.

### Editor Slice (`editorSlice`)

Used by Journey Editor for editing existing journeys:

| Action | Description |
|---|---|
| `setJourneyData` | Load from Firestore |
| `updateScript` | Edit script content |
| `updateTemplateProps` | Replace templateProps JSON |
| `updateRouterOptions` | Update condition/router options |
| `addPage` / `deletePage` / `duplicatePage` | Page CRUD |
| `reorderPages` | Change page order |
| `connectPages` | Canvas drag-connect |
| `addCondition` / `deleteCondition` | Condition management |
| `updateNextScreen` | Change navigation target |
| `updateListItemDestination` | Change list item target |
| `updateDynamicButtonPath` | Change button navigation |
| `markAsSaved` | Reset unsaved flag |

---

## File Structure

```
src/
├── organisms/BuilderFlowCanvas/
│   ├── BuilderFlowCanvas.tsx              # Main canvas component (ReactFlow wrapper)
│   ├── BuilderFlowCanvas.hook.tsx         # State management, event handlers
│   ├── BuilderFlowCanvas.style.tsx        # Styled components
│   ├── utils/
│   │   └── buildNodesAndEdges.ts          # Journey data → nodes + edges transformer
│   └── Layout/
│       ├── BuilderNodeComponent.tsx        # Standard node renderer
│       ├── CompactRouterNode.tsx           # Compact pill node for conditions
│       ├── BuilderEdge.tsx                 # Custom edge with color-coding + delete
│       ├── BuilderArrowMarker.tsx          # SVG arrow markers per edge type
│       └── index.ts                       # Layout exports
│
├── organisms/FlowCanvas/                  # Editor mode canvas (read + connect)
│   ├── FlowCanvas.tsx
│   ├── FlowCanvas.hook.tsx
│   └── FlowCanvas.style.tsx
│
├── molecules/NodeContent/                 # Shared node card content
│   └── NodeContent.tsx
│
├── utils/
│   ├── autoLayout.ts                      # Dagre auto-layout engine
│   ├── constants.ts                       # Edge colors, layout constants, node type maps
│   ├── pageReferenceUtils.ts              # Find/clear page references for deletion
│   └── types.ts                           # JourneyNode, JourneyEdge, NodeType, etc.
│
├── store/slices/
│   ├── builder/
│   │   ├── builderSlice.ts                # Builder state + all canvas actions
│   │   └── builderTypes.ts                # Action payload types
│   └── editor/
│       ├── editorSlice.ts                 # Editor state + page-level actions
│       └── editorTypes.ts                 # Action payload types
│
├── pages/
│   ├── JourneyBuilderPage/
│   │   ├── JourneyBuilderPage.tsx         # Builder page with toolbar + canvas
│   │   └── JourneyBuilderPage.hook.tsx    # Builder page state (create, publish, import)
│   └── JourneyEditorPage/
│       ├── JourneyEditorPage.tsx           # Editor page with toolbar + canvas
│       ├── JourneyEditorPage.hook.tsx      # Editor page state (load, save, history)
│       └── hooks/
│           ├── usePageManagement.ts        # Add/delete/duplicate/reorder pages
│           └── useSaveReminder.ts          # Auto-save reminder
│
└── constants/templates/
    ├── metadata.ts                         # Template labels, descriptions, icons
    └── defaults/                           # Default templateProps per template type
```

---

## Template Palette (Builder Mode)

The Journey Builder provides a draggable palette of template types. Each template creates a node with pre-configured defaults:

### CMS2 Dynamic Templates

| Template | Node Type | Default Props |
|---|---|---|
| DynamicScript | script | conditions, dataField, dynamicButtons, onLoad/onMount actions |
| DynamicScriptedForm | form | conditions, fields, dynamicButtons, nextScreen |
| DynamicScriptedList | list | conditions, dataField, listItems |
| DynamicPricing | list | nextScreen, rebutalType, dynamicList (pricing sections), dynamicButtons |
| DynamicEndCall | end | dynamicButtons, onLoad/onMount actions |
| DynamicDatePicker | datePicker | dataField, availableMonths, bottomOptions, dynamicButtons |
| DynamicAddStopForm | form | conditions, addressFields, button options |
| DynamicMixedForm | form | conditions, chipSections, nextScreen |
| DynamicMoveScope | form | fields, nextScreen |
| DynamicDeposit | form | onClickActions |
| DynamicScriptedListForm | form | condition, dataField, trueFormFields, falseFormFields, listItems |

### Special Node Types

| Type | Description |
|---|---|
| **Condition** | Router node with `options[]` array — no template, just conditional routing |
| **JourneyReference** | Embeds another journey via `journey` field + `journeyProps.nextScreen` |

---

## Performance Considerations

| Aspect | Implementation |
|---|---|
| **Memoization** | All node/edge components wrapped in `memo()` |
| **Position tracking** | Node positions stored in Redux, not re-derived from dagre on every render |
| **Lazy edge building** | Edges computed only when journey data changes (via `useMemo`) |
| **Node type registration** | Custom node types registered once at component mount |
| **MiniMap colors** | Pre-computed color map, no per-render lookups |
| **Dagre runs once** | Auto-layout only computes when `nodePositions` is empty |

---

## Known Limitations

- **No undo/redo** — canvas operations (delete node, connect, disconnect) have no undo mechanism (rollback is via Change History)
- **No multi-select** — can only select one node or one edge at a time
- **No edge labels on canvas** — edge labels (like "next", "condition") are in the data but not rendered visually
- **Position not persisted to Firestore** — node positions are only in Redux (builder) or local state (editor), not saved to the journey document
- **No zoom-to-fit on load** — canvas starts at default zoom (0.8), user must manually pan/zoom
- **Cross-journey edges not visible** — `./other-journey/path` references create edges to the extracted path but the referenced journey's nodes aren't shown

---

## Future Considerations

- **Undo/redo** — action stack with Ctrl+Z / Ctrl+Y support
- **Multi-select** — select multiple nodes for bulk delete/move
- **Edge labels** — show connection type labels on edges
- **Persist positions to Firestore** — save node layout as part of the journey document
- **Zoom-to-fit on load** — auto-frame all nodes in viewport
- **Minimap interaction** — click minimap to navigate
- **Node grouping** — visual groups for related screens (e.g., "Pricing Flow", "Booking Flow")
- **Real-time collaboration** — show other users' cursors and selections

---

## Glossary

| Term | Definition |
|---|---|
| **Node** | A visual representation of a journey page on the canvas |
| **Edge** | A connection line between two nodes representing navigation flow |
| **Handle** | A connection point on a node (source = right/blue, target = left/gray) |
| **pageKey** | The unique identifier for a page within a journey (e.g., "Pricing", "MainConditions") |
| **path** | The URL segment for a page (e.g., "pricing", "conditions") — used for navigation matching |
| **pageOrder** | Array of pageKeys defining the sequence of pages in the journey |
| **nodePositions** | Map of pageKey → {x, y} coordinates on the canvas |
| **dagre** | Directed graph layout engine used for auto-arranging nodes |
| **rank** | A column in the dagre layout — nodes at the same depth share a rank |
| **connection type** | Classification of an edge: default, condition, list, button, or journey |
| **compact router** | Pill-shaped node for condition/router pages (180×44px vs 280×130px) |
| **entry node** | The first node in pageOrder — marked with a "START" badge |

# Firebase & Firestore — Schema Design & Patterns

> **Parent:** [Architecture](./architecture.md)  
> **Last Updated:** March 2026

---

## Overview

The AMS Admin Tool uses **Firebase Firestore** as its primary database and **Firebase Authentication** for Google SSO. All data is organized into environment-scoped collections, with a compatibility layer wrapping the Firebase v9 modular SDK to provide a clean chainable API for service classes.

---

## Firebase Initialization

### Configuration

Firebase is initialized in `src/firebase/index.ts` using environment variables:

```typescript
const firebaseConfig = {
  apiKey: import.meta.env.VITE_FIREBASE_API_KEY,
  authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN,
  projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID,
  databaseURL: import.meta.env.VITE_FIREBASE_DATABASE_URL,
  storageBucket: import.meta.env.VITE_FIREBASE_STORAGE_BUCKET,
  messagingSenderId: import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID,
  appId: import.meta.env.VITE_FIREBASE_APP_ID,
};
```

### Compatibility Wrapper

The codebase uses a **compatibility wrapper** that provides a v8-style chainable API on top of the v9 modular SDK. This keeps service files clean and consistent with CMS2 backend conventions:

```typescript
// Chainable API (what services use)
firestore.collection('journeys-wip').doc('ccflownew').get()
firestore.collection('journeys-wip').doc('ccflownew').set(data)
firestore.collection('journeys-wip').doc('ccflownew').update(data)
firestore.collection('journeys-wip').doc('ccflownew').delete()
firestore.collection('journeys-wip').get()  // all docs
firestore.collection('journeys-wip').where('field', '==', value).get()

// Raw Firestore instance also available for direct access
import { db } from './firebase';
```

The wrapper supports: `get`, `set`, `update`, `delete`, `add`, `onSnapshot`, `where`, and collection-level `get`.

### Two Firestore Access Patterns

| Pattern | Used By | Import |
|---|---|---|
| **Compatibility wrapper** (`firestore.collection().doc()`) | journeyService, actionTypeService, variableService | `import { firestore } from './index'` |
| **Raw modular SDK** (`collection()`, `doc()`, `getDocs()`) | historyService (needs `writeBatch`, `where`, `orderBy`) | `import { db } from './index'` |

The history service uses the raw SDK because it needs Firestore features not exposed by the wrapper (composite queries, batch writes).

---

## Environment-Scoped Collections

Every data domain maps to a different Firestore collection per environment. The environment is determined by the `VITE_ENV` variable at build/runtime:

### Collection Mapping

| Domain | Collection Pattern | wip | prewip | dev | uat | prod | preprodflex |
|---|---|---|---|---|---|---|---|
| **Journeys** | `journeys-{env}` | journeys-wip | journeys-prewip | journeys-wip | journeys-uat | journeys-prod | journeys-preprodflex |
| **Action Types** | `actiontype-{env}` | actiontype-wip | actiontype-prewip | actiontype-wip | actiontype-uat | actiontype-prod | actiontype-preprodflex |
| **Change History** | `change-history-{env}` | change-history-wip | change-history-prewip | change-history-dev | change-history-uat | change-history-prod | change-history-prod |
| **Variables** | `script-variables` | (shared) | (shared) | (shared) | (shared) | (shared) | (shared) |
| **Monitoring** | `monitoring-{env}` | varies | varies | varies | varies | varies | varies |

> **Note:** `dev` shares the `journeys-wip` and `actiontype-wip` collections. Variables use a single shared collection.

### Resolution Logic

Each service resolves its collection name at instantiation:

```typescript
// journeyService.ts
const collectionConfig = {
  "wip": "journeys-wip",
  "prewip": "journeys-prewip",
  "dev": "journeys-wip",        // shares with wip
  "uat": "journeys-uat",
  "prod": "journeys-prod",
  "preprodflex": "journeys-preprodflex"
};

const collection = collectionConfig[import.meta.env.VITE_ENV] || 'journeys-prewip';
```

---

## Document Schemas

### Journey Document

**Collection:** `journeys-{env}`  
**Document ID:** Journey ID (e.g., `ccflownew`, `quote`, `reachrate2026`)

```typescript
{
  id: "ccflownew",                    // Same as document ID
  name: "CC Flow New",               // Display name
  description: "Main call flow",      // Optional description
  isActive: true,                     // Soft enable/disable

  // Pages — map of pageKey → page config
  pages: {
    "Pricing": {
      path: "pricing",               // URL path segment
      template: "DynamicPricing",    // Template type
      templateProps: {
        nextScreen: "conditions",
        rebutalType: "pricing",
        showRebutal: true,
        dynamicButtons: [
          {
            path: "conditions",
            btnProps: { text: "CONTINUE", variant: "contained", color: "primary" },
            onClickActions: [{ type: "navigate", path: "conditions" }]
          }
        ],
        dynamicList: [
          {
            condition: "true == true",
            chipList: [
              { label: "2 Movers", text: 2, hourlyRate: "{{hourlyRate}}", noOfTrucks: 1 }
            ],
            listItem: [
              { scriptMessage: "The hourly rate is <b>${{hourlyRate}}</b>.", iconName: "Calculate" }
            ]
          }
        ],
        onMountActions: [{ type: "gql_update_lead_status", variables: { ... } }]
      }
    },
    "MainConditions": {
      path: "conditions",
      options: [                      // Router/condition page (no template)
        { condition: "{{storageInterest}} === ''", redirect: "booking" },
        { condition: "true === true", redirect: "self-ams-storage" }
      ]
    },
    "SubJourney": {
      path: "sub-journey",
      journey: "penske-flow",         // Journey reference
      journeyProps: { nextScreen: "booking" }
    }
  },

  // Page order — defines sequence and entry point
  pageOrder: ["Pricing", "MainConditions", "SubJourney", "Booking"],

  // Metadata
  metadata: {
    source: "cms2",                   // "cms2" or "marketing"
    version: 1,
    createdAt: Timestamp,
    updatedAt: Timestamp,
    lastModifiedBy: "user@example.com"
  }
}
```

**Key design decisions:**
- `pages` is a **map** (not array) — enables O(1) lookup by pageKey
- `pageOrder` is a **separate array** — decouples ordering from page data
- `templateProps` is a **flexible JSON object** — different per template type, validated at the editor level
- Router pages have `options[]` but **no template** — distinguishes routing logic from content pages
- Journey references use `journey` + `journeyProps` — embeds another journey by ID

### Action Type Document

**Collection:** `actiontype-{env}`  
**Document ID:** Auto-generated type code (e.g., `gql_get_pricing_details`, `api_save_lead`)

```typescript
{
  type: "gql_get_pricing_details",    // Same as document ID
  name: "Get Pricing Details",        // Display name
  actionType: "graphql",             // "api" | "graphql" | "remote"
  isActive: true,
  description: "Fetches pricing from backend",

  // GraphQL-specific
  url: "https://api.example.com/graphql",
  query: "mutation($input: PricingInput!) { getPricing(input: $input) { rate } }",
  variables: {
    "input": {
      "customerId": "{{customerId}}",
      "moveDate": "{{moveDate}}"
    }
  },

  // API-specific (for actionType: "api")
  method: "POST",
  payload: { "key": "{{value}}" },

  // Remote-specific (for actionType: "remote")
  remoteType: "ldTransfer",

  // Shared
  updateApiContext: {
    "rate": "${{response.data.getPricing.rate}}"
  },

  createdAt: Timestamp,
  updatedAt: Timestamp
}
```

**Key design decisions:**
- **Discriminated by `actionType` field** — schema varies by kind (api/graphql/remote)
- **Type code as document ID** — enables direct lookup without queries
- **`updateApiContext` uses `${{response...}}` syntax** — maps response paths to context variables
- **Variables support nesting** — `input.customerId` creates nested objects at runtime

### Change History Document

**Collection:** `change-history-{env}`  
**Document ID:** Auto-generated (Firestore)

```typescript
{
  batchId: "550e8400-e29b...",        // Groups entries from one save
  journeyId: "ccflownew",
  journeyName: "CC Flow New",
  pageKey: "Pricing",                 // Optional — null for journey-level changes

  fieldPath: "pages.Pricing.templateProps.dynamicButtons[0].btnProps.text",
  fieldLabel: "Button #1 Text",       // Human-readable
  changeType: "update",               // "create" | "update" | "delete"

  previousValue: "NEXT",              // Encoded string
  newValue: "CONTINUE",

  user: { name: "John Smith", email: "john@example.com" },
  environment: "wip",
  timestamp: Timestamp,

  metadata: {
    description: "Updated Button #1 Text",
    pagePath: "pricing",
    changeCategory: "template-config",
    previousValueEncoding: "plain",
    newValueEncoding: "plain",
    isTruncated: false
  }
}
```

**Composite indexes** (defined in `firestore.indexes.json`):
- `journeyId` + `timestamp DESC` — filter by journey
- `journeyId` + `pageKey` + `timestamp DESC` — filter by page within journey
- `batchId` + `timestamp DESC` — fetch entire batch

### Variable Document

**Collection:** `script-variables`  
**Document ID:** Auto-generated (Firestore)

```typescript
{
  name: "customerName",               // Unique, used in {{customerName}} syntax
  label: "Customer Name",
  description: "Full name of the customer",
  category: "customer",               // "customer" | "move" | "quote" | "agent" | "system" | "custom"
  exampleValue: "John Smith",
  isSystem: true,                     // System vars can't be deleted
  isActive: true,
  createdAt: Timestamp,
  updatedAt: Timestamp,
  createdBy: "admin@example.com",
  updatedBy: "admin@example.com"
}
```

---

## Service Layer Pattern

Each Firestore domain follows the same service class pattern:

```typescript
class ServiceName {
  private collection = getCollectionName();  // Resolved at class instantiation

  async getAll(): Promise<Type[]> { ... }
  async getById(id: string): Promise<Type | null> { ... }
  async create(input: CreateInput): Promise<Type> { ... }
  async update(id: string, input: UpdateInput): Promise<Type> { ... }
  async delete(id: string): Promise<boolean> { ... }
}

const serviceName = new ServiceName();  // Singleton
export default serviceName;
```

### Services

| Service | File | Singleton | Methods |
|---|---|---|---|
| `JourneyService` | `journeyService.ts` | `journeyService` | getAll, getById, create, update, delete, subscribe |
| `ActionTypeService` | `actionTypeService.ts` | `actionTypeService` | getAll, getByType, create, update, delete, bulkCreate, import |
| `HistoryService` | `historyService.ts` | `historyService` | addEntry, addEntries (batch), getEntries (filtered), getByJourney, getByPageKey, rollback |
| `VariableService` | `variableService.ts` | `variableService` | getAll, create, update, delete, bulkImport, seed defaults |

### TanStack Query Integration

Services are never called directly from components. Instead, **TanStack Query hooks** wrap service calls:

```
Component → useJourneys() → journeyService.getAllJourneys() → Firestore
                ↓
        TanStack Query cache (staleTime: 60s, gcTime: 5min)
                ↓
        Mutation → journeyService.updateJourney() → Firestore
                ↓
        queryClient.invalidateQueries() → refetch
```

| Hook File | Service | Cache Strategy |
|---|---|---|
| `useJourneys.ts` | journeyService | 1-min stale, 5-min GC |
| `useActionTypes.ts` | actionTypeService | 1-min stale, 5-min GC |
| `useHistory.ts` | historyService | 30-sec stale |
| `useVariables.ts` | variableService | 1-min stale |

---

## Write Patterns

### Single Document Write

```typescript
// Create
await firestore.collection(this.collection).doc(typeCode).set(payload);

// Update
await firestore.collection(this.collection).doc(typeCode).update(payload);

// Delete
await firestore.collection(this.collection).doc(typeCode).delete();
```

### Batch Write (History Service)

The history service uses `writeBatch()` for atomic multi-document writes, chunked into groups of 500 (Firestore limit):

```typescript
const batch = writeBatch(db);
for (const entry of chunk) {
  const docRef = doc(collectionRef);
  batch.set(docRef, { ...entry, timestamp: Timestamp.now() });
}
await batch.commit();
```

### Optimistic Updates

TanStack Query mutations use `onSuccess` to invalidate caches, triggering automatic refetch:

```typescript
useMutation({
  mutationFn: (input) => actionTypeService.createActionType(input),
  onSuccess: () => {
    queryClient.invalidateQueries({ queryKey: actionTypeKeys.all });
  },
});
```

---

## Read Patterns

### Full Collection Read

Used for list pages (journeys, action types, variables):

```typescript
const snapshot = await firestore.collection(this.collection).get();
snapshot.forEach((doc) => {
  results.push(mapToType(doc.id, doc.data()));
});
```

### Single Document Read

Used for detail views and edit pages:

```typescript
const doc = await firestore.collection(this.collection).doc(id).get();
if (doc.exists) return mapToType(doc.id, doc.data());
```

### Indexed Query (History)

Uses composite indexes for efficient filtered reads:

```typescript
const q = query(
  collectionRef,
  where('journeyId', '==', journeyId),
  where('pageKey', '==', pageKey),
  orderBy('timestamp', 'desc'),
  limit(50)
);
const snapshot = await getDocs(q);
```

### Real-time Subscription

Used for monitoring (not for main CRUD):

```typescript
firestore.collection(this.collection).doc(id).onSnapshot(
  (snapshot) => { /* handle update */ },
  (error) => { /* handle error */ }
);
```

---

## Data Migration

### Export/Import (Action Types)

The Action Builder supports JSON export/import for cross-environment migration:

- **Export:** Downloads all action types as `action-types-{env}-{date}.json`
- **Import:** Reads JSON, calls `firestore.doc(type).set(payload)` for each entry — **overwrites existing**

### Fallback Journeys

Marketing journeys have hardcoded fallback data (`src/data/marketingJourneyFallback.ts`) that is auto-persisted to Firestore when the journey doesn't exist yet. This ensures default journeys are always available without manual setup.

---

## Security & Access

| Concern | Implementation |
|---|---|
| **Authentication** | Firebase Auth with Google provider — all API calls require authenticated session |
| **Authorization** | Currently all-or-nothing — authenticated users have full read/write access |
| **Environment isolation** | Separate collections per env — no cross-env data leakage |
| **No Firestore rules** | Security relies on the app layer, not Firestore security rules (admin tool, not public-facing) |
| **Sensitive data** | API keys in env vars, never in code. Firestore config injected at build time |

---

## Known Patterns & Anti-Patterns

### Patterns We Follow

- **Collection per environment** — complete data isolation
- **Document ID as business key** — journey IDs and action type codes are document IDs for O(1) lookup
- **Singleton services** — one instance per service, collection name resolved once
- **Non-blocking history** — history write failures never block primary operations
- **Batch writes for bulk ops** — atomic, faster than sequential

### Known Anti-Patterns / Technical Debt

- **No Firestore security rules** — app-level auth only, should add rules for defense-in-depth
- **Variables use shared collection** — `script-variables` is not environment-scoped (unlike other domains)
- **No pagination on full collection reads** — `getAllJourneys()` fetches every document, won't scale past ~500 journeys
- **Compatibility wrapper overhead** — the v8-style wrapper creates intermediate objects on every call
- **No offline support** — Firestore offline persistence is not configured

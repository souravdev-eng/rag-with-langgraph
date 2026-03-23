# AMS Admin Tool — Technical Documentation

**Last updated:** 2026-03-18
**Audience:** Developers, Architects, Business Stakeholders

---

## Overview

The AMS Admin Tool is a React-based internal platform for managing customer call journeys, agent scripts, and screen configurations. It uses Firebase Firestore as the data layer, Redux Toolkit for UI state, TanStack Query for server state, and MUI with a multi-theme system for styling.

This documentation hub is organized by category. Start with the section most relevant to your role or task.

---

## Architecture

Core technical design, patterns, and infrastructure decisions.

| Document | Description |
|---|---|
| [Architecture Overview](./architecture/overview.md) | Tech stack, app structure, providers, data layer, routing, build system |
| [Firebase & Firestore](./architecture/firebase.md) | Firestore schema, environment-scoped collections, service layer, read/write patterns |
| [Redux Implementation](./architecture/redux.md) | Store config, persistence, slices (editor, builder, auth, theme), data flow |
| [Multi-Theming & Styling](./architecture/theming.md) | Theme registry, MUI theme creation, styled components, dark mode, node colors |
| [Code Quality & Style Guide](./architecture/code-quality.md) | TypeScript rules, component patterns, atomic design, naming, testing standards |

---

## Features

Deep-dive documentation for major product features.

| Document | Description |
|---|---|
| [Journey Builder Canvas](./features/journey-canvas.md) | React Flow canvas, node/edge system, auto-layout, drag-drop, screen management |
| [Action Builder](./features/action-builder.md) | Action type CRUD, GraphQL/REST config, remote actions, variable mapping |
| [Change History Service](./features/history-service.md) | Change detection, batch tracking, field labels, rollback, version comparison |

---

## Operations

Processes, plans, and operational readiness.

| Document | Description |
|---|---|
| [Code Quality & Audit Plan](./operations/code-quality-audit-plan.md) | Prioritized plan (P1/P2/P3) for third-party code audit readiness |

---

## Business

Non-technical documentation for business stakeholders.

| Document | Description |
|---|---|
| *(Planned)* Capabilities SOW | What business can and cannot configure — screen-by-screen breakdown |

---

## Navigation Guide

| I need to... | Read this |
|---|---|
| Understand the tech stack | [Architecture Overview](./architecture/overview.md) |
| Work with Firestore data | [Firebase & Firestore](./architecture/firebase.md) |
| Add or modify Redux state | [Redux Implementation](./architecture/redux.md) |
| Style a component | [Multi-Theming & Styling](./architecture/theming.md) |
| Follow coding standards | [Code Quality & Style Guide](./architecture/code-quality.md) |
| Build on the journey canvas | [Journey Builder Canvas](./features/journey-canvas.md) |
| Configure API actions | [Action Builder](./features/action-builder.md) |
| Understand change tracking | [Change History Service](./features/history-service.md) |
| Prepare for code audit | [Code Quality & Audit Plan](./operations/code-quality-audit-plan.md) |
| Explain capabilities to business | *(Planned)* Capabilities SOW |

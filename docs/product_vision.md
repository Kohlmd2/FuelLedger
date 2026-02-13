# Product Vision: From Internal Tool to Sellable Platform

## Intent
FuelLedger is being developed as a commercial back-office platform for fuel/convenience operators, with a roadmap toward POS-connected operations and eventually an all-in-one system.

## North Star
Build a system that can operate at the same level as modern POS back-office products (e.g., Square/Verifone-compatible workflows), while staying practical for independent stores.

## Product Direction
1. Multi-tenant SaaS-ready architecture (accounts, roles, data isolation, auditability).
2. POS integration-first design (import today, API sync tomorrow).
3. Reliable financial operations (profitability, invoices, loans, inventory, reconciliation).
4. Operator-first UX (fast daily workflow, low training overhead, clear reports).
5. Extensible platform model (new modules without reworking core).

## Integration Strategy (Planned)
1. Start with robust CSV adapters and mapping layers per POS export type.
2. Introduce connector abstraction (`provider`, `auth`, `sync_jobs`, `webhooks`).
3. Add direct API sync for supported POS providers.
4. Normalize all provider data into internal canonical schemas.
5. Maintain fallback import mode for providers with limited API access.

## Architecture Principles Going Forward
1. Keep business logic modular and testable (domain modules over monolith growth).
2. Separate UI, domain logic, and storage/integration concerns.
3. Design schemas for historical consistency and migration safety.
4. Add observability for sync and calculations (logs, status, error surfaces).
5. Prefer backward-compatible changes with explicit migrations.

## Data & Compliance Readiness
1. Treat all merchant data as sensitive by default.
2. Add explicit retention/backups/export controls.
3. Build for least-privilege access and action auditing.
4. Prepare for payment/security compliance requirements as integrations expand.

## Commercial Readiness Milestones
1. Stable core modules: fuel, inside COGS, invoices, inventory, store P&L.
2. Tenant/account model hardened for multiple clients.
3. API connector framework implemented with at least one live provider.
4. Reporting/exports suitable for accountant and owner workflows.
5. Onboarding + support tooling (configuration, mapping, diagnostics).

## Immediate Build Guidance
For all new features, evaluate:
1. Can this work in a multi-client environment?
2. Does this fit a future API-sync workflow?
3. Is this represented in a clean canonical schema?
4. Is it testable and migration-safe?
5. Does it improve daily operator workflow speed and clarity?

---
This document is the active product direction reference for ongoing development decisions.

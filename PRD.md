# Product Requirements Document
## SalesSaathi: AI Co-pilot for Tata Motors Sales & Marketing

**Author:** [Your Name] | **Role (target):** Senior Manager – Product Manager, AI & Innovation
**Version:** 1.0 (Concept Demo) | **Date:** September 2026
**Status:** Prototype for stakeholder review

---

## 1. Background & Problem Statement

Tata Motors has publicly positioned AI as core to its sales and marketing strategy — from hyper-local AI targeting for EV buyers (Punch.ev, Curvv) based on proximity to charging infrastructure, to AI-assisted sales processes aimed at improving operational efficiency. Marketing already runs a "Digital-First, Data-Always" model, with ~45% of budget on digital channels.

However, a gap remains between **lead generation** (which AI/digital marketing already does well) and **lead conversion at the dealer front-end**, where the process is still largely manual:

- Sales executives receive leads via Salesforce Sales Cloud but manually look up variant comparisons, EV range/charging details, and finance schemes for every customer conversation.
- Follow-up messages are generic and slow, especially for leads from digital/social campaigns — a major driver of lead leakage to competitors (Hyundai, Maruti, Mahindra).
- Customer-facing product/finance queries (website, showroom, rural mobile showrooms like "Anubhav") aren't consistently answered with the latest scheme and inventory data.
- Every new AI capability today risks being built as a one-off integration with Salesforce/DMS, increasing technical debt.

**Business impact if unsolved:** slower lead response time, inconsistent product knowledge across dealer network, lower digital-to-showroom conversion, and rising cost per acquired lead as digital spend increases.

---

## 2. Product Vision

> Give every Tata Motors sales executive and digital touchpoint an AI co-pilot that instantly answers product/finance questions, qualifies and follows up on leads in real time, and connects seamlessly into existing CRM/DMS systems — without adding integration overhead for each new AI use case.

This directly supports the AI & Innovation charter: **AI-first roadmap, customer experience, operational efficiency.**

---

## 3. Target Users & Stakeholders

| Stakeholder | Role in this product |
|---|---|
| Dealer Sales Executive | Primary user — uses the co-pilot during and after customer interactions |
| PV/EV Digital Marketing team | Owns campaign data feeding leads; consumer of engagement insights |
| Customer (end consumer) | Indirect user — faster, more accurate responses via chat/showroom kiosk |
| Architecture & Solution team | Technical stakeholder — owns integration standards (this is where MCP fits) |
| CRM/DMS vendor (Salesforce/Siebel, DMS) | External system of record — must integrate, not be replaced |
| TTL / QK / other vendors | Implementation partners for scale-up phase |

---

## 4. Goals & Success Metrics (KPIs)

| Goal | Metric | Target (Phase 1 pilot) |
|---|---|---|
| Faster lead response | Avg. time from lead creation to first meaningful follow-up | Reduce by 40% |
| Higher conversion | Lead-to-test-drive conversion rate | +10–15% relative improvement |
| Consistent product knowledge | % of customer product/finance queries answered correctly without escalation | >90% |
| Adoption | % of dealer sales execs actively using the co-pilot weekly | >70% within 8 weeks of rollout |
| Reduced integration cost | New AI feature integration time (via MCP layer vs. point-to-point) | -50% dev effort per new tool added |

---

## 5. Scope

### In Scope (Phase 1 — Pilot, 1–2 dealerships)
- RAG-based product & finance scheme assistant (internal, sales-exec-facing)
- Agentic lead scoring + auto-drafted follow-up (WhatsApp/email draft, human-approved send)
- MCP tool layer connecting the agent to mock/sandbox CRM, inventory, and calendar systems
- Basic analytics dashboard: response time, query volume, lead score distribution

### Out of Scope (Phase 1)
- Fully autonomous customer-facing chatbot (no human in the loop)
- Direct write-access to production Salesforce/DMS (sandbox/read-first approach)
- Pricing/discount negotiation by AI
- Multi-language voice support (candidate for Phase 3)

---

## 6. User Stories & Acceptance Criteria

**Epic 1: Product & Scheme Knowledge Assistant (RAG)**

- *As a sales executive*, I want to ask a natural-language question comparing two models or trims, so that I can answer customer questions accurately without leaving the conversation.
  - **AC:** Given a query like "Nexon EV vs Creta Electric, budget ₹18L, range priority," the system returns a grounded comparison citing source brochure/spec sheet, with no fabricated specs or prices.
  - **AC:** If the answer isn't in the knowledge base, the system says so rather than guessing (critical for pricing/compliance accuracy).

- *As a sales executive*, I want the assistant to surface the most relevant active finance scheme for a customer's profile, so that I don't miss active offers.
  - **AC:** Scheme results are filtered by region and vehicle model, with an explicit "last updated" date shown.

**Epic 2: Lead Qualification & Follow-Up Agent**

- *As a sales executive*, I want new digital leads automatically scored (hot/warm/cold) with a reason, so that I prioritize my calls.
  - **AC:** Score is generated within 2 minutes of lead creation in CRM; reasoning (e.g. "test-drive requested + EV interest + high-income pincode") is shown, not a black-box number.

- *As a sales executive*, I want a draft follow-up message personalized to the lead's stated interest, so that I can review and send quickly instead of writing from scratch.
  - **AC:** Draft references the correct model, one relevant scheme, and a suggested test-drive slot; requires explicit human approval before sending (no autonomous send in Phase 1).

**Epic 3: MCP Integration Layer**

- *As the Architecture team*, I want a standard tool interface for AI agents to read CRM/inventory/calendar data, so that future AI features don't each require custom integration.
  - **AC:** New "tools" (e.g., check_inventory, get_lead, book_slot) can be added/exposed to any agent without changing the agent's core logic.

---

## 7. Solution Overview (High-Level Architecture)

```
Customer / Sales Exec
        │
        ▼
 [ Co-pilot UI: chat / dashboard ]
        │
        ▼
 ┌───────────────────────────┐
 │   Agent Orchestration      │  ← Agentic AI: lead scoring, follow-up drafting
 └─────────────┬─────────────┘
               │  (calls tools via MCP)
               ▼
 ┌───────────────────────────┐
 │      MCP Tool Layer        │  ← get_lead_from_crm, check_inventory,
 │  (Salesforce / DMS / Cal)  │     schedule_test_drive
 └─────────────┬─────────────┘
               │
               ▼
     CRM (Salesforce Sales Cloud) / DMS / Calendar
               ▲
               │
 ┌───────────────────────────┐
 │   RAG Knowledge Base       │  ← Brochures, spec sheets, finance schemes,
 │  (Vector store + retriever)│     competitor comparisons
 └───────────────────────────┘
```

**Why MCP specifically:** rather than building a bespoke integration each time a new AI feature needs CRM or DMS data, MCP standardizes this as a reusable tool contract — reducing future integration cost, a direct lever on the "APIs and system integrations / microservices architecture" competency this role calls for.

---

## 8. Rollout Plan

| Phase | Scope | Duration |
|---|---|---|
| Phase 1: Pilot | RAG assistant + lead scoring, 1–2 dealerships, sandbox CRM data | 6–8 weeks |
| Phase 2: Controlled expansion | Add auto-follow-up (human-approved), connect to live CRM read-only, expand to 10+ dealerships | 8–12 weeks |
| Phase 3: Scale & autonomy | Customer-facing assistant (showroom kiosk/WhatsApp), broader DMS integration, explore multi-language | Post-pilot, based on KPI results |

---

## 9. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Hallucinated pricing/spec info (compliance/legal exposure) | RAG restricted to approved source documents; "I don't know" fallback; human approval before customer-facing use |
| Data privacy (customer PII in CRM/leads) | Sandbox data for pilot; access via MCP scoped tool permissions, not raw DB access |
| Sales exec adoption resistance | Co-pilot positioned as assistant, not replacement; involve 2–3 sales execs in pilot design (change management) |
| Over-reliance on autonomous agent actions | Human-in-the-loop for all customer-facing sends in Phase 1; autonomy increased only after KPI validation |
| Integration fragility with legacy DMS | MCP layer isolates agent logic from system-specific APIs, easing future migrations |

---

## 10. Open Questions for Stakeholder Discussion

1. Which dealership(s) and region would be best suited for a Phase 1 pilot (data availability, sales team readiness)?
2. Does Architecture & Solution team have a preferred sandbox/test environment for Salesforce Sales Cloud integration?
3. What's the current process for approving customer-facing AI messaging from a brand/compliance standpoint?
4. Are there existing finance scheme data feeds that could be reused as the RAG knowledge source, or would this need a new content pipeline?

---

*This PRD accompanies a working technical prototype demonstrating the RAG assistant, agent-based lead scoring, and MCP tool layer described above.*

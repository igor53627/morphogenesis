---
id: TASK-1.2
title: Gate verifiable PIR (sumcheck/binius) behind feature flag
status: Done
assignee: []
created_date: '2026-02-04 14:00'
updated_date: '2026-02-04 14:05'
labels: []
dependencies: []
parent_task_id: TASK-1
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make verifiable-pir opt-in: sumcheck and binius dependencies gated across morphogen-core/client/server; proof field omitted when feature disabled; tests updated.
<!-- SECTION:DESCRIPTION:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented verifiable-pir feature flag to gate sumcheck/binius across core/client/server and omit proof field when disabled; added mismatch tests for sumcheck.
<!-- SECTION:FINAL_SUMMARY:END -->

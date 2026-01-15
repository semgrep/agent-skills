---
title: Prevent SQL Injection
impact: CRITICAL
impactDescription: Data leakage
tags: sql
---

## Prevent SQL Injection

Never concatenate or use format strings to execute sql, always use parameterized queries.
 
**Incorrect (vulnerable SQL - Node.js / TypeScript):**

```typescript
// Vulnerable: concatenates user input directly into SQL
import { Pool } from 'pg'
const pool = new Pool()

export async function handler(req: any, res: any) {
  const userId = req.query.id // attacker can supply "1 OR 1=1"
  const sql = `SELECT id, username, email FROM users WHERE id = ${userId}`
  const { rows } = await pool.query(sql)
  res.json(rows)
}
```

**Correct (parameterized query - Node.js / TypeScript):**

```typescript
// Safe: use parameterized queries to avoid SQL injection
import { Pool } from 'pg'
const pool = new Pool()

export async function handler(req: any, res: any) {
  const userId = req.query.id
  const sql = 'SELECT id, username, email FROM users WHERE id = $1'
  const { rows } = await pool.query(sql, [userId])
  res.json(rows)
}
```
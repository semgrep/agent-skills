# Code Security

**Version 0.1.0**  
Semgrep Engineering  
January 2026

> **Note:**  
> This document is mainly for agents and LLMs to follow when maintaining,  
> generating, or refactoring codebases with a focus on security best practices. Humans  
> may also find it useful, but guidance here is optimized for automation  
> and consistency by AI-assisted workflows.

---

## Abstract

Comprehensive code security guide, designed for AI agents and LLMs.

---

## Table of Contents

0. [Section 0](#0-section-0) — **CRITICAL**
   - 0.1 [Prevent SQL Injection](#01-prevent-sql-injection)

---

## 0. Section 0

**Impact: CRITICAL**

### 0.1 Prevent SQL Injection

**Impact: CRITICAL (Data leakage)**

Never concatenate or use format strings to execute sql, always use parameterized queries.

**Incorrect:**

```typescript
export async function GET(request: Request) {
  const session = await auth()
  const config = await fetchConfig()
  const data = await fetchData(session.user.id)
  return Response.json({ data, config })
}
```

**Correct:**

```typescript
export async function GET(request: Request) {
  const sessionPromise = auth()
  const configPromise = fetchConfig()
  const session = await sessionPromise
  const [config, data] = await Promise.all([
    configPromise,
    fetchData(session.user.id)
  ])
  return Response.json({ data, config })
}
```

**Incorrect: vulnerable SQL - Node.js / TypeScript**

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

**Correct: parameterized query - Node.js / TypeScript**

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

---


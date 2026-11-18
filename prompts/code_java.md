# Application code review focus

## High-Insight Categories to Prioritize (What to Hunt For)

Focus attention on subtle, high-consequence defects that automated linters and surface-level diff reads miss:

- **Security & Authorization Boundaries**:
  - Missing authentication or authorization checks on new or updated endpoints/methods (e.g., endpoints accessible anonymously by mistake).
  - Insecure Direct Object References (IDOR), privilege escalation, path traversal, injection vulnerabilities (SQL, command, LDAP, template).
  - Exposure of sensitive data (tokens, credentials, PII, internal stack traces) in logs, API responses, or error messages.
- **Transaction Atomicity & Resource Lifecycles**:
  - Premature transaction commits or closes mid-operation that break atomicity with parent operations.
  - Unclosed connections, streams, file descriptors, or sockets (missing `try-finally`, context managers, or `try-with-resources`).
  - Partial state mutation where an error halfway through a multi-step operation leaves data corrupted.
- **Incomplete Multi-Site Refactoring & Dual Caches**:
  - Incomplete invalidation (e.g., user cache invalidated, but role cache forgotten; or entity cache cleared, but query cache retained).
  - Updating a public interface or enum without updating all call sites, exhaustiveness checks, or serializer/deserializer pairs.
  - Hardening or modifying one HTTP verb/endpoint method while leaving adjacent methods unprotected.
- **Data Edge Cases & Type Coercion**:
  - Serialization/deserialization assumptions (e.g., assuming input is always a structured object when scalar numbers/strings or null can arrive).
  - Off-by-one errors, empty collection handling, zero division, or integer overflow/truncation.
  - Unintended None/null dereferences caused by removed guards or changed return types.
- **Concurrency & State Corruption**:
  - Non-thread-safe modifications of shared state or class singletons.
  - Mutable default arguments (e.g., `def fn(items=[])`).
  - Race conditions between checking existence and acting on state (TOCTOU).
- **Silent Failure & Error Handling Antipatterns**:
  - Swallowing exceptions without logging or propagating (`except: pass`, empty catch blocks).
  - Overly broad exception handlers catching control-flow or cancellation signals.
  - Returning misleading success statuses or fallback defaults when operations fail critically.

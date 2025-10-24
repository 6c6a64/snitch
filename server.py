import os
import datetime as dt
from typing import List, Dict, Any, Tuple

import yaml
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ----------------------
# Config
# ----------------------
CONFIG_PATH = os.environ.get("TDASH_CONFIG", "config.yml")
with open(CONFIG_PATH, "r") as f:
    CFG = yaml.safe_load(f)

BASE_URL: str = CFG["taiga"]["base_url"].rstrip("/")
VERIFY_SSL: bool = CFG["taiga"].get("verify_ssl", True)
AUTH_TYPE: str = CFG["taiga"].get("auth_type", "normal")  # normal|token
USERNAME: str = CFG["taiga"].get("username", "")
PASSWORD: str = CFG["taiga"].get("password", "")
API_TOKEN: str = CFG["taiga"].get("api_token", "")

TEAM_USERS: List[Dict[str, Any]] = CFG["team"]["users"]
RAW_PROJECTS: Dict[str, Any] = CFG["projects"]
CUSTOM_TIME_ATTR = CFG.get("custom_time_attr", "Temps passé")
TASK_CUSTOM_TIME_ATTR = CFG.get("task_custom_time_attr", CUSTOM_TIME_ATTR)

DEFAULT_HOURS_PER_DAY = float(CFG.get("expected_hours_per_day", 7))
DEFAULT_DAYS_PER_WEEK = float(CFG.get("expected_days_per_week", 5))

# allow int or {id,time_sources}
def _normalize_projects(raw: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out = {}
    for name, val in raw.items():
        if isinstance(val, int):
            out[name] = {"id": val, "time_sources": ["userstories"]}
        else:
            out[name] = {
                "id": int(val["id"]),
                "time_sources": list(val.get("time_sources", ["userstories"]))
            }
    return out

PROJECTS = _normalize_projects(RAW_PROJECTS)
SESSION = requests.Session()
DEFAULT_TIMEOUT: Tuple[int,int] = (5, 25)

def _get(url, **kw):
    kw.setdefault("timeout", DEFAULT_TIMEOUT)
    print(f"GET {url} with {kw}")
    return SESSION.get(url, **kw)

def _post(url, **kw):
    kw.setdefault("timeout", DEFAULT_TIMEOUT)
    print(f"POST {url} with {kw}")
    return SESSION.post(url, **kw)

# ----------------------
# Auth
# ----------------------
def get_auth_headers() -> Dict[str, str]:
    if AUTH_TYPE == "token":
        return {"Authorization": f"Bearer {API_TOKEN}", "x-disable-pagination": "true"}
    body = {"type": "normal", "username": USERNAME, "password": PASSWORD}
    r = _post(f"{BASE_URL}/auth", json=body, verify=VERIFY_SSL)
    if not r.ok:
        raise HTTPException(status_code=502, detail=f"Taiga auth failed: {r.status_code} {r.text}")
    token = r.json().get("auth_token")
    return {"Authorization": f"Bearer {token}", "x-disable-pagination": "true"}

def headers_for_member(member: Dict[str, Any]) -> Dict[str, str]:
    tok = member.get("token")
    if tok:
        return {"Authorization": f"Bearer {tok}", "x-disable-pagination": "true"}
    return get_auth_headers()

# ----------------------
# Helpers
# ----------------------
def iso(dtobj: dt.datetime) -> str:
    if dtobj.tzinfo is None:
        dtobj = dtobj.replace(tzinfo=dt.timezone.utc)
    return dtobj.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")

def workdays_between(start: dt.date, end: dt.date) -> int:
    days = 0
    cur = start
    while cur <= end:
        if cur.weekday() < 5:
            days += 1
        cur += dt.timedelta(days=1)
    return days

# filter by date within window
def fetch_userstories_window(headers: Dict[str,str], project_id: int,
                             start_iso: str, end_iso: str) -> List[Dict[str, Any]]:
    params = {
        "project": project_id,
        "modified_date__gte": start_iso,
        "modified_date__lte": end_iso,
    }
    r = _get(f"{BASE_URL}/userstories", headers=headers, params=params, verify=VERIFY_SSL)
    if not r.ok:
        raise HTTPException(status_code=502, detail=f"Userstories fetch failed: {r.status_code} {r.text}")
    return [us for us in r.json() if us.get("project") == project_id]

def fetch_tasks_window(headers: Dict[str,str], project_id: int,
                       start_iso: str, end_iso: str) -> List[Dict[str, Any]]:
    params = {
        "project": project_id,
        "modified_date__gte": start_iso,
        "modified_date__lte": end_iso,
    }
    r = _get(f"{BASE_URL}/tasks", headers=headers, params=params, verify=VERIFY_SSL)
    if not r.ok:
        raise HTTPException(status_code=502, detail=f"Tasks fetch failed: {r.status_code} {r.text}")
    return [t for t in r.json() if t.get("project") == project_id]


def fetch_us_history(headers: Dict[str, str], us_id: int) -> List[Dict[str, Any]]:
    r = _get(f"{BASE_URL}/history/userstory/{us_id}", headers=headers, verify=VERIFY_SSL)
    if not r.ok:
        raise HTTPException(status_code=502, detail=f"History fetch failed for US {us_id}: {r.status_code} {r.text}")
    return r.json()

def fetch_task_history(headers: Dict[str, str], task_id: int) -> List[Dict[str, Any]]:
    r = _get(f"{BASE_URL}/history/task/{task_id}", headers=headers, verify=VERIFY_SSL)
    if not r.ok:
        raise HTTPException(status_code=502, detail=f"History fetch failed for Task {task_id}: {r.status_code} {r.text}")
    return r.json()

def extract_time_changes(history: List[Dict[str, Any]], target_username: str,
                         start_dt: dt.datetime, end_dt: dt.datetime,
                         attr_name: str) -> List[Dict[str, Any]]:
    out = []
    for change in history:
        if change.get("user", {}).get("username") != target_username:
            continue
        created_at = change.get("created_at")
        if not created_at:
            continue
        try:
            cdt = dt.datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=dt.timezone.utc)
        except ValueError:
            cdt = dt.datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
        if not (start_dt <= cdt <= end_dt):
            continue

        diff = change.get("values_diff", {})
        ca = diff.get("custom_attributes")
        if not ca:
            continue

        for attr in (ca.get("new") or []):
            if attr.get("name") == attr_name:
                val = str(attr.get("value", "")).replace(",", ".")
                try:
                    out.append({"time": float(val), "date": cdt})
                except ValueError:
                    pass

        for attr in (ca.get("changed") or []):
            if attr.get("name") == attr_name:
                ch = attr.get("changes", {}).get("value", [None, None])
                try:
                    old_v = float(str(ch[0]).replace(",", "."))
                    new_v = float(str(ch[1]).replace(",", "."))
                    out.append({"time": new_v - old_v, "date": cdt})
                except Exception:
                    pass
    return out

# ----------------------
# API
# ----------------------
app = FastAPI(title="Taiga Team Time Dashboard", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)

@app.get("/api/projects")
def list_projects():
    # keys of your normalized PROJECTS dict
    return {"projects": list(PROJECTS.keys())}

@app.get("/api/summary")
def summary(projects: str, start: str, end: str, users: str | None = None) -> Dict[str, Any]:
    try:
        start_date = dt.datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc, hour=0, minute=0, second=0)
        end_date = dt.datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc, hour=23, minute=59, second=59)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format; use YYYY-MM-DD")

    project_names = [p.strip() for p in projects.split(",") if p.strip()]
    proj_defs = []
    for name in project_names:
        if name not in PROJECTS:
            raise HTTPException(status_code=400, detail=f"Unknown project: {name}")
        proj_defs.append({"name": name, **PROJECTS[name]})

    roster = ([u for u in TEAM_USERS if u["username"] in [x.strip() for x in users.split(",") if x.strip()]]
              if users else TEAM_USERS)

    wd = workdays_between(start_date.date(), end_date.date())
    years = range(start_date.year, end_date.year + 1)

    # group by credential set (per-user token vs default) to limit duplicate fetches
    groups: Dict[str, Dict[str, Any]] = {}
    for m in roster:
        key = f"user:{m['username']}" if m.get("token") else "default"
        if key not in groups:
            groups[key] = {"headers": headers_for_member(m) if m.get("token") else get_auth_headers(),
                           "members": []}
        groups[key]["members"].append(m)

    results = []

    for key, grp in groups.items():
        headers = grp["headers"]

        # build sey for this credential set
        us_map: Dict[int, Dict[str, Any]] = {}
        task_map: Dict[int, Dict[str, Any]] = {}

        start_iso = iso(start_date)
        end_iso = iso(end_date)

        for pd in proj_defs:
            pid = pd["id"]
            sources = set(pd.get("time_sources", ["userstories"]))
            for y in years:
                if "userstories" in sources:
                    for us in fetch_userstories_window(headers, pid, start_iso, end_iso):
                        us_map[us["id"]] = us
                if "tasks" in sources:
                    for t in fetch_tasks_window(headers, pid, start_iso, end_iso):
                        task_map[t["id"]] = t

        # fetch histories once per entity
        from concurrent.futures import ThreadPoolExecutor, as_completed

        us_hist: Dict[int, List[Dict[str, Any]]] = {}
        task_hist: Dict[int, List[Dict[str, Any]]] = {}

        def _get_us_hist(i): return fetch_us_history(headers, i)
        def _get_task_hist(i): return fetch_task_history(headers, i)

        with ThreadPoolExecutor(max_workers=12) as ex:
            futs = {ex.submit(_get_us_hist, i): ("us", i) for i in us_map.keys()}
            futs.update({ex.submit(_get_task_hist, i): ("task", i) for i in task_map.keys()})
            for fut in as_completed(futs):
                kind, iid = futs[fut]
                try:
                    if kind == "us":
                        us_hist[iid] = fut.result()
                    else:
                        task_hist[iid] = fut.result()
                except Exception:
                    (us_hist if kind == "us" else task_hist)[iid] = []

        # aggregate per member over both sources
        for member in grp["members"]:
            uname = member["username"]
            display = member.get("display", uname)
            entries = []

            # User Stories
            for us_id, us in us_map.items():
                for c in extract_time_changes(us_hist.get(us_id, []), uname, start_date, end_date, CUSTOM_TIME_ATTR):
                    entries.append({
                        "userstory": us.get("subject", ""),
                        "task": None,
                        "date": c["date"].strftime("%Y-%m-%d"),
                        "hours": round(c["time"], 2),
                        "source": "US",
                    })

            # Tasks
            for t_id, t in task_map.items():
                subj = t.get("subject", "")
                # try to include parent US subject if present in task payload
                parent_us_subject = ""
                if isinstance(t.get("user_story"), dict):
                    parent_us_subject = t["user_story"].get("subject", "")
                for c in extract_time_changes(task_hist.get(t_id, []), uname, start_date, end_date, TASK_CUSTOM_TIME_ATTR):
                    entries.append({
                        "userstory": parent_us_subject,
                        "task": subj,
                        "date": c["date"].strftime("%Y-%m-%d"),
                        "hours": round(c["time"], 2),
                        "source": "TASK",
                    })

            total_hours = round(sum(e["hours"] for e in entries), 2)

            hours_per_day = float(member.get("expected_hours_per_day", DEFAULT_HOURS_PER_DAY))
            days_per_week = float(member.get("expected_days_per_week", DEFAULT_DAYS_PER_WEEK))
            day_scale = (days_per_week / DEFAULT_DAYS_PER_WEEK) if DEFAULT_DAYS_PER_WEEK > 0 else 0.0
            expected = round(wd * day_scale * hours_per_day, 2)

            results.append({
                "username": uname,
                "display": display,
                "total_hours": total_hours,
                "expected_hours": expected,
                "diff": round(total_hours - expected, 2),
                "entries": entries,
            })

    return {
        "range": {"start": start, "end": end, "workdays": wd},
        "projects": [p["name"] for p in proj_defs],
        "custom_time_attr": CUSTOM_TIME_ATTR,
        "task_custom_time_attr": TASK_CUSTOM_TIME_ATTR,
        "results": sorted(results, key=lambda r: r["display"].lower()),
    }

# serve static index.html
app.mount("/", StaticFiles(directory=".", html=True), name="static")

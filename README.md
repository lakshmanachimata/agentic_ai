# Agentic AI (Simple Guide)

This project is a small multi-agent assistant.
You can ask about:

- Weather
- Travel time
- Towns in between a route
- Weather at those towns at arrival time
- Restaurants (in an area, on route, or by towns on route)
- Travel calendar events (`.ics` + Google Calendar add link; optional API push)

## 1) What you need

- Python 3.10+ (you already have Python installed in most cases)
- [Ollama](https://ollama.com/) running locally
- Internet (for free map/weather APIs)

## 2) Setup (copy-paste)

From this project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Start Ollama (if not already running), then pull the model used by the scripts:

```bash
ollama pull qwen3.5:latest
```

## 3) Easiest way to run

### GUI (Streamlit)

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Opens a browser UI. Default agent is the **orchestrator**. Left panel: agent type, **LLM model**, **temperature**, and **top-k**. Ask a question, see the answer, and if a calendar invite was created use **Add to Calendar** to open the Google Calendar URL.

### CLI

Use the unified launcher:

```bash
python run_agents.py
```

You will get a prompt like `>`.
Type your questions there.

To clear chat memory in interactive mode, type:

```text
/reset
```

## 4) Run specific agents

### Orchestrator (auto-routes your query)

```bash
python run_agents.py --client orchestrator
```

### Travel only

```bash
python run_agents.py --client travel
```

### Weather only

```bash
python run_agents.py --client weather
```

### Restaurants only

```bash
python run_agents.py --client restaurants
```

### Calendar only

```bash
python run_agents.py --client calendar
```

## 5) Example questions

### Travel + intermediate towns + weather at arrival

```text
Drive from Paris to Lyon starting at 9:30 AM. Show major towns in between, duration to each, and weather at arrival.
```

If you do **not** provide start time, system assumes **8:00 AM**.

### Restaurants by intermediate towns

```text
From Paris to Lyon, list restaurants town-by-town for intermediate towns only (exclude source and destination). Start time 8 AM.
```

### Restaurants in one area

```text
Find good restaurants near London Bridge.
```

### Calendar event / trip on Google Calendar

```text
Create a calendar event for driving Paris to Lyon tomorrow at 9 AM lasting 4 hours.
```

Or with the orchestrator after a travel question: `add that trip to my calendar`.

### Combined request (best with orchestrator)

```text
Travel from Milan to Rome tomorrow at 8 AM, show towns in between with weather, and give restaurant options in those towns.
```

## 6) One-shot commands (no interactive prompt)

```bash
python run_agents.py "Weather in Tokyo"
python run_agents.py -c travel "Drive from Boston to New York at 7 AM with towns and weather"
python run_agents.py -c restaurants "Restaurants between Boston and Portland, not at endpoints"
python run_agents.py -c calendar "Create calendar event Drive Paris to Lyon tomorrow 9 AM for 4 hours"
```

## 7) Notes (important)

- APIs used are free/public:
  - Nominatim / Overpass / OSRM / wttr.in
- Calendar agent always writes an `.ics` file under `calendar_events/` and a Google Calendar quick-add link (no API key).
- Optional Google Calendar API push: put OAuth desktop `credentials.json` in this folder, then
  `pip install google-api-python-client google-auth-oauthlib google-auth-httplib2`
  and create an event (browser auth saves `token.json`).
- Data may be incomplete or approximate.
- Travel time has no live traffic.
- Always verify restaurant details before visiting.

## 8) LangSmith (debug the agent flow)

Traces show **orchestrator → specialist → tool → HTTP API** (Nominatim, OSRM, wttr.in, Overpass, calendar).

1. Create an API key at [smith.langchain.com](https://smith.langchain.com)
2. Copy `.env.example` to `.env` and set:

```bash
cp .env.example .env
# then edit .env:
# LANGSMITH_TRACING=true
# LANGSMITH_API_KEY=lsv2_pt_...
# LANGSMITH_PROJECT=agentic-ai
```

3. Install extras (already in `requirements.txt`) and run as usual:

```bash
pip install -r requirements.txt
python run_agents.py
# or
streamlit run streamlit_app.py
```

If `LANGSMITH_API_KEY` is set, tracing turns on automatically. CLI prints a one-line status; the Streamlit sidebar shows the same. Open the `agentic-ai` project in LangSmith after a query.

Each agent span records **token usage** from Ollama (`prompt_eval_count` / `eval_count`): input, output, total, and LLM-call count. The orchestrator trace rolls up nested specialists (weather + travel + …). The same line is printed in the CLI and under Streamlit replies.

Set `LANGSMITH_TRACING=false` to disable without removing the key.

## 9) Troubleshooting

- **`Connection refused` or model error**: make sure Ollama is running and model is pulled.
- **No results for route restaurants**: try a bigger city route or less strict filters.
- **No weather data**: check internet connection and try again.

## 10) Project files (quick map)

- `run_agents.py` - main CLI launcher
- `streamlit_app.py` - Streamlit GUI (query + calendar invite card)
- `tracing_common.py` - LangSmith tracing (env + named spans)
- `orchestrator_agent.py` - routes to specialists
- `travel_agent.py` - travel and route-town weather logic
- `weather_agent.py` - weather lookup
- `restaurant_agent.py` - restaurant tools
- `calendar_agent.py` - travel calendar (.ics + Google links / optional API)
- `route_common.py` - shared route/time/weather helpers

# Agentic AI (Simple Guide)

This project is a small multi-agent assistant.
You can ask about:

- Weather
- Travel time
- Towns in between a route
- Weather at those towns at arrival time
- Restaurants (in an area, on route, or by towns on route)

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
ollama pull qwen3.5:9b
```

## 3) Easiest way to run

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

### Combined request (best with orchestrator)

```text
Travel from Milan to Rome tomorrow at 8 AM, show towns in between with weather, and give restaurant options in those towns.
```

## 6) One-shot commands (no interactive prompt)

```bash
python run_agents.py "Weather in Tokyo"
python run_agents.py -c travel "Drive from Boston to New York at 7 AM with towns and weather"
python run_agents.py -c restaurants "Restaurants between Boston and Portland, not at endpoints"
```

## 7) Notes (important)

- APIs used are free/public:
  - Nominatim / Overpass / OSRM / wttr.in
- Data may be incomplete or approximate.
- Travel time has no live traffic.
- Always verify restaurant details before visiting.

## 8) Troubleshooting

- **`Connection refused` or model error**: make sure Ollama is running and model is pulled.
- **No results for route restaurants**: try a bigger city route or less strict filters.
- **No weather data**: check internet connection and try again.

## 9) Project files (quick map)

- `run_agents.py` - main launcher
- `orchestrator_agent.py` - routes to specialists
- `travel_agent.py` - travel and route-town weather logic
- `weather_agent.py` - weather lookup
- `restaurant_agent.py` - restaurant tools
- `route_common.py` - shared route/time/weather helpers

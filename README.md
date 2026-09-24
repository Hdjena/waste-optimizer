# PS-3B: Dynamic Municipal Waste Route Optimization

[![Live Web Application](https://img.shields.io/badge/Streamlit_Cloud-Live_App-A94A4D?style=for-the-badge&logo=streamlit)](https://share.streamlit.io)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Google OR-Tools](https://img.shields.io/badge/OR--Tools-CVRPTW_Solver-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://developers.google.com/optimization)
[![UN SDGs](https://img.shields.io/badge/UN_SDGs-9_%7C_11_%7C_13-008080?style=for-the-badge)](https://sdgs.un.org/)

An end-to-end, context-aware AI waste dispatch and route optimization platform built to address **UN Sustainable Development Goals 9 (Industry, Innovation & Infrastructure), 11 (Sustainable Cities & Communities), and 13 (Climate Action)**.

---

## Executive Overview

Municipal waste collection traditionally operates on fixed calendars, sending heavy diesel collection vehicles to service bins regardless of actual fill levels. This creates operational waste, unnecessary carbon emissions, urban traffic gridlock, and safety risks around school zones during peak morning hours.

Our platform transitions cities from static schedules to  dynamic, demand-driven dispatching:
* **Predictive IoT Fill Scoring:** Uses machine learning to forecast real-time bin fill levels.
* **Context-Aware Routing:** Solves the Capacitated Vehicle Routing Problem with Time Windows (CVRPTW) using Google OR-Tools.
* **Safety & Civic Constraints:** Enforces early morning service time windows for **School-Zone Bins** and dynamically prioritizes **311 Citizen Complaint Hotspot Zones**.
* **OSRM Street Grid Snapping:** Integrates the Open Source Routing Machine (OSRM) API to snap truck routes to real-world street directions instead of straight lines.
* **EV & Financial Scenario Planner:** Calculates operational fuel savings ($) and net emissions reduction (kg CO2) based on dynamic fleet electrification percentages.

---

## Core System Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       INPUT DATA & CITIZEN SIGNAL LAYER                         │
│  • IoT Bin Sensors (Predictive Fill Levels)                                     │
│  • School Zone Spatial GIS Overlay (Morning Traffic Safety Time-Windows)        │
│  • Municipal 311 Citizen Complaint Log (Overflow & Illegal Dumping Tickets)     │
└───────────────────────────────────────┬─────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     DATA ENGINE & PREDICTIVE SCORING (data_engine.py)           │
│  • Random Forest Regressor predicting bin fill percentages (%)                  │
│  • Priority Score = (Predicted Fill %) + (311 Hotspot Weight) + (School Flag)   │
└───────────────────────────────────────┬─────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                   OPTIMIZATION ENGINE & SOLVER (vrp_engine.py)                  │
│  • Google OR-Tools Guided Local Search Metaheuristic                            │
│  • Hard Constraint: Vehicle capacity limits & depot loop                        │
│  • Time Window Constraint: Service school zones prior to 7:30 AM drop-off       │
└───────────────────────────────────────┬─────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    FRONTEND & MAPPING DASHBOARD (app.py)                        │
│  • Streamlit Web UI with custom high-contrast dark theme(.streamlit/config.toml)│
│  • Folium Interactive Map with OSRM API real-world street grid route snapping   │
│  • Human-in-the-Loop Manual Route Override Console                              │
│  • EV Fleet Electrification & Return on Investment (ROI) Scenario Planner       │
└─────────────────────────────────────────────────────────────────────────────────┘
```
Standout Technical Features
1. School-Zone Safety Time Windows
Heavy waste collection vehicles near school corridors pose severe safety hazards during drop-off and pick-up hours. School-zone bins are assigned hard time-window penalties in Google OR-Tools, forcing the solver to route trucks to these locations early in the morning before school zone speed limits and pedestrian surges take effect.

2. 311 Citizen Complaint Hotspot Clusters
To complement IoT fill sensors, the engine ingests municipal 311 overflow reports. Spatial clustering identifies complaint hotspots and inflates the urgency weight of surrounding bins, triggering dynamic insertions into the day's collection route regardless of standard schedule thresholds.

3. OSRM Real-World Street Routing
Instead of straight-line distance approximations, the application connects to the Open Source Routing Machine (OSRM) HTTP API. Latitude and longitude coordinates for each stop are converted into precise, turn-by-turn driving geometries overlaid directly onto Leaflet maps.

4. Human-in-the-Loop Dispatch Console
Automated AI guidance is paired with administrative control. Dispatchers can manually drag, drop, add, or remove bin stops directly from the dashboard to handle emergency street closures or equipment failures on the ground.
## Submission Deliverables & Hackathon Verification

Per the hackathon submission criteria:

| Deliverable | Status | Location / Link |
| :--- | :--- | :--- |
| **Source Code** | Completed | Public GitHub Repository (`/waste-optimizer`) |
| **Working Prototype** | Live | Deployed Streamlit Cloud Web Application |
| **Presentation Deck** | Completed | `10-Slide Pitch Deck (PDF/PPTX)` (10 Slides Maximum) |
| **Impact Statement** | Completed | Embedded below & submitted as 1-Page Document |
```
1-Page Sustainability Impact Statement
UN Sustainable Development Goals Targeted
SDG 9: Industry, Innovation & Infrastructure: Modernizing municipal infrastructure with predictive AI and IoT sensor integration.

SDG 11: Sustainable Cities & Communities: Reducing heavy truck congestion, eliminating urban trash overflow, and ensuring child safety around school zones.

SDG 13: Climate Action: Direct reductions in vehicle miles traveled (VMT), fuel consumption, and greenhouse gas emissions (CO2, NOx, PM2.5).
```
Quantitative Environmental Outcomes

25% – 40% Reduction in Vehicle Miles Traveled (VMT): Eliminating collection runs to bins under the fill threshold drastically cuts total fleet distance.

Emissions Mitigation: Every gallon of heavy-duty diesel saved prevents 10.18 kg of direct CO2 tailpipe emissions.

Accelerated Fleet Electrification: The integrated EV Impact Planner provides city planners with concrete financial ROI projections, demonstrating fuel savings and payback periods for transitioning to Electric Waste Collection Vehicles (eWCVs).



Local Setup & Installation 
To run this repository locally on your machine:
# 1. Clone the repository
git clone [https://github.com/your-username/waste-optimizer.git](https://github.com/your-username/waste-optimizer.git)
cd waste-optimizer

# 2. Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# 3. Install required dependencies
pip install -r requirements.txt

# 4. Launch the Streamlit Dashboard
streamlit run app.py

The web application will automatically open in your default browser at http://localhost:8501.


Tech Stack Dependencies (requirements.txt)
streamlit – Interactive Web Interface

folium & streamlit-folium – Interactive Mapping Overlays

ortools – Google Constraint Programming Solver (CVRPTW)

scikit-learn – Random Forest Predictive Regressor

pandas – Data Manipulation & Time-Series Aggregations

requests – OSRM API Highway & Street Grid Routing

altair & pydeck – Visual Charting & Geospatial Rendering

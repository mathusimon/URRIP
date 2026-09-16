# ============================================================
# PREPARE URIP DATA BUNDLE FOR STREAMLIT
# ============================================================
#
# Run this cell in the SAME Colab notebook/session after all
# URIP analytical cells have successfully run.
#
# It exports the in-memory URIP objects needed by app.py.
# ============================================================

import pickle

urip_data = {
    "G": G,
    "G_traffic": G_traffic,
    "G_combined": G_combined,
    "hospitals_cbd": hospitals_cbd,
    "incidents_wgs84": incidents_wgs84,
    "flood_gdf": flood_gdf,
    "edges_projected": edges_projected,
}

with open("urip_data.pkl", "wb") as f:
    pickle.dump(urip_data, f, protocol=pickle.HIGHEST_PROTOCOL)

print("URIP data bundle created:")
print("  urip_data.pkl")
print()
print("Also place these files beside app.py:")
print("  URIP_Nairobi_Flood_Scenario.tif")
print("  URIP_Nairobi_Flood_Susceptibility.tif")

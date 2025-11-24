import numpy as np
import pandas as pd
from scipy.stats import poisson, uniform, norm

# ------- Step 1: Parameters and Input Data -------

# Region-specific parameters (example values, adapt to real case)
annual_occurrence_rate = 2.0  # avg earthquakes/year
study_years = 50
region_bounds = {
    'lon_min': 110, 'lon_max': 112,
    'lat_min': 34, 'lat_max': 36,
    'depth_min': 5, 'depth_max': 20
}
magnitude_range = [5.0, 7.0]
num_simulations = 10000

# Seismic mechanism proportions (example: strike-slip 0.6, normal 0.2, reverse 0.2)
mechanism_probs = {
    'strike_slip': 0.6,
    'normal': 0.2,
    'reverse': 0.2
}

# ------- Step 2: Monte Carlo Earthquake Simulation -------

def simulate_earthquakes():
    num_events = poisson.rvs(mu=annual_occurrence_rate * study_years)
    events = []
    for _ in range(num_events):
        lon = uniform.rvs(loc=region_bounds['lon_min'], scale=region_bounds['lon_max'] - region_bounds['lon_min'])
        lat = uniform.rvs(loc=region_bounds['lat_min'], scale=region_bounds['lat_max'] - region_bounds['lat_min'])
        depth = uniform.rvs(loc=region_bounds['depth_min'], scale=region_bounds['depth_max'] - region_bounds['depth_min'])
        mag = uniform.rvs(loc=magnitude_range[0], scale=magnitude_range[1] - magnitude_range[0])
        mechanism = np.random.choice(list(mechanism_probs.keys()), p=list(mechanism_probs.values()))
        events.append({'lon': lon, 'lat': lat, 'depth': depth, 'mag': mag, 'mech': mechanism})
    return pd.DataFrame(events)

# ------- Step 3: Source Model Selection -------

def assign_source_model(event):
    if event['mech'] == 'strike_slip':
        # For demonstration, use mixed source for strike-slip, others as point source
        source_type = 'mixed'
    else:
        source_type = 'point'
    return source_type

def simulate_rupture(event, rupture_length=20, rupture_width=10):
    """
    For mixed source, randomly locate rupture point in the rupture area.
    For simplicity, assume rupture area is rectangular, centered at event's (lon, lat), oriented randomly.
    Return rupture location (lon, lat).
    """
    theta = uniform.rvs(loc=0, scale=2 * np.pi)
    dx = (rupture_length / 2) * np.cos(theta)
    dy = (rupture_length / 2) * np.sin(theta)
    return event['lon'] + dx * 0.01, event['lat'] + dy * 0.01  # crude scaling

# ------- Step 4: Spectral Calculation Functions -------

def gmpe_spectrum(event, period_arr):
    """
    Simplified GMPE function for response spectrum (SA) at different periods.
    Replace with real GMPE/attenuation law as needed.
    Uses Haversine formula for more accurate distance calculation.
    """
    mag_scaling = norm.pdf(event['mag'], loc=6.0, scale=0.5)
    depth_scaling = np.exp(-event['depth'] / 15.0)
    # Haversine formula for distance calculation (in km)
    site_lon, site_lat = 111.0, 35.0
    R = 6371  # Earth radius in km
    lon1, lat1 = np.radians(event['lon']), np.radians(event['lat'])
    lon2, lat2 = np.radians(site_lon), np.radians(site_lat)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    distance = R * c  # distance in km
    atten = np.exp(-distance / 60.0)
    spectrum = [mag_scaling * depth_scaling * atten * np.exp(-0.3 * p) for p in period_arr]
    return np.array(spectrum)

# ------- Step 5: Simulation & Uniform Hazard Spectrum (UHS) -------

periods = np.array([0.01, 0.2, 0.5, 1.0, 2.0, 3.0])  # s, can be extended (0.01s for PGA)
all_sa = np.zeros((num_simulations, len(periods)))

for sim in range(num_simulations):
    event_df = simulate_earthquakes()
    sa_max = np.zeros(len(periods))
    for idx, event in event_df.iterrows():
        source_type = assign_source_model(event)
        if source_type == 'mixed':
            # select rupture point randomly
            rlon, rlat = simulate_rupture(event)
            event = event.copy()
            event['lon'], event['lat'] = rlon, rlat
        sa = gmpe_spectrum(event, periods)
        sa_max = np.maximum(sa_max, sa)  # Record maximum shaking in this simulation
    all_sa[sim, :] = sa_max

# ------- Step 6: UHS Calculation (10% PE in 50 years) -------

pe_target = 0.10  # exceedance probability in 50 years
sa_uhs = []
for i, p in enumerate(periods):
    # Get the value exceeded by pe_target simulations
    sa_sorted = np.sort(all_sa[:, i])[::-1]  # descending order
    index = int(pe_target * num_simulations)
    sa_uhs.append(sa_sorted[index])
sa_uhs = np.array(sa_uhs)

# ------- Step 7: Conditional Mean Spectrum (CMS) -------

def calculate_cms(all_sa, periods, target_period_idx):
    """
    For a target control period, compute CMS: the mean spectrum conditional on the selected SA at that period.
    """
    target_sa = sa_uhs[target_period_idx]
    # filter simulations with similar SA at target period
    tolerance = 0.05 * target_sa
    if tolerance == 0:
        tolerance = 0.01  # avoid zero tolerance
    mask = (np.abs(all_sa[:, target_period_idx] - target_sa) <= tolerance)
    mask_array = np.asarray(mask).flatten()  # ensure mask is ndarray
    if np.sum(mask_array) == 0:
        # fallback: if no simulations match, use all simulations
        cms = np.mean(all_sa, axis=0)
    else:
        cms = np.mean(all_sa[mask_array], axis=0)
    return cms

target_period_idx = 2  # e.g., T=0.5s, can be changed
cms_spectrum = calculate_cms(all_sa, periods, target_period_idx)

# ------- Step 8: Output Results -------

print('Uniform Hazard Spectrum (UHS) at 10% PE in 50 years:', dict(zip(periods, sa_uhs)))
print('Conditional Mean Spectrum (CMS) at T=%.2fs:' % periods[target_period_idx], dict(zip(periods, cms_spectrum)))

# The arrays `sa_uhs` and `cms_spectrum` can now be used to select or scale time histories for seismic analysis.

# ------- Step 9: Seismic Wave Selection (conceptual, not implemented) -------

# (In practice, to select actual ground motions matching the CMS, one would search a database for records
# with spectral shape similar to CMS, e.g. using mean squared error or other metrics.)

# ------- End of Script -------

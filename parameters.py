import pandas as pd


# [NOTE] Timestamps denoting the time between which data must have been recorded
EPOCH = pd.Timestamp('2024-12-01 00:00:00')
HORIZON = pd.Timestamp('2025-01-01 00:00:00')

# [NOTE] Weather station metrics to include as node features
WEATHER_FEATURES = ['wind', 'wind_max', 'temperature', 'rain', 'rain_duration', 'fog', 'snow', 'thunder', 'ice']

# [NOTE] Disruption causes deemed unrelated and thus filtered out
DISRUPTION_CAUSE_FILTER = ['police action', 'copper theft', 'technical investigation', 'fire alarm', 'deployment of the fire department',
	'police investigation', 'strike', 'strike of Keolis staff', 'strike of Arriva staff', 'strike of Qbuzz staff', 'strike of Breng staff',
	'staff strikes abroad', 'strike of Connexxion staff', 'strike at ProRail', 'deployment of security staff', 'for security reasons',
	'vandalism', 'an emergency call', 'engineering works', 'over-running engineering works']

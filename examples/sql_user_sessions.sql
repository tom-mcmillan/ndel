-- Aggregated user-level sessions over the last 30 days
SELECT
  user_id,
  COUNT(*) AS sessions,
  COUNT(DISTINCT event_date) AS active_days,
  COUNT(*) * 1.0 / NULLIF(COUNT(DISTINCT event_date), 0) AS sessions_per_day
FROM events
WHERE event_type = 'session_start'
  AND event_date >= CURRENT_DATE - INTERVAL '30' DAY
GROUP BY user_id;

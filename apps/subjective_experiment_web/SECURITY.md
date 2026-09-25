# Security and privacy notes

- Do not collect names, phone numbers, identity numbers, licence numbers or precise addresses.
- Use a strong random `EXPERIMENT_SECRET_KEY` and `EXPERIMENT_ADMIN_TOKEN`.
- Run behind HTTPS and set `EXPERIMENT_SECURE_COOKIE=1` in production.
- Keep the SQLite database outside the public web root, restrict file permissions, and back it up securely.
- Keep the admin route behind an institutional VPN or reverse-proxy access control where possible.
- Do not place Human/AV, inside/outside, deviation side or system/driver identity in public media filenames.
- Treat the full JSON export and the stimulus manifest as restricted research data.
- The app applies signed cookies, CSRF checks, prepared SQL statements, request-size limits, path traversal checks and restrictive browser headers. It is still an application-layer prototype and should receive institutional security review before internet-facing deployment.

"""SQLAlchemy model package.

Model registration is centralized in :mod:`app.db.base`.
Keep this package initializer free of eager model imports so importing a single
model module (for example from a standalone seed script) cannot leave string
relationships half-registered.
"""

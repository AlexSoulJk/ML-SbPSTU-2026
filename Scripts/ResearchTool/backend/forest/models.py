from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class ImportBatch(Base):
    __tablename__ = "forest_import_batches"

    import_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    stored_path: Mapped[str] = mapped_column(Text, nullable=False)
    source_format: Mapped[str] = mapped_column(String(32), nullable=False)
    imported_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    duplicate_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    rejected_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    errors_json: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class Sample(Base):
    __tablename__ = "forest_samples"

    sample_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    source_file: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    source_format: Mapped[str] = mapped_column(String(32), nullable=False)
    source_id: Mapped[str | None] = mapped_column(String(255), index=True)
    display_name: Mapped[str | None] = mapped_column(String(255), index=True)
    lat: Mapped[float] = mapped_column(Float, nullable=False, index=True)
    lon: Mapped[float] = mapped_column(Float, nullable=False, index=True)
    source_event_year: Mapped[int | None] = mapped_column(Integer, index=True)
    driver_primary: Mapped[str | None] = mapped_column(String(255), index=True)
    driver_primary_code: Mapped[str | None] = mapped_column(String(64))
    confidence_primary: Mapped[str | None] = mapped_column(String(64), index=True)
    driver_secondary: Mapped[str | None] = mapped_column(String(255))
    confidence_secondary: Mapped[str | None] = mapped_column(String(64))
    region: Mapped[str | None] = mapped_column(String(255), index=True)
    region_code: Mapped[str | None] = mapped_column(String(64))
    tags: Mapped[str | None] = mapped_column(Text)
    notes: Mapped[str | None] = mapped_column(Text)
    raw_properties_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
    )

    statuses: Mapped[list["SampleStatus"]] = relationship(
        back_populates="sample",
        cascade="all, delete-orphan",
    )
    manual_validation: Mapped["ManualValidation | None"] = relationship(
        back_populates="sample",
        cascade="all, delete-orphan",
    )


class SampleAssignment(Base):
    __tablename__ = "forest_sample_assignments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sample_id: Mapped[str] = mapped_column(ForeignKey("forest_samples.sample_id"), index=True)
    assignment_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    assignment_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class SampleStatus(Base):
    __tablename__ = "forest_sample_statuses"
    __table_args__ = (UniqueConstraint("sample_id", "source", name="uq_forest_sample_status"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sample_id: Mapped[str] = mapped_column(ForeignKey("forest_samples.sample_id"), index=True)
    source: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    reason: Mapped[str | None] = mapped_column(Text)
    metrics_json: Mapped[str | None] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
    )

    sample: Mapped[Sample] = relationship(back_populates="statuses")


class ManualValidation(Base):
    __tablename__ = "forest_manual_validations"

    sample_id: Mapped[str] = mapped_column(ForeignKey("forest_samples.sample_id"), primary_key=True)
    validation: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    notes: Mapped[str | None] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
    )

    sample: Mapped[Sample] = relationship(back_populates="manual_validation")


class HansenAnalysis(Base):
    __tablename__ = "forest_hansen_analyses"

    analysis_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    sample_id: Mapped[str] = mapped_column(ForeignKey("forest_samples.sample_id"), index=True)
    analysis_area_type: Mapped[str] = mapped_column(String(64), nullable=False)
    plot_size_m: Mapped[float] = mapped_column(Float, nullable=False)
    treecover_threshold: Mapped[float | None] = mapped_column(Float)
    event_year: Mapped[int | None] = mapped_column(Integer, index=True)
    total_loss_area_ha: Mapped[float | None] = mapped_column(Float)
    dominant_loss_area_ha: Mapped[float | None] = mapped_column(Float)
    dominant_year_share: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    histogram_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class HansenTile(Base):
    __tablename__ = "forest_hansen_tiles"
    __table_args__ = (
        UniqueConstraint("layer", "tile_id", "version", name="uq_forest_hansen_tile"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    layer: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    tile_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    version: Mapped[str] = mapped_column(String(64), nullable=False)
    local_path: Mapped[str] = mapped_column(Text, nullable=False)
    downloaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class HansenMask(Base):
    __tablename__ = "forest_hansen_masks"

    mask_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    sample_id: Mapped[str] = mapped_column(ForeignKey("forest_samples.sample_id"), index=True)
    analysis_id: Mapped[str] = mapped_column(
        ForeignKey("forest_hansen_analyses.analysis_id"),
        index=True,
    )
    mask_type: Mapped[str] = mapped_column(String(64), nullable=False)
    local_path: Mapped[str] = mapped_column(Text, nullable=False)
    bbox_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class SentinelSearch(Base):
    __tablename__ = "forest_sentinel_searches"

    search_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    sample_id: Mapped[str] = mapped_column(ForeignKey("forest_samples.sample_id"), index=True)
    period: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    event_year: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    query_json: Mapped[str] = mapped_column(Text, nullable=False)
    results_json: Mapped[str] = mapped_column(Text, nullable=False)
    cache_path: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class SentinelDownload(Base):
    __tablename__ = "forest_sentinel_downloads"

    download_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    sample_id: Mapped[str] = mapped_column(ForeignKey("forest_samples.sample_id"), index=True)
    scene_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    period: Mapped[str | None] = mapped_column(String(32), index=True)
    local_path: Mapped[str] = mapped_column(Text, nullable=False)
    bbox_json: Mapped[str] = mapped_column(Text, nullable=False)
    bands_json: Mapped[str] = mapped_column(Text, nullable=False)
    cloud_fraction: Mapped[float | None] = mapped_column(Float)
    shadow_fraction: Mapped[float | None] = mapped_column(Float)
    is_bad_cloud: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    metadata_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class DerivedPreview(Base):
    __tablename__ = "forest_derived_previews"

    preview_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    download_id: Mapped[str] = mapped_column(
        ForeignKey("forest_sentinel_downloads.download_id"),
        index=True,
    )
    view: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    png_path: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_path: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class ForestExport(Base):
    __tablename__ = "forest_exports"

    export_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    scope: Mapped[str] = mapped_column(String(64), nullable=False)
    csv_path: Mapped[str] = mapped_column(Text, nullable=False)
    geojson_path: Mapped[str] = mapped_column(Text, nullable=False)
    sample_count: Mapped[int] = mapped_column(Integer, nullable=False)
    filters_json: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
